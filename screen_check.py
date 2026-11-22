"""Keep stream clips on the game, not on the streamer's desktop.

A stream VOD often opens (and breaks) with the streamer talking over a stream
dashboard, a chat window or a browser. The transcript of those stretches is
lively, so the moment picker likes them, and the clips come out as a webcam
over a wall of chat (seen on a GTA VOD: 7 of 11 clips were the dashboard).

So, on stream VODs (a job with a stream layout, STREAM_LAYOUT), a few frames
of each candidate go to Gemini with one closed question: gameplay, desktop,
camera or other, per frame. Twice:

- on the scored windows, before the detail pass, so the shortlist is filled
  from gameplay windows and the job still gets its clip count;
- on the final clips, which drops the ones where most frames are a desktop.

SCREEN_CHECK=0 turns it off, SCREEN_CHECK=1 turns it on for any job. Never
raises: a frame or model problem keeps the candidate (fail open), a clip is
only ever dropped on a clear answer.
"""
from __future__ import annotations

import enum
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from pydantic import BaseModel

FRAMES = int(os.environ.get("SCREEN_CHECK_FRAMES", "4"))
WIDTH = int(os.environ.get("SCREEN_CHECK_WIDTH", "640"))
WORKERS = 4


class Screen(str, enum.Enum):
    gameplay = "gameplay"
    desktop = "desktop"
    camera = "camera"
    other = "other"


class ScreenLabels(BaseModel):
    frames: List[Screen]


PROMPT = """These are {n} frames, in order, from a live stream recording.
Label each frame with exactly one of:
- gameplay: a game is on screen (the game world, or the game's own menus, \
shops, maps, inventory or loading screens), with or without a webcam box.
- desktop: most of the frame is computer UI outside the game: a stream \
dashboard, a chat window, a browser, streaming software, social media, a video \
player or the desktop itself.
- camera: the streamer's webcam fills most of the frame, with no game or \
desktop around it.
- other: anything else (a "be right back" card, a black frame).
Judge by what fills most of the frame. Return one label per frame, in order."""


def enabled() -> bool:
    flag = os.environ.get("SCREEN_CHECK", "").strip()
    if flag == "0":
        return False
    if flag == "1":
        return True
    return os.environ.get("STREAM_LAYOUT", "").strip().lower() in ("general", "streamer")


def sample_times(start, end, n=None):
    """``n`` timestamps spread over [start, end], never on its edges."""
    n = n or FRAMES
    start, end = float(start), float(end)
    span = max(0.0, end - start)
    return [start + span * (i + 0.5) / n for i in range(n)]


def mostly_desktop(labels) -> bool:
    """True when more than half the labelled frames are a desktop."""
    labels = [str(getattr(l, "value", l)) for l in (labels or [])]
    return bool(labels) and labels.count("desktop") * 2 > len(labels)


def _ask_gemini(frames, prompt, api_key):
    """One vision call; returns the list of labels. Split out for tests."""
    from google import genai
    from google.genai import types as genai_types
    import gemini_worker

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"
    parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]
    response = client.models.generate_content(
        model=model_name,
        contents=parts + [prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScreenLabels,
        ))
    gemini_worker.raise_if_blocked(response)
    return (json.loads(response.text) or {}).get("frames") or []


def labels(video_path, start, end) -> Optional[list]:
    """Per-frame labels for source seconds [start, end], or None on failure."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    import hook_grounding
    frames = hook_grounding.frames_at(video_path, sample_times(start, end), width=WIDTH)
    if not frames:
        return None
    prompt = PROMPT.format(n=len(frames))
    for attempt in range(2):
        try:
            out = [str(l) for l in _ask_gemini(frames, prompt, api_key)]
            return out if out else None
        except Exception as e:
            if attempt:
                print(f"   ⚠️ Screen check failed ({type(e).__name__}: {e}) — keeping it.")
                return None
            time.sleep(5)
    return None


def _classify(video_path, spans):
    """labels() for each (start, end), in parallel, in order."""
    if not spans:
        return []
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(spans))) as pool:
        return list(pool.map(lambda s: labels(video_path, s[0], s[1]), spans))


def keep_game_windows(video_path, shortlist, ranked, target):
    """``shortlist`` without its desktop windows, topped back up to ``target``
    from ``ranked`` (best first), checking the next ones as it goes. Returns
    the shortlist unchanged when every window checked is a desktop: the clip
    check then says so instead of the job failing with no clips at all."""
    checked, kept = set(), []

    def run(batch):
        for w, got in zip(batch, _classify(video_path, [(w["start"], w["end"]) for w in batch])):
            checked.add(w["id"])
            if not mostly_desktop(got):
                kept.append(w)

    run(list(shortlist))
    dropped = len(shortlist) - len(kept)
    spare = [w for w in ranked if w["id"] not in {s["id"] for s in shortlist}]
    budget = max(4, 2 * int(target))  # windows checked beyond the shortlist, at most
    while len(kept) < target and spare and budget > 0:
        size = min(budget, target - len(kept))
        batch, spare = spare[:size], spare[size:]
        budget -= len(batch)
        run(batch)
    if not kept:
        print(f"   🖥️ All {len(checked)} window(s) checked show a desktop; keeping the shortlist.")
        return shortlist
    if dropped:
        print(f"   🖥️ {dropped} shortlisted window(s) show a desktop, not the game: "
              f"replaced from the next best ({len(kept)} window(s) now).")
    kept.sort(key=lambda w: float(w.get("start") or 0))
    return kept


def drop_desktop_clips(clips, video_path):
    """The clips whose frames are mostly NOT a desktop; ``clip['screen']``
    keeps the labels of each kept clip."""
    got = _classify(video_path, [(c["start"], c["end"]) for c in clips])
    kept = []
    for clip, frames in zip(clips, got):
        if mostly_desktop(frames):
            print(f"   🖥️ Skipping {clip.get('video_title_for_youtube_short', 'a clip')!r}: "
                  f"it shows a desktop, not the game ({', '.join(frames)}).")
            continue
        if frames:
            clip["screen"] = frames
        kept.append(clip)
    return kept
