#!/usr/bin/env python3
"""Run one OpenShorts clipping job against a local API and collect the clips.

Used by .github/workflows/clip-video.yml, but works against any self-hosted
instance: ``python run_clip_job.py --source video.mp4 --out clips/``.

Steps: optionally cut the source to [start, end], re-encode it when it is over
the upload limit, upload it to /api/process, follow /api/status until the job
ends, then download every clip and write metadata.json + titles.md next to
them. Standard library only; needs curl, ffmpeg and ffprobe on PATH. Never
reads or prints a Gemini key: the API takes it from its own environment.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

# The API refuses uploads over MAX_FILE_SIZE_MB (2048); keep a margin.
UPLOAD_LIMIT_BYTES = int(1.9 * 1024 ** 3)
TERMINAL = {"completed", "failed", "cancelled", "error"}
# Consecutive failed status checks (20 s apart) before the backend counts as
# dead, e.g. the container was OOM-killed: 5 minutes instead of the deadline.
MAX_STATUS_FAILURES = 15
LAYOUTS = {"streamer", "general", "auto"}


def fail(msg):
    # ::error:: makes GitHub show the line on the run page; harmless elsewhere.
    print(f"::error::{msg}", flush=True)
    sys.exit(1)


def parse_time(value):
    """Seconds from '', '754', '12:34' or '1:02:03'. None when blank."""
    value = (value or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?|\d+(?::\d{1,2}){1,2}", value):
        fail(f"Bad time {value!r}: use seconds (754) or [HH:]MM:SS (12:34, 1:02:03).")
    if ":" not in value:
        return float(value)
    seconds = 0
    for part in value.split(":"):
        seconds = seconds * 60 + int(part)
    return float(seconds)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    height = int((info.get("streams") or [{}])[0].get("height") or 0)
    return float(info["format"]["duration"]), height


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def prepare(source, start, end, workdir):
    """(file to upload, its offset in the source in seconds): cut to
    [start, end] and re-encoded only if needed."""
    duration, height = probe(source)
    offset = 0.0
    print(f"Source: {duration / 60:.1f} min, {height}p, "
          f"{os.path.getsize(source) / 1024 ** 3:.2f} GB", flush=True)
    path = source
    if start is not None or end is not None:
        s = start or 0.0
        e = min(end, duration) if end is not None else duration
        if e <= s:
            fail(f"End ({e:.0f}s) must be after start ({s:.0f}s) and inside the "
                 f"video ({duration:.0f}s long).")
        cut = os.path.join(workdir, "cut.mp4")
        print(f"Cutting {s:.0f}s-{e:.0f}s", flush=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", str(s), "-i", path, "-t", str(e - s),
                        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                        "-avoid_negative_ts", "make_zero", cut], check=True)
        path = cut
        duration, height = probe(path)
        # A stream-copy cut starts at the keyframe before ``s``, so the real
        # offset is where the cut ends minus how long it came out.
        offset = max(0.0, e - duration)
    if os.path.getsize(path) > UPLOAD_LIMIT_BYTES or height > 1080:
        small = os.path.join(workdir, "upload.mp4")
        print("Re-encoding to 1080p30 H.264 (CRF 23) to fit the 2 GB upload limit",
              flush=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
                        "-y", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
                        "-vf", "scale=-2:'min(1080,ih)',fps=30",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                        "-movflags", "+faststart", small], check=True)
        path = small
        if os.path.getsize(path) > UPLOAD_LIMIT_BYTES:
            fail("Still over 2 GB after re-encoding: set start/end to clip a "
                 "shorter stretch (about 60 minutes or less per run).")
    return path, offset


def check_manifest(manifest_path, current):
    """Stop before a transcript is laid over a different video or stretch:
    the captions and cut points would all be off."""
    try:
        with open(manifest_path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"The earlier run's transcript has no readable manifest ({e}).")
    problems = []
    if str(old.get("video_url", "")).strip() != current["video_url"].strip():
        problems.append("a different video link")
    for key in ("start", "end"):
        if old.get(key) != current[key]:
            problems.append(f"a different {key} ({old.get(key)!r} vs {current[key]!r})")
    try:
        if abs(float(old.get("duration", -1)) - current["duration"]) > 1.0:
            problems.append(f"a different length ({float(old['duration']):.0f}s vs "
                            f"{current['duration']:.0f}s)")
    except (TypeError, ValueError):
        problems.append("no recorded length")
    if problems:
        fail("The transcript from that run was made for " + ", ".join(problems)
             + ". Use the same video_url, start and end as that run, or leave "
               "transcript_from_run empty to transcribe again.")


def submit(api, path, layout, clips, transcript=None):
    fields = {
        "output_format": "vertical",
        "clip_min_seconds": "20",
        "clip_max_seconds": "60",
        "auto_hook": "1",
        "auto_hook_style": "pill",
        "target_clips": str(clips),
        "acknowledged": "1",
    }
    if layout != "auto":
        fields["layouts"] = layout
    cmd = ["curl", "-sS", "--fail-with-body", "--max-time", "3600",
           "-X", "POST", f"{api}/api/process",
           "-F", f"file=@{path};type=video/mp4"]
    if transcript:
        cmd += ["-F", f"transcript=@{transcript};type=application/json"]
    for k, v in fields.items():
        cmd += ["--form-string", f"{k}={v}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        fail(f"Upload to OpenShorts failed: {(res.stdout or res.stderr)[:500]}")
    job_id = json.loads(res.stdout).get("job_id")
    if not job_id:
        fail(f"OpenShorts returned no job id: {res.stdout[:500]}")
    print(f"Job {job_id} queued (layout={layout}, clips={clips})", flush=True)
    return job_id


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def follow(api, job_id, stop_at):
    seen = 0
    failures = 0
    while time.time() < stop_at:
        try:
            st = get_json(f"{api}/api/status/{job_id}")
            failures = 0
        except Exception as e:  # one slow answer must not end the run
            failures += 1
            print(f"(status check failed: {e})", flush=True)
            if failures >= MAX_STATUS_FAILURES:
                fail("The OpenShorts backend stopped answering (it may have run "
                     "out of memory): see the backend log below.")
            time.sleep(20)
            continue
        logs = st.get("logs") or []
        for line in logs[seen:]:
            print(f"  | {line}", flush=True)
        seen = len(logs)
        status = st.get("status")
        if status in TERMINAL:
            return status, st.get("result") or {}
        time.sleep(20)
    fail("Job did not finish in the time this run has left: use start/end "
         "to clip a shorter stretch.")


def slug(text, limit=50):
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-")
    return (s[:limit].rstrip("-") or "clip")


def download(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1024 * 1024)


def collect(api, result, out, offset=0.0):
    clips = [c for c in (result.get("clips") or []) if isinstance(c, dict)]
    if not clips:
        fail("The job finished without clips.")
    os.makedirs(out, exist_ok=True)
    lines = ["# Clips", ""]
    saved = []
    for i, clip in enumerate(clips, 1):
        url = clip.get("video_url") or ""
        if not url:
            continue
        full = url if url.startswith("http") else api + url
        title = clip.get("video_title_for_youtube_short") or clip.get("title") or ""
        ext = os.path.splitext(urllib.parse.urlparse(full).path)[1] or ".mp4"
        name = f"{i:02d}_{slug(title)}{ext}"
        download(full, os.path.join(out, name))
        clip["file"] = name
        saved.append(clip)
        start, end = clip.get("start"), clip.get("end")
        span = ""
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            # Times are in the uploaded file; shift them back onto the VOD.
            clip["vod_start"] = round(offset + start, 2)
            clip["vod_end"] = round(offset + end, 2)
            span = f"{hms(clip['vod_start'])}-{hms(clip['vod_end'])} in the VOD"
        warn = " ⚠ strong language" if clip.get("strong_language") else ""
        lines += [f"## {i}. {title}{warn}", "",
                  f"- File: `{name}`" + (f" ({span})" if span else ""),
                  f"- Hook: {clip.get('viral_hook_text') or ''}",
                  f"- TikTok: {clip.get('video_description_for_tiktok') or ''}",
                  f"- Instagram: {clip.get('video_description_for_instagram') or ''}",
                  ""]
        print(f"Saved {name}", flush=True)
    if not saved:
        fail("The job finished but none of its clips could be downloaded.")
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)
    md = "\n".join(lines)
    with open(os.path.join(out, "titles.md"), "w", encoding="utf-8") as f:
        f.write(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return len(saved)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default=".")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--layout", default="streamer")
    p.add_argument("--clips", default="8")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--video-url", default="",
                   help="the source link, recorded with the transcript")
    p.add_argument("--transcript", default="",
                   help="transcript.json from an earlier run: skips transcription")
    p.add_argument("--transcript-manifest", default="",
                   help="manifest.json saved with that transcript")
    p.add_argument("--state-dir", default="",
                   help="where to write job.json (job id + manifest) for saving the "
                        "transcript afterwards")
    p.add_argument("--deadline-minutes", type=int, default=300,
                   help="minutes from now the job may take, unless --finish-by")
    p.add_argument("--finish-by", type=float, default=0,
                   help="epoch seconds by which the job must be done (the "
                        "workflow passes its own budget, so download, build "
                        "and re-encode time count too)")
    a = p.parse_args(argv)

    layout = a.layout.strip().lower()
    if layout not in LAYOUTS:
        fail(f"layout must be one of {sorted(LAYOUTS)}, got {a.layout!r}")
    try:
        clips = int(a.clips)
    except ValueError:
        fail(f"clip_count must be a whole number, got {a.clips!r}")
    if not 1 <= clips <= 15:
        fail("clip_count must be between 1 and 15.")

    stop_at = a.finish_by or (time.time() + a.deadline_minutes * 60)
    path, offset = prepare(a.source, parse_time(a.start), parse_time(a.end), a.workdir)
    # Seconds, so "35:00" and "2100" count as the same start.
    manifest = {"video_url": a.video_url, "start": parse_time(a.start), "end": parse_time(a.end),
                "duration": round(probe(path)[0], 2)}
    if a.transcript:
        if not os.path.exists(a.transcript):
            fail("transcript_from_run is set but that run has no transcript "
                 "artifact (it may have failed before transcribing, or expired).")
        check_manifest(a.transcript_manifest, manifest)
        print("Reusing the earlier run's transcript: no transcription this time", flush=True)
    job_id = submit(a.api.rstrip("/"), path, layout, clips, a.transcript or None)
    if a.state_dir:
        os.makedirs(a.state_dir, exist_ok=True)
        with open(os.path.join(a.state_dir, "job.json"), "w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "manifest": manifest}, f)
    status, result = follow(a.api.rstrip("/"), job_id, stop_at)
    if status != "completed":
        fail(f"OpenShorts job ended as '{status}' (see the log above).")
    n = collect(a.api.rstrip("/"), result, a.out, offset)
    print(f"Done: {n} clip(s) in {a.out}", flush=True)


if __name__ == "__main__":
    main()
