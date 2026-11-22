# Clip Studio

Turn long streams and podcasts into vertical 9:16 Shorts, Reels and TikToks.

## What it does

- **Picks the moments.** Gemini reads the transcript (or watches silent gameplay)
  and picks the best 20–60 s stretches. It also writes titles, descriptions and
  hook text for each one.
- **Frames them vertically.** It follows the speaker, stacks two speakers, keeps
  a screen readable, or uses one of the stream layouts below.
- **Captions and hooks.** Word-by-word captions are burned in, and a hook title
  is placed where it doesn't cover a face.
- **Runs locally or on GitHub.** Use the web dashboard on your own machine, or
  run the "Clip a video" workflow on GitHub Actions without opening a UI.

## Run locally

Requires Docker Desktop with about 8 GB of memory available.

```bash
cp .env.example .env          # then add GEMINI_API_KEY=... (or paste it in Settings later)
docker compose up --build
```

Open http://localhost:5175, add your Gemini key in **Settings** if it's not in
`.env`, then upload a video (up to 2 GB) or paste a link. The API runs on
http://localhost:8000.

Optional `.env` settings:

| Variable | What it does |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini key |
| `GEMINI_MODEL_CLIPS` | Model for moment picking, e.g. `gemini-3.6-flash` |
| `GEMINI_MODEL_CLIPS_FALLBACK` | Comma-separated models to try when that one is busy, in order (default `gemini-3.5-flash,gemini-3.7-flash,gemini-3.1-flash-lite`) |
| `GEMINI_MODEL` | Model for everything else (default `gemini-3.1-flash-lite`) |
| `CONTENT_FILTER` | `0` turns the content filter off (on by default) |
| `CONTENT_FILTER_EXTRA` | Your own comma-separated words, treated like slurs |
| `SCREEN_CHECK` | `0` keeps clips that show a desktop or stream dashboard (stream layouts only; `1` checks every job) |
| `DARK_LIFT` | `0` turns off the brightness lift for dark clips |

When a model is busy (503 "high demand", 429, 500 or a timeout), it gets one
retry, then the next model in the list takes over. If every model is busy, the
job waits 60 s, 120 s and 240 s and goes through the list again each time
before it gives up. The log names the model that answered.

## Content filter

On by default, so clips are safe to post:

- A clip is skipped if anyone in it uses a racial or other slur. The check
  covers the words spoken, which are also the captions, and the generated text.
- Strong swear words are masked in the burned-in captions (`f***`, `s***`,
  `b****`). The audio isn't changed.
- Swear words and slurs are removed from titles, descriptions and hooks.
- In the workflow's `titles.md`, clips with strong language are marked
  **⚠ strong language**.

The slur list is stored only as hashes. A word list can miss misspellings and
words the speech-to-text misheard, so watch clips before posting them.

## Run on GitHub Actions

The **Clip a video** workflow (`.github/workflows/clip-video.yml`) runs the
whole pipeline on a GitHub runner. It builds the backend image, processes the
video and uploads the clips as an artifact you can download for 7 days.

1. Add the repository secrets (**Settings → Secrets and variables → Actions**):
   - `GEMINI_API_KEY` (required)
   - `GEMINI_MODEL_CLIPS` (optional, e.g. `gemini-3.7-flash`)
2. Open **Actions → Clip a video → Run workflow** and fill in:

| Input | Default | Notes |
|---|---|---|
| `video_url` | | A Kick, YouTube or Google Drive link. Drive files must be shared as "Anyone with the link". |
| `layout` | `streamer` | `streamer`, `general` or `auto` (see Layouts) |
| `clip_count` | `8` | 1–15 |
| `start` / `end` | | Optional, in seconds or `[HH:]MM:SS`. Use these to clip part of a long VOD. |
| `creator_name` | | Optional. Titles and hooks lead with this name, e.g. "Dasgas Dom runs into TIGERS". |
| `facecam_box` | | Optional, `streamer` layout only: the webcam's `x,y,w,h` in percent of the 16:9 frame, e.g. `0,0,20,35` for a top-left box 20% wide and 35% tall. Use it when the webcam isn't found. Every scene is then stacked with that box. |
| `transcript_from_run` | | Optional run ID of an earlier run with the same link, start and end. That run's transcript is reused, so transcription is skipped. |

3. Download the `clips-…` artifact. It contains the clips, `titles.md` (titles,
   descriptions, hook text and timestamps in the original video) and
   `metadata.json`.

Every run also saves a `transcript-…` artifact, even when it fails. If a run
dies after transcribing (for example, when every Gemini model is busy), start a
new run with the same inputs and set `transcript_from_run` to the failed run's
ID. The run ID is the number at the end of the run's URL.

If Kick or YouTube blocks GitHub's servers, the run stops and tells you to use a
Google Drive link instead. If this repository is public, anyone signed in to
GitHub can download its artifacts and read its logs. That includes the
`transcript-…` artifact, which holds the full transcript of the video.

## Stream VODs

With the `streamer` or `general` layout:

- Clips that mostly show a desktop, stream dashboard or chat instead of the
  game are skipped. The picker replaces those moments with gameplay before it
  picks, so you still get your clip count when the VOD has enough gameplay.
- Dark clips (night scenes) get a mild brightness and contrast lift.
- Titles and hooks never name a streaming platform the stream didn't mention.

## Layouts

| Layout | Best for | What you get |
|---|---|---|
| **Stream: facecam over gameplay** (`streamer`) | Stream VODs with a webcam overlay | The webcam on the top half and a centre crop of the gameplay on the bottom, with captions on the seam. A small webcam box in a corner is found and scaled up. The gameplay crop steps sideways if it would cut through the webcam. Scenes without the webcam use the blurred layout. |
| **Stream: blurred gameplay** (`general`) | Gameplay with or without a facecam | The whole 16:9 frame over a blurred copy of itself |
| **Auto** | Podcasts, talking heads, tutorials | The AI picks per video: follow the speaker, stack two speakers, or show a screen over the presenter |
| Two speakers stacked | Two-person podcasts | One speaker per half |
| Screen over presenter | Tutorials, screen shares | The screen at full width above the speaker |
| Single crop only | Anything | A plain 9:16 crop that follows the main face |

Pick the layout under **advanced options** in the dashboard. In the workflow,
use the `layout` input.

---

Based on OpenShorts (MIT License; the cloud/ folder is under its own license, see NOTICE).
