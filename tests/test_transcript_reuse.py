"""A transcript from an earlier run can be sent with an upload so the job skips
transcription; the workflow helper refuses one made for another video/stretch."""
import asyncio
import importlib.util
import json
import os

import pytest

HELPER = os.path.join(os.path.dirname(__file__), "..", ".github", "scripts", "run_clip_job.py")


def _helper():
    spec = importlib.util.spec_from_file_location("run_clip_job", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Upload:
    """Just enough of starlette's UploadFile: async chunked read()."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    async def read(self, n=-1):
        chunk = self.data[self.pos:self.pos + n] if n > 0 else self.data[self.pos:]
        self.pos += len(chunk)
        return chunk


class TestUploadedTranscript:
    @pytest.fixture
    def app_module(self):
        return pytest.importorskip("app")

    def _save(self, app_module, data, job_dir):
        return asyncio.run(app_module._save_uploaded_transcript(_Upload(data), str(job_dir)))

    def test_valid_transcript_is_written_for_main(self, app_module, tmp_path):
        good = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi",
                                                "words": [{"word": " hi", "start": 0, "end": 1}]}]}
        path = self._save(app_module, json.dumps(good).encode(), tmp_path / "job")
        assert path.endswith("source_transcript.json")
        with open(path) as f:
            assert json.load(f) == good

    @pytest.mark.parametrize("data,status", [
        (b"{not json", 400),
        (b"\xff\xfe", 400),
        (b"[1, 2]", 400),
        (json.dumps({"segments": []}).encode(), 400),
        (json.dumps({"language": "en"}).encode(), 400),
    ])
    def test_bad_transcripts_are_refused(self, app_module, tmp_path, data, status):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as e:
            self._save(app_module, data, tmp_path / "job")
        assert e.value.status_code == status

    def test_oversized_transcript_is_refused(self, app_module, tmp_path, monkeypatch):
        from fastapi import HTTPException
        monkeypatch.setattr(app_module, "MAX_TRANSCRIPT_UPLOAD_BYTES", 10)
        with pytest.raises(HTTPException) as e:
            self._save(app_module, b'{"segments": [1, 2, 3]}', tmp_path / "job")
        assert e.value.status_code == 413


class TestTranscriptNeedsAnUpload:
    def test_transcript_with_a_link_is_refused(self, tmp_path):
        app_module = pytest.importorskip("app")
        testclient = pytest.importorskip("fastapi.testclient")
        client = testclient.TestClient(app_module.app)
        r = client.post("/api/process",
                        data={"url": "https://www.youtube.com/watch?v=abc"},
                        files={"transcript": ("t.json", b'{"segments": [1]}', "application/json")})
        assert r.status_code == 400
        assert "file upload" in r.json()["detail"]


class TestManifestCheck:
    CURRENT = {"video_url": "https://kick.com/x/videos/1", "start": 2100.0,
               "end": 4200.0, "duration": 2101.3}

    def _check(self, tmp_path, old):
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(old))
        _helper().check_manifest(str(p), dict(self.CURRENT))

    def test_same_video_and_stretch_passes(self, tmp_path):
        self._check(tmp_path, dict(self.CURRENT, duration=2100.6))  # within 1 s

    @pytest.mark.parametrize("change", [
        {"video_url": "https://kick.com/x/videos/2"},
        {"start": 0.0},
        {"end": None},
        {"duration": 1800.0},
    ])
    def test_other_video_or_stretch_is_refused(self, tmp_path, change, capsys):
        with pytest.raises(SystemExit):
            self._check(tmp_path, dict(self.CURRENT, **change))
        assert "::error::" in capsys.readouterr().out

    def test_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            _helper().check_manifest(str(tmp_path / "none.json"), dict(self.CURRENT))

    def test_times_compare_as_seconds(self):
        helper = _helper()
        assert helper.parse_time("35:00") == helper.parse_time("2100") == 2100.0


class TestMainSavesReusedTranscript:
    """A run that reused a transcript leaves the same checkpoint a fresh
    transcription does, so its transcript artifact can be reused again."""

    def test_checkpoint_round_trip(self, tmp_path):
        main = pytest.importorskip("main")
        transcript = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi"}]}
        main.save_transcript_checkpoint(str(tmp_path), transcript, "/x/upload.mp4", 2101.3)
        with open(tmp_path / main.TRANSCRIPT_CHECKPOINT) as f:
            assert json.load(f)["transcript"] == transcript


class TestSectionDownload:
    """Kick/YouTube with start/end: the workflow downloads only that section,
    the helper does not cut it again, and the manifest and titles.md still
    speak in the original VOD's link and times."""

    WORKFLOW = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows",
                            "clip-video.yml")

    @pytest.mark.parametrize("start,end,spec", [
        ("35:00", "1:10:00", "*2100-4200"),
        ("35:00", "", "*2100-inf"),
        ("", "600", "*0-600"),
        ("", "", None),
    ])
    def test_section_spec(self, start, end, spec):
        h = _helper()
        assert h.section_spec(h.parse_time(start), h.parse_time(end)) == spec

    def test_end_before_start_is_refused(self, capsys):
        h = _helper()
        with pytest.raises(SystemExit):
            h.section_spec(600.0, 500.0)
        assert "::error::" in capsys.readouterr().out

    @pytest.mark.parametrize("start,end,duration,offset", [
        (2100.0, 4200.0, 2101.8, 2098.2),   # starts at the keyframe 1.8 s early
        (60.0, 90.0, 30.033, 59.967),
        (2100.0, 99999.0, 3000.0, 2100.0),  # end past the VOD: measured is nonsense
        (2100.0, None, 5000.0, 2100.0),     # open end: the requested start
        (None, 600.0, 600.0, 0.0),
    ])
    def test_section_offset(self, start, end, duration, offset):
        assert _helper().section_offset(start, end, duration) == pytest.approx(offset, abs=0.01)

    def _run(self, tmp_path, monkeypatch, extra, source_seconds):
        """main() with the API, ffmpeg and ffprobe stubbed: returns (ffmpeg
        commands, the state manifest, titles.md)."""
        h = _helper()
        src = tmp_path / "source.mp4"
        src.write_bytes(b"x" * 1000)
        ffmpeg = []

        def fake_run(cmd, **kw):
            ffmpeg.append(cmd)
            open(cmd[-1], "wb").write(b"x" * 1000)

            class R:
                returncode = 0
            return R()
        monkeypatch.setattr(h.subprocess, "run", fake_run)
        monkeypatch.setattr(h, "probe", lambda path: (source_seconds if path == str(src) else 2100.0, 1080))
        monkeypatch.setattr(h, "submit", lambda *a, **k: "job1")
        result = {"clips": [{"video_url": "/videos/job1/a.mp4", "start": 10.0, "end": 40.0,
                             "video_title_for_youtube_short": "Tigers"}]}
        monkeypatch.setattr(h, "follow", lambda *a: ("completed", result))
        monkeypatch.setattr(h, "download", lambda url, dest: open(dest, "wb").write(b"v"))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        out, state = tmp_path / "clips", tmp_path / "state"
        h.main(["--source", str(src), "--out", str(out), "--workdir", str(tmp_path),
                "--video-url", "https://kick.com/x/videos/1", "--state-dir", str(state),
                "--start", "35:00", "--end", "1:10:00", *extra])
        manifest = json.loads((state / "job.json").read_text())["manifest"]
        return ffmpeg, manifest, (out / "titles.md").read_text()

    def test_section_is_not_cut_again(self, tmp_path, monkeypatch):
        ffmpeg, manifest, titles = self._run(tmp_path, monkeypatch, ["--source-is-section"], 2101.8)
        assert ffmpeg == []                                   # no second cut
        assert manifest == {"video_url": "https://kick.com/x/videos/1",
                            "start": 2100.0, "end": 4200.0, "duration": 2101.8}
        # 10 s into a section that starts at 2098.2 s (35:00 minus the keyframe)
        assert "0:35:08-0:35:38 in the VOD" in titles

    def test_whole_download_is_still_cut(self, tmp_path, monkeypatch):
        ffmpeg, manifest, titles = self._run(tmp_path, monkeypatch, [], 36000.0)  # the whole 10 h VOD
        assert len(ffmpeg) == 1 and "-ss" in ffmpeg[0] and "2100.0" in ffmpeg[0]
        assert manifest == {"video_url": "https://kick.com/x/videos/1",
                            "start": 2100.0, "end": 4200.0, "duration": 2100.0}
        assert "0:35:10-0:35:40 in the VOD" in titles

    def test_workflow_downloads_only_the_section_for_ytdlp(self):
        text = open(self.WORKFLOW, encoding="utf-8").read()
        ytdlp = text[text.index("yt-dlp --no-playlist"):]
        ytdlp = ytdlp[:ytdlp.index("-- \"$VIDEO_URL\"")]
        assert '"${sections[@]}"' in ytdlp
        assert "--force-keyframes-at-cuts" not in text      # no re-encode
        gdown = text[text.index("gdown --fuzzy"):].splitlines()[0]
        assert "section" not in gdown                         # Drive unchanged
        assert 'if [ "$kind" = ytdlp ] && [ -n "$section" ]' in text
        assert "--source-is-section" in text

    def test_format_prefers_1080p_h264_and_never_av1(self):
        text = open(self.WORKFLOW, encoding="utf-8").read()
        fmt = text[text.index("-f 'bv*"):].split("'")[1]
        choices = fmt.split("/")
        # Separate streams first (YouTube's combined files stop at 360p),
        # H.264 first, AV1 excluded (the backend cannot decode it).
        assert choices[0] == "bv*[height<=1080][vcodec^=avc1]+ba"
        assert choices[1] == "bv*[height<=1080][vcodec!^=av01]+ba"
        assert choices[2:] == ["b[height<=1080]", "b"]
