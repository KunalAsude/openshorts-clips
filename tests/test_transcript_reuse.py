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
