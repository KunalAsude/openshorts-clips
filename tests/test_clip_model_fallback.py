"""Moment picking runs on GEMINI_MODEL_CLIPS and falls back to flash-lite when
that model is busy; a policy block or a bad answer never switches model."""
import json

import pytest
main = pytest.importorskip("main")
import gemini_worker


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("GEMINI_MODEL", "GEMINI_MODEL_CLIPS", "GEMINI_MODEL_CLIPS_FALLBACK",
                "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(main, "_clip_model_fell_back", None)
    monkeypatch.setattr(main.llm_backend, "active", lambda: False)


def _windows(n):
    return [{"id": f"w{i}"} for i in range(n)]


def _split(model="gemini-3.7-flash", n=2):
    return main._run_stage_split(None, model, _windows(n), json.dumps, None,
                                 "windows", [], "score")


class TestModelNames:
    def test_repo_default_without_env(self):
        assert main.clip_model_names() == ("gemini-3.1-flash-lite", None)

    def test_clip_model_wins_over_the_general_one(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        monkeypatch.setenv("GEMINI_MODEL_CLIPS", "gemini-3.7-flash")
        assert main.clip_model_names() == ("gemini-3.7-flash", "gemini-3.1-flash-lite")

    def test_general_model_still_applies(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
        assert main.clip_model_names() == ("gemini-3.6-flash", "gemini-3.1-flash-lite")

    def test_fallback_is_configurable(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_CLIPS", "gemini-3.7-flash")
        monkeypatch.setenv("GEMINI_MODEL_CLIPS_FALLBACK", "gemini-3.6-flash")
        assert main.clip_model_names()[1] == "gemini-3.6-flash"


class TestFallback:
    @pytest.fixture(autouse=True)
    def _clip_model(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_CLIPS", "gemini-3.7-flash")

    def _stage(self, monkeypatch, fail_with):
        models = []

        def fake(client, model, prompt, schema):
            models.append(model)
            if model == "gemini-3.7-flash" and fail_with is not None:
                raise fail_with
            ids = [w["id"] for w in json.loads(prompt)]
            return {"windows": [{"id": i} for i in ids]}, {"model": model}

        monkeypatch.setattr(main, "_run_gemini_stage", fake)
        return models

    @pytest.mark.parametrize("err", [
        "503 UNAVAILABLE. The model is overloaded.",
        "429 RESOURCE_EXHAUSTED",
        "500 INTERNAL",
        "504 DEADLINE_EXCEEDED",
        "The read operation timed out",
    ])
    def test_busy_primary_falls_back(self, monkeypatch, err):
        models = self._stage(monkeypatch, RuntimeError(err))
        assert [w["id"] for w in _split()] == ["w0", "w1"]
        assert models == ["gemini-3.7-flash", "gemini-3.1-flash-lite"]

    def test_stays_on_the_fallback_for_the_rest_of_the_job(self, monkeypatch):
        models = self._stage(monkeypatch, RuntimeError("503 UNAVAILABLE"))
        _split()
        _split()
        assert models == ["gemini-3.7-flash", "gemini-3.1-flash-lite",
                          "gemini-3.1-flash-lite"]

    def test_policy_block_never_switches_model(self, monkeypatch):
        models = self._stage(monkeypatch,
                             gemini_worker.GeminiBlockedError("PROHIBITED_CONTENT"))
        assert _split(n=1) == []  # a lone blocked window is dropped, as before
        assert models == ["gemini-3.7-flash"]

    def test_bad_answer_never_switches_model(self, monkeypatch):
        self._stage(monkeypatch, ValueError("Failed to parse Gemini JSON response"))
        with pytest.raises(ValueError):
            _split()

    @pytest.mark.parametrize("err", [
        # A JSON error quotes positions, a validation error echoes the answer:
        # neither number is an HTTP status.
        ValueError("Failed to parse Gemini JSON response: Expecting value: "
                   "line 1 column 500 (char 499)"),
        ValueError("1 validation error for ScoreResponse score input_value=503"),
    ])
    def test_numbers_in_a_bad_answer_never_switch_model(self, monkeypatch, err):
        models = self._stage(monkeypatch, err)
        with pytest.raises(ValueError):
            _split()
        assert models == ["gemini-3.7-flash"]
        assert main._clip_model_fell_back is None

    def test_a_busy_fallback_is_reported_not_looped(self, monkeypatch):
        def always_busy(client, model, prompt, schema):
            raise RuntimeError(f"503 UNAVAILABLE ({model})")
        monkeypatch.setattr(main, "_run_gemini_stage", always_busy)
        with pytest.raises(RuntimeError, match="flash-lite"):
            _split()

    def test_cost_label_names_both_models_after_a_fallback(self, monkeypatch):
        assert main.clip_models_used("gemini-3.7-flash") == "gemini-3.7-flash"
        self._stage(monkeypatch, RuntimeError("503 UNAVAILABLE"))
        _split()
        assert (main.clip_models_used("gemini-3.7-flash")
                == "gemini-3.7-flash -> gemini-3.1-flash-lite")

    def test_healthy_primary_is_one_call(self, monkeypatch):
        models = self._stage(monkeypatch, None)
        _split()
        assert models == ["gemini-3.7-flash"]

    def test_local_llm_never_falls_back(self, monkeypatch):
        monkeypatch.setattr(main.llm_backend, "active", lambda: True)
        self._stage(monkeypatch, RuntimeError("503 UNAVAILABLE"))
        with pytest.raises(RuntimeError):
            _split()
