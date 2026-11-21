"""Moment picking walks a chain of Gemini models when they are busy, waits and
goes round again when all of them are, and never switches model on a policy
block or a bad answer."""
import json

import pytest
main = pytest.importorskip("main")
import gemini_worker

PRIMARY = "gemini-3.6-flash"
CHAIN = [PRIMARY, "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite"]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("GEMINI_MODEL", "GEMINI_MODEL_CLIPS", "GEMINI_MODEL_CLIPS_FALLBACK",
                "CLIP_CHAIN_WAITS", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(main, "_clip_model_current", None)
    monkeypatch.setattr(main, "_clip_models_answered", [])
    monkeypatch.setattr(main, "_clip_models_unusable", set())
    monkeypatch.setattr(main.llm_backend, "active", lambda: False)


@pytest.fixture
def waits(monkeypatch):
    slept = []
    monkeypatch.setattr(main, "_chain_sleep", slept.append)
    return slept


def _windows(n):
    return [{"id": f"w{i}"} for i in range(n)]


def _split(model=PRIMARY, n=2):
    return main._run_stage_split(None, model, _windows(n), json.dumps, None,
                                 "windows", [], "score")


class TestChain:
    def test_repo_default_without_env(self):
        assert main.clip_model_chain() == [
            "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.7-flash"]

    def test_clip_model_leads_then_the_default_list(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_CLIPS", PRIMARY)
        assert main.clip_model_chain() == CHAIN

    def test_custom_list_keeps_order_and_drops_repeats(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_CLIPS", "a")
        monkeypatch.setenv("GEMINI_MODEL_CLIPS_FALLBACK", " b, a ,c,,b ")
        assert main.clip_model_chain() == ["a", "b", "c"]

    def test_general_model_still_applies(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-x")
        assert main.clip_model_chain()[0] == "gemini-x"

    def test_waits_default_and_override(self, monkeypatch):
        assert main.clip_chain_waits() == (60, 120, 240)
        monkeypatch.setenv("CLIP_CHAIN_WAITS", "1,2")
        assert main.clip_chain_waits() == (1.0, 2.0)


class TestFallback:
    @pytest.fixture(autouse=True)
    def _clip_model(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_CLIPS", PRIMARY)

    def _stage(self, monkeypatch, busy=(), fail_with=None):
        """Fake stage: models in ``busy`` raise ``fail_with``."""
        models = []

        def fake(client, model, prompt, schema):
            models.append(model)
            if model in busy:
                raise fail_with
            ids = [w["id"] for w in json.loads(prompt)]
            return {"windows": [{"id": i} for i in ids]}, {"model": model}

        monkeypatch.setattr(main, "_run_gemini_stage", fake)
        return models

    @pytest.mark.parametrize("err", [
        "503 UNAVAILABLE. This model is currently experiencing high demand.",
        "429 RESOURCE_EXHAUSTED",
        "500 INTERNAL",
        "504 DEADLINE_EXCEEDED",
        "The read operation timed out",
    ])
    def test_busy_model_hands_over_to_the_next(self, monkeypatch, waits, err):
        models = self._stage(monkeypatch, busy={PRIMARY}, fail_with=RuntimeError(err))
        assert [w["id"] for w in _split()] == ["w0", "w1"]
        assert models == [PRIMARY, "gemini-3.5-flash"]
        assert waits == []

    def test_walks_the_whole_chain_in_order(self, monkeypatch, waits):
        models = self._stage(monkeypatch, busy=set(CHAIN[:3]),
                             fail_with=RuntimeError("503 UNAVAILABLE"))
        _split()
        assert models == CHAIN
        assert main.clip_models_used(PRIMARY) == "gemini-3.1-flash-lite"

    def test_next_call_starts_with_the_model_that_answered(self, monkeypatch, waits):
        models = self._stage(monkeypatch, busy={PRIMARY},
                             fail_with=RuntimeError("503 UNAVAILABLE"))
        _split()
        _split()
        assert models == [PRIMARY, "gemini-3.5-flash", "gemini-3.5-flash"]

    def test_all_busy_waits_then_recovers(self, monkeypatch, waits):
        calls = {"n": 0}

        def fake(client, model, prompt, schema):
            calls["n"] += 1
            if calls["n"] <= len(CHAIN) * 2:      # two full busy passes
                raise RuntimeError("503 UNAVAILABLE high demand")
            return {"windows": [{"id": "w0"}]}, {"model": model}

        monkeypatch.setattr(main, "_run_gemini_stage", fake)
        assert _split(n=1) == [{"id": "w0"}]
        assert waits == [60, 120]

    def test_all_busy_through_every_wait_fails_clearly(self, monkeypatch, waits):
        models = self._stage(monkeypatch, busy=set(CHAIN),
                             fail_with=RuntimeError("503 UNAVAILABLE"))
        with pytest.raises(main.AllModelsBusy, match="Every clip model was busy"):
            _split()
        assert waits == [60, 120, 240]
        assert len(models) == len(CHAIN) * 4

    def test_policy_block_never_switches_model(self, monkeypatch, waits):
        models = self._stage(monkeypatch, busy={PRIMARY},
                             fail_with=gemini_worker.GeminiBlockedError("PROHIBITED_CONTENT"))
        assert _split(n=1) == []  # a lone blocked window is dropped, as before
        assert models == [PRIMARY]

    @pytest.mark.parametrize("err", [
        ValueError("Failed to parse Gemini JSON response"),
        # A JSON error quotes positions, a validation error echoes the answer:
        # neither number is an HTTP status.
        ValueError("Failed to parse Gemini JSON response: Expecting value: "
                   "line 1 column 500 (char 499)"),
        ValueError("1 validation error for ScoreResponse score input_value=503"),
    ])
    def test_bad_answer_never_switches_model(self, monkeypatch, waits, err):
        models = self._stage(monkeypatch, busy={PRIMARY}, fail_with=err)
        with pytest.raises(ValueError):
            _split()
        assert models == [PRIMARY]
        assert main._clip_model_current is None

    def test_healthy_primary_is_one_call(self, monkeypatch, waits):
        models = self._stage(monkeypatch)
        _split()
        assert models == [PRIMARY]
        assert main.clip_models_used(PRIMARY) == PRIMARY

    def test_cost_label_names_every_model_that_answered(self, monkeypatch, waits):
        busy = {PRIMARY}
        models = []

        def fake(client, model, prompt, schema):
            models.append(model)
            if model in busy:
                raise RuntimeError("503 UNAVAILABLE")
            return {"windows": []}, None

        monkeypatch.setattr(main, "_run_gemini_stage", fake)
        _split()
        busy.add("gemini-3.5-flash")
        _split()
        assert (main.clip_models_used(PRIMARY)
                == "gemini-3.5-flash -> gemini-3.7-flash")

    def test_unknown_model_is_skipped_for_the_rest_of_the_job(self, monkeypatch, waits):
        models = []

        def fake(client, model, prompt, schema):
            models.append(model)
            if model == PRIMARY:
                raise RuntimeError("503 UNAVAILABLE")
            if model == "gemini-3.5-flash":
                raise RuntimeError("404 NOT_FOUND. models/gemini-3.5-flash is not found")
            return {"windows": []}, None

        monkeypatch.setattr(main, "_run_gemini_stage", fake)
        monkeypatch.setattr(main, "_clip_models_unusable", set())
        _split()
        monkeypatch.setattr(main, "_clip_model_current", None)  # start over at the primary
        _split()
        assert models == [PRIMARY, "gemini-3.5-flash", "gemini-3.7-flash",
                          PRIMARY, "gemini-3.7-flash"]

    def test_no_usable_model_fails_with_the_reason(self, monkeypatch, waits):
        monkeypatch.setattr(main, "_clip_models_unusable", set())
        self._stage(monkeypatch, busy=set(CHAIN),
                    fail_with=RuntimeError("404 NOT_FOUND"))
        with pytest.raises(main.AllModelsBusy, match="can be used with this key"):
            _split()
        assert waits == []

    def test_total_wait_is_capped(self, monkeypatch, waits):
        monkeypatch.setenv("CLIP_CHAIN_MAX_SECONDS", "100")
        self._stage(monkeypatch, busy=set(CHAIN), fail_with=RuntimeError("503 UNAVAILABLE"))
        with pytest.raises(main.AllModelsBusy):
            _split()
        assert waits == [60]          # 60 + 120 would pass the 100 s cap

    def test_local_llm_never_falls_back(self, monkeypatch, waits):
        monkeypatch.setattr(main.llm_backend, "active", lambda: True)
        self._stage(monkeypatch, busy={PRIMARY}, fail_with=RuntimeError("503 UNAVAILABLE"))
        with pytest.raises(RuntimeError):
            _split()


class TestRetryOncePerModel:
    """Inside _run_gemini_stage: a busy model gets ONE retry before the chain
    moves on; an answer that does not parse keeps all three tries."""

    class _Client:
        def __init__(self, errors):
            self.errors = list(errors)
            self.calls = 0
            self.models = self

        def generate_content(self, model, contents, config):
            self.calls += 1
            raise self.errors[min(self.calls, len(self.errors)) - 1]

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(main.time, "sleep", lambda s: None)

    def test_busy_is_retried_once(self):
        client = self._Client([RuntimeError("503 UNAVAILABLE high demand")])
        with pytest.raises(RuntimeError):
            main._run_gemini_stage(client, PRIMARY, "p", None)
        assert client.calls == 2

    def test_timeout_without_a_message_is_retried_once(self):
        class ReadTimeout(Exception):
            pass
        client = self._Client([ReadTimeout()])
        with pytest.raises(ReadTimeout):
            main._run_gemini_stage(client, PRIMARY, "p", None)
        assert client.calls == 2

    def test_bad_answer_keeps_three_tries(self):
        client = self._Client([ValueError("Failed to parse Gemini JSON response")])
        with pytest.raises(ValueError):
            main._run_gemini_stage(client, PRIMARY, "p", None)
        assert client.calls == 3
