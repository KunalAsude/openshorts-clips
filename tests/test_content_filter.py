"""Content filter: clips with a slur are skipped, strong swearing is masked in
captions and removed from titles/hooks/descriptions.

The real slur list is stored as hashes only, and this public repository never
spells a slur out, tests included: the matching is exercised with a made-up
word put on the list for the test ("zorblax")."""
import hashlib
import importlib

import pytest

import content_filter as cf

MADE_UP = "zorblax"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv("CONTENT_FILTER", raising=False)
    monkeypatch.delenv("CONTENT_FILTER_EXTRA", raising=False)


@pytest.fixture
def listed(monkeypatch):
    """Put the made-up word on the hashed list, the way real entries are."""
    digest = hashlib.sha256(cf.normalize(MADE_UP).encode()).hexdigest()
    monkeypatch.setattr(cf, "SLUR_HASHES", cf.SLUR_HASHES | {digest})


def _transcript(*words):
    """One segment, one word per second: [(text, start), ...]."""
    return {"segments": [{"words": [
        {"word": f" {w}", "start": float(t), "end": float(t) + 0.8} for w, t in words]}]}


class TestList:
    def test_list_is_hashes_only(self):
        assert len(cf.SLUR_HASHES) >= 50
        assert all(len(h) == 64 and set(h) <= set("0123456789abcdef")
                   for h in cf.SLUR_HASHES)

    def test_ordinary_words_are_not_slurs(self):
        for word in ("Niger", "raccoon", "classic", "Scunthorpe", "spice",
                     "negro", "assessment", "cocoon", "retardant"):
            assert not cf.is_slur(word), word


class TestSlurMatching:
    @pytest.mark.parametrize("token", [
        "zorblax", "Zorblax!", "ZORBLAX", "z0rbl4x", "zorblaxxxx", "zoorrbblax",
        "zorblaxs", "zorblaxz", '"zorblax,"',
    ])
    def test_variants_match(self, listed, token):
        assert cf.is_slur(cf._split_edges(token)[1])

    def test_unlisted_word_does_not_match(self):
        assert not cf.is_slur(MADE_UP)

    def test_extra_words_come_from_the_env(self, monkeypatch):
        monkeypatch.setenv("CONTENT_FILTER_EXTRA", "Flimflam, wibble")
        assert cf.is_slur("flimflam") and cf.is_slur("WIBBLE")
        assert not cf.is_slur("wobble")


class TestCaptions:
    @pytest.mark.parametrize("word,masked", [
        ("fuck", "f***"),
        ("fucking,", "f******,"),
        ("Shit!", "S***!"),
        ("bitch", "b****"),
        ("motherfucker", "m***********"),
        ("bullshit.", "b*******."),
        ("fuuuuck", "f******"),
        ("cunt", "c***"),
        ("$hit", "$***"),
        ("sh!t", "s***"),
    ])
    def test_strong_words_are_masked(self, word, masked):
        assert cf.mask_word(word) == masked

    @pytest.mark.parametrize("word", ["hello", "Scunthorpe", "classic", "pitch", "shift",
                                      "shiitake", "Shih-Tzu"])
    def test_other_words_are_untouched(self, word):
        assert cf.mask_word(word) == word

    def test_slur_in_a_caption_is_masked_too(self, listed):
        assert cf.mask_word("zorblax") == "z******"

    def test_off_switch(self, monkeypatch):
        monkeypatch.setenv("CONTENT_FILTER", "0")
        assert cf.mask_word("fuck") == "fuck"

    def test_burned_captions_use_the_mask(self, tmp_path):
        subtitles = importlib.import_module("subtitles")
        out = tmp_path / "c.srt"
        subtitles.generate_srt(_transcript(("what", 0), ("the", 1), ("fuck", 2)),
                               0.0, 3.0, str(out))
        text = out.read_text(encoding="utf-8")
        assert "f***" in text and "fuck" not in text


class TestCopy:
    def test_swear_words_are_removed_not_masked(self):
        got = cf.clean_text("This is fucking insane 🔥 #shit #gaming")
        assert got == "This is insane 🔥 #gaming"

    def test_punctuation_is_tidied(self):
        assert cf.clean_text("Fuck, he did it!") == "he did it!"

    def test_clean_copy_keeps_its_formatting(self):
        text = "Watch till the end 😳\n\n#gta6  #streamer\n- part 2 soon"
        assert cf.clean_text(text) is text

    def test_dirty_copy_keeps_its_line_breaks(self):
        got = cf.clean_text("He fucking did it 😳\n\n#gta6 #shit #streamer")
        assert got == "He did it 😳\n\n#gta6 #streamer"

    def test_slurs_are_removed_from_copy(self, listed):
        assert cf.clean_text("what a zorblax move") == "what a move"


class TestApply:
    def _clip(self, start=0, end=10, **text):
        clip = {"start": start, "end": end,
                "video_title_for_youtube_short": "He clutched the heist",
                "viral_hook_text": "No way he did that 😳",
                "video_description_for_tiktok": "Watch till the end #gta6",
                "video_description_for_instagram": "Insane #gta6"}
        clip.update(text)
        return clip

    def test_clip_with_a_spoken_slur_is_skipped(self, listed):
        transcript = _transcript(("bro", 1), ("zorblax", 2), ("nice", 15))
        clean, dirty = self._clip(12, 20), self._clip(0, 10)
        kept = cf.apply([dirty, clean], transcript)
        assert kept == [clean]

    def test_slur_split_across_word_fragments_is_caught(self, listed):
        # The speech-to-text sometimes splits a word: " zorb" + "lax".
        transcript = {"segments": [{"words": [
            {"word": " zorb", "start": 1.0, "end": 1.3},
            {"word": "lax", "start": 1.3, "end": 1.6}]}]}
        assert cf.apply([self._clip()], transcript) == []

    def test_clip_with_a_slur_in_the_copy_is_skipped(self, listed):
        clip = self._clip(viral_hook_text="zorblax moment")
        assert cf.apply([clip], _transcript(("hello", 1))) == []

    def test_strong_language_is_flagged_and_copy_cleaned(self):
        clip = self._clip(video_title_for_youtube_short="This shit is crazy",
                          viral_hook_text="What the fuck 😳")
        kept = cf.apply([clip], _transcript(("holy", 1), ("shit", 2)))
        assert kept[0]["strong_language"] is True
        assert kept[0]["video_title_for_youtube_short"] == "This is crazy"
        assert kept[0]["viral_hook_text"] == "What the 😳"

    def test_empty_hook_falls_back_to_the_title(self):
        clip = self._clip(viral_hook_text="FUCK!!")
        kept = cf.apply([clip], _transcript(("hi", 1)))
        assert kept[0]["viral_hook_text"] == "He clutched the heist"

    def test_title_that_was_only_swearing_gets_a_fallback(self):
        clip = self._clip(video_title_for_youtube_short="FUCK SHIT")
        kept = cf.apply([clip], _transcript(("hi", 1)))
        assert kept[0]["video_title_for_youtube_short"] == "No way he did that 😳"

    def test_clean_clip_is_unflagged_and_unchanged(self):
        clip = self._clip()
        before = dict(clip)
        kept = cf.apply([clip], _transcript(("hello", 1)))
        assert kept[0]["strong_language"] is False
        assert {k: v for k, v in kept[0].items() if k != "strong_language"} == before

    def test_silent_video_checks_the_copy_only(self):
        kept = cf.apply([self._clip(viral_hook_text="Holy shit")], None)
        assert kept[0]["viral_hook_text"] == "Holy" and kept[0]["strong_language"]

    def test_off_switch_returns_clips_untouched(self, monkeypatch, listed):
        monkeypatch.setenv("CONTENT_FILTER", "0")
        clip = self._clip(viral_hook_text="zorblax fuck")
        assert cf.apply([clip], _transcript(("zorblax", 1))) == [clip]
        assert "strong_language" not in clip

    def test_prompt_rule_only_while_on(self, monkeypatch):
        assert "slur" in cf.prompt_rule()
        monkeypatch.setenv("CONTENT_FILTER", "0")
        assert cf.prompt_rule() == ""
