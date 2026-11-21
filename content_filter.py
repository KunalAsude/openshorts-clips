"""Keep clips safe to post: no slurs, no strong swearing in captions or copy.

On by default; ``CONTENT_FILTER=0`` turns every part of it off.

- A clip whose words (the transcript inside it, which is also its captions) or
  whose generated text contain a slur is dropped before it is rendered.
- Strong swear words are masked in burned captions (``f***``, ``s***``,
  ``b****``: first letter kept, one ``*`` per remaining letter). The audio is
  not touched.
- Titles, descriptions and hooks never carry a swear word or a slur: those
  words are removed from the text, not masked. A hook left empty falls back to
  the title.
- A kept clip with strong language is flagged ``strong_language: true``.

The slur list is stored only as SHA-256 hashes of the normalised words, so
this public repository never spells them out. That is obfuscation, not
secrecy: hashes of single words can be found with a dictionary. ``CONTENT_FILTER_EXTRA`` adds
your own words (comma-separated); they are treated like slurs: a clip that says
one is skipped, and they are masked/removed everywhere else.

Matching is per word after normalising: lower case, common leetspeak
(1/! -> i, 0 -> o, 3 -> e, 4/@ -> a, 5/$ -> s, 7 -> t), punctuation removed,
and repeated letters squeezed ("fuuuuck"). A word list cannot catch every
misspelling or a word the speech-to-text misheard, and it only knows English
words written in Latin letters; the prompt rule asks Gemini to avoid such
moments too, but clips should still be watched before posting.
"""
import functools
import hashlib
import os
import re

TEXT_FIELDS = ("video_title_for_youtube_short", "video_description_for_tiktok",
               "video_description_for_instagram", "viral_hook_text", "title")
TITLE_FIELD = "video_title_for_youtube_short"
HOOK_FIELD = "viral_hook_text"

# Strong swearing. Stems match anywhere in a word (fucking, motherfucker,
# bullshit, bitches); the whole words cover spellings the stems miss.
STRONG_STEMS = ("fuck", "shit", "bitch")
STRONG_WORDS = frozenset({
    "cunt", "cunts", "fuk", "fuks", "fukin", "fukkin", "fukking", "fucc",
    "fuccin", "phuck", "shyt", "biatch", "biotch",
})

# Ordinary words a stem would catch once their letters are squeezed
# ("shiitake" -> "shitake").
NOT_STRONG = frozenset({"shiitake", "shitake", "shihtzu", "shitzu"})

# sha256(normalised word). Variants (plurals, spellings) are listed one by one
# rather than stemmed, so an ordinary word can never be cut down into a match.
SLUR_HASHES = frozenset({
    "038c9ccdd226f5728bd0a945bdbb0a25c0f877f2f36f4092ee8c004e810aa300",
    "044eb98b18769b887d5a0258f675e413058b8e9fc9b9786b418bfc6f03f26c98",
    "08a841e996781e9e77d30a4e4420a8f501a280b00624e6d1224bf54aaff73eba",
    "093b7636b24aca21881555cbf01456403ee52306a78a181f55fd58a5bb5f8b10",
    "0ce875d620076b533c6e681abaa0dc5d9a941366bb000a02ec5351f341b96c61",
    "120f6e5b4ea32f65bda68452fcfaaef06b0136e1d0e4a6f60bc3771fa0936dd6",
    "1290d04b901d01be931cf9d6dc72938028c730f3f74e12629d7735f4fdfed660",
    "12e6274e4309293e2d480272b49a6c7c73a6a6b22678ba226b533c67006c17d1",
    "158869a97379229b7681efae9d7f9c9214134e836d649ba53477c0c111414d59",
    "16ea09fc78ca83ca502cbcf2377acdf280bf18f61e259153f0868405eedab5ef",
    "17a110e5332848721f27510e1e2bca710a321663bb8a5632e0e46a820fc9e28a",
    "17bde8b4064612e266eaa1b5b1dfa054d1e6dd263f66b392159006d2c9a8ae8a",
    "1e02eec4f1095143be282056557034d4e8ab915342c1af508223141d472d2347",
    "21a991a802ed50e71f8fc395fc2094c137720b4f27bc2d04728f691e6d26bfbf",
    "22fc75e65a0e9d34324092a7c6a8dba961853294abca4e5914e60c550f48e0c2",
    "268651b3ece980102f18871fde07189372961e056f858ca147a28d004f876b03",
    "333f7618092958c75b8c5af6f1ec77b42803922a0fc6ff1570a8af3a3aab3b4a",
    "33740da8a87dc7defbdbb7d06549dd8d82a33000a25293ceb9af687eb16bef22",
    "341d56384afc0f47b34ca18273e793be555507a49444c30d3d0588688de46cb3",
    "3b1e0d7c5dd45583867e897943e37a940a7e7321022317dd3deea01963ee365e",
    "3c8400b402baa7f1b8cac94e00b9b9132bc1ec7d7fcf60bf208dc138aec73393",
    "3cc0a581d3bbe605317791e06aa1b4f3348b4e00c4c106c9e6b8210e6933b66c",
    "3eed6d8f9ba46205b8fba3d622d26832fcc2a342b65e940a538aeb8c4d32af2d",
    "402f7ebd98864afed4817ff4718d357dbfa95458aa4236d6c5959536956fa4e5",
    "45cd3e1bb472d8e285a160a95f6409ec5ce86102e9caaaf0b63fc11db04ab559",
    "5b3ae48be122f7ed19b4cc587649f41f9d2565df51cfa332f8e7806f4ebb9032",
    "63bd9333b1a0b11da8d0b206c4b86cd2a5be63fb5e7b59cc5ac7131f3a257489",
    "693ee9961bef39c102f077c046c93cc97968f1acbef0a5491e7dfdb69ac842e5",
    "6d1833779e389477ac6757e19c371df8e141bd67bc83e4b596dfc8a1947db29d",
    "70e4043b678ff365b6377e3ceeb5067bc27bbcd03707e17a36b62bf0201340bc",
    "7122c9a14ba2d20ebb6ebc763eb8433eee8877a22d84ff75e6056a81bf7cd0ee",
    "806a24e8192c27b681a816c1c9cf1e842c8bee6be4c2b48394cc1651319ea936",
    "82159dd02870c13ec30e4f146db7c317aca568469d0e7e22b8610a2eaa3d3924",
    "83dd8e733b1e2c6ec43e98cc2426c9f7e94ee0261a978c57bdc214e764c7f2df",
    "85580f904474c32c4d826196710555cab41ef442bfb99806d47a09161b0f2ac9",
    "87d1690e74a1d8276d11401dc3da66b774e8f1cade9758b9ddf8737166ca73b7",
    "886d51e97ad7931d0d2af8439ca6d9e4887e3c2b469ed247cbd68ceb3649ccde",
    "8f5083e3e5c7dc8932f2bf58212f963f3a44752618c96297f82623f736c52738",
    "92a9bf818e7e00021b6d6959f877f5df11b0f989e486db38f35853ebbd552086",
    "94c6d2e10a76c3134a31abb9a1e0e54feb4fe2b999cc5b788d5e9ef09c59b659",
    "98b52c4b6b7d1f48e7477a5ccc10955dd195d0ac5a38c8281bfeb08762634909",
    "9915ba2d822280f22c283df4e76584a40e0119fc58f73c5f84d4fdb04d04fa6f",
    "ba59331131251da4ca819aa8bd51bb0636e3969bd224b1210d6420c84da5ad49",
    "bb209d506bb8b17815af0e50923c3d8f65985dab91083f026f1881a7430c5cc7",
    "bd331fb1d24298f52943034a243a341877957b895f4372b11babb87262904ed6",
    "c10da7b2b13906ece589905aaff43a8ea666c89fe4d426d3bbc931d9ed4930e5",
    "c1cabb6f6e431f9c4dea4c3b3264d6fe3829241d674f3496a2dfff6658f0363e",
    "c3de533e9b7fe63b79f648687a30d2861edd92fe7c3cd1f2c485e0a605367624",
    "c74677aba35aef499bbe96dcc049c77700ca13835bb4930706da85ecdba4e156",
    "cc02032349c833ac5e97bac094560ed40e09acf34cb1978ab7a9840b9bf15b4d",
    "d7cc7ecc7ef00cd6b15c5855d61a95ddcb39b4a715aa62ef5ba3a573057f7c85",
    "da0a6b1b9213d48ff10cc45733cc8a925dbd2afb26320ed488ac2efa9a8ae33a",
    "daec0c235d3135fc1c60aee787f784c59c4d53b2041ff2bacd0d3bebda7246a8",
    "dc675e448132fd2a4fed47c1736784e83fe01e8cf137dcf97cca9fc7e337e8b4",
    "e742d17ff490c87e19d0cfbfdb18c691aad9543e5d61498df7a6ba6dc56e3124",
    "e793d67686f2824f67dd73b1c955a74095b7fc54ae95c7e1b4fde436e27d15a9",
    "e7b98c6aa5b944e0b315d350d423f895ac9e44fb84f1534b18c2572370a67b9e",
    "eef3bd091670c3447022d619c06ad15de96da72b5a66f28bb8b75d1b1c12a05f",
    "ef6e97ba7fa99b7aedf8d6f2c63eb1c363b9110fae815a74b14b76a7f5b90366",
    "f9d0d9b18ae9033a5ea36df19bf279b059e887a9ae785db81117bceaecc95933",
})

_LEET = str.maketrans({"1": "i", "!": "i", "0": "o", "3": "e", "4": "a",
                       "@": "a", "5": "s", "$": "s", "7": "t"})
# Punctuation around a word, kept when it is masked. Leading $ @ ! are NOT
# stripped: they are leetspeak for s/a/i ("$hit").
_EDGE = re.compile(r"^([^\w$@!]*)(.*?)([^\w]*)$", re.UNICODE)


def enabled():
    return os.environ.get("CONTENT_FILTER", "1").strip() != "0"


def normalize(token):
    """Lower-case letters only, after mapping leetspeak."""
    return re.sub(r"[^a-z]", "", str(token or "").lower().translate(_LEET))


def _candidates(token):
    """The forms a word is checked in: as written, with letter runs of three
    or more squeezed to two ("niggggga"), and to one ("fuuuck"), each also
    without a plural s/z."""
    base = normalize(token)
    if not base:
        return set()
    forms = {base,
             re.sub(r"(.)\1{2,}", r"\1\1", base),
             re.sub(r"(.)\1+", r"\1", base)}
    for f in list(forms):
        if len(f) > 3 and f[-1] in "sz":
            forms.add(f[:-1])
    return forms


def _digest(word):
    return hashlib.sha256(word.encode("utf-8")).hexdigest()


def _extra_words():
    return _parse_extra(os.environ.get("CONTENT_FILTER_EXTRA", ""))


@functools.lru_cache(maxsize=8)
def _parse_extra(raw):
    return frozenset(normalize(w) for w in raw.split(",") if normalize(w))


def is_slur(token):
    forms = _candidates(token)
    if not forms:
        return False
    if forms & _extra_words():
        return True
    return any(_digest(f) in SLUR_HASHES for f in forms)


def is_strong(token):
    if normalize(token) in NOT_STRONG:
        return False
    forms = _candidates(token)
    return any(f in STRONG_WORDS or any(s in f for s in STRONG_STEMS) for f in forms)


def _split_edges(word):
    m = _EDGE.match(str(word or ""))
    return m.group(1), m.group(2), m.group(3)


def mask_word(word):
    """``word`` with a swear (or slur) masked, punctuation kept: "fucking," ->
    "f******,". Everything else comes back unchanged."""
    if not enabled():
        return word
    lead, core, trail = _split_edges(word)
    if core and (is_strong(core) or is_slur(core)):
        core = core[0] + "*" * (len(core) - 1)
    return f"{lead}{core}{trail}"


def clean_text(text):
    """``text`` without its swear words and slurs. Unchanged (line breaks,
    hashtag blocks and all) when it has none; otherwise only the spacing
    around the removed words is tidied."""
    if not text or not enabled():
        return text
    parts = re.split(r"(\s+)", str(text))   # words at even indexes, spacing at odd
    removed = False
    for i in range(0, len(parts), 2):
        _lead, core, _trail = _split_edges(parts[i])
        if core and (is_strong(core) or is_slur(core)):
            parts[i] = ""
            removed = True
    if not removed:
        return text
    out = "".join(parts)
    out = re.sub(r"[ \t]{2,}", " ", out)                 # gap left by a word
    out = re.sub(r"[ \t]+([,.!?;:])", r"\1", out)        # "word ," -> "word,"
    out = re.sub(r"^[ \t,.;:!?]+", "", out, flags=re.M)   # no leading punctuation
    out = re.sub(r"[ \t]+$", "", out, flags=re.M)
    return out.strip()


def _text_has(clip, check):
    for field in TEXT_FIELDS:
        value = clip.get(field)
        if isinstance(value, str) and any(check(_split_edges(t)[1]) for t in value.split()):
            return True
    return False


def clip_words(transcript, start, end):
    """The words spoken inside [start, end], continuation fragments merged
    (the speech-to-text splits some words in two)."""
    import subtitles
    flat = []
    for segment in (transcript or {}).get("segments", []) or []:
        flat.extend(segment.get("words", []) or [])
    words = subtitles.merge_continuation_words(flat)
    out = []
    for w in words:
        try:
            if float(w.get("end", 0)) > start and float(w.get("start", 0)) < end:
                out.append(str(w.get("word", "")))
        except (TypeError, ValueError):
            continue
    return out


def clean_clip_text(clip):
    """Remove swear words and slurs from the clip's title, descriptions and
    hook, in place. A hook left empty falls back to the (clean) title.
    Returns True when anything changed."""
    if not enabled():
        return False
    changed = False
    for field in TEXT_FIELDS:
        value = clip.get(field)
        if isinstance(value, str) and value:
            cleaned = clean_text(value)
            if cleaned != value:
                clip[field] = cleaned
                changed = True
    title, hook = clip.get(TITLE_FIELD), clip.get(HOOK_FIELD)
    if isinstance(title, str) and not title.strip():
        # A title that was nothing but swearing: the hook, else a plain label.
        clip[TITLE_FIELD] = (hook.strip() if isinstance(hook, str) and hook.strip()
                             else "Stream highlight")
    if isinstance(hook, str) and not hook.strip() and clip.get(TITLE_FIELD):
        clip[HOOK_FIELD] = " ".join(str(clip[TITLE_FIELD]).split()[:10])
    return changed


def apply(shorts, transcript):
    """The clips that are safe to make, with clean copy and the
    ``strong_language`` flag. Dropped clips are logged, never rendered."""
    if not enabled():
        return shorts
    kept, dropped, strong = [], 0, 0
    for clip in shorts or []:
        try:
            start, end = float(clip.get("start", 0)), float(clip.get("end", 0))
        except (TypeError, ValueError):
            start = end = 0.0
        words = clip_words(transcript, start, end)
        if any(is_slur(_split_edges(w)[1]) for w in words) or _text_has(clip, is_slur):
            dropped += 1
            continue
        has_strong = (any(is_strong(_split_edges(w)[1]) for w in words)
                      or _text_has(clip, is_strong))
        clean_clip_text(clip)
        clip["strong_language"] = bool(has_strong)
        strong += has_strong
        kept.append(clip)
    if dropped or strong:
        print(f"🧼 Content filter: skipped {dropped} clip(s) with a slur; "
              f"{strong} kept clip(s) have strong language (masked in captions).")
    return kept


def prompt_rule():
    """Extra instructions for the Gemini prompts while the filter is on."""
    if not enabled():
        return ""
    return ("\n\nCONTENT RULES: Do not pick moments where anyone uses a racial or "
            "other slur. Never use swear words, profanity or slurs in titles, "
            "descriptions or hook text, even when the speaker does; write "
            "around them.\n")
