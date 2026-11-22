"""The creator's name in titles and hooks (CREATOR_NAME, set per job).

Without it the copy is generic ("Unexpected wildlife encounter gets
chaotic"); with it every title and hook leads with who did what ("Dasgas Dom
runs into TIGERS"), which is how stream clips are titled. The name is checked
by the API (``valid``) and again here, because it goes into prompts.
"""
from __future__ import annotations

import os
import re

MAX_LEN = 40
_NAME = re.compile(r"^[\w .'\-]{1,%d}$" % MAX_LEN)


def valid(text) -> bool:
    text = (text or "").strip()
    return bool(text) and bool(_NAME.match(text))


def name() -> str:
    text = os.environ.get("CREATOR_NAME", "").strip()
    return text if valid(text) else ""


def prompt_rule() -> str:
    """Extra lines for every prompt that writes titles, hooks or descriptions."""
    rule = ("\n\nNever name a streaming platform (Twitch, Kick, YouTube) in a title, "
            "hook or description unless the transcript names it.")
    who = name()
    if who:
        rule += (
            f"\nThe creator is {who}. Write every title and hook as \"{who}\" plus the "
            f"concrete thing that happens in this clip, e.g. \"{who} runs into TIGERS\". "
            "Name the game when the transcript or the frames make it clear; never guess "
            "one. At most one word in CAPITALS, the most surprising one. No generic "
            "phrases such as \"gets chaotic\" or \"you won't believe\".")
    return rule
