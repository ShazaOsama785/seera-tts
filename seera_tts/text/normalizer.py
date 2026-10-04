"""Clean narration text so the TTS model reads it the way a human narrator would."""

from __future__ import annotations

import re

_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Symbols / abbreviations a narrator would expand aloud.
_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\(?\s*ﷺ\s*\)?"), " صلى الله عليه وسلم "),
    (re.compile(r"\(\s*ص\s*\)"), " صلى الله عليه وسلم "),
    (re.compile(r"\(\s*رض\s*\)"), " رضي الله عنه "),
    (re.compile(r"ﷻ"), " جل جلاله "),
    (re.compile(r"(\d+)\s*هـ"), r"\1 للهجرة"),
    (re.compile(r"(\d+)\s*م(?=[\s.,،؛]|$)"), r"\1 للميلاد"),
]

# Things that should never be spoken.
_REMOVALS: list[re.Pattern[str]] = [
    re.compile(r"https?://\S+"),
    re.compile(r"\[[^\]]*\]"),  # source tags like [الدرر السنية]
    re.compile(r"\((?:ج\s*\d+\s*[،,]?\s*)?ص\s*\d+\)"),  # page refs like (ج2، ص45)
    re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]"),  # invisible bidi chars
    re.compile(r"[*_#`>|]+"),  # markdown leftovers from the LLM
]

_TATWEEL = "\u0640"


def normalize(text: str) -> str:
    text = text.translate(_EASTERN_DIGITS)
    for pattern in _REMOVALS:
        text = pattern.sub(" ", text)
    for pattern, replacement in _EXPANSIONS:  # before tatweel removal: "هـ" relies on it
        text = pattern.sub(replacement, text)
    text = text.replace(_TATWEEL, "")
    text = re.sub(r"\s+([،,.؛:!؟?])", r"\1", text)  # no space before punctuation
    return re.sub(r"\s+", " ", text).strip()
