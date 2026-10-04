"""Pronunciation overrides for names and places the model tends to misread.

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

# Proclitics that may be glued to a name: و، ف، ب، ل، ك
_PREFIX = r"(?P<prefix>[وفبلك]?)"
_PLACEHOLDER_BASE = 0xE000  # Unicode private-use area, ignored by diacritizers


class PronunciationLexicon:
    def __init__(self, entries: dict[str, str]):
        self._entries = {k.strip(): v.strip() for k, v in entries.items() if k.strip()}
        if self._entries:
            keys = sorted(self._entries, key=len, reverse=True)  # longest match first
            alternation = "|".join(re.escape(k) for k in keys)
            self._pattern = re.compile(rf"(?<!\w){_PREFIX}(?P<word>{alternation})(?!\w)")
        else:
            self._pattern = None

    @classmethod
    def from_json(cls, path: Path) -> "PronunciationLexicon":
        if not path.exists():
            return cls({})
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self._entries)

    def apply(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: m["prefix"] + self._entries[m["word"]], text)

    def protect(self, text: str) -> tuple[str, Callable[[str], str]]:
        """Swap lexicon matches for placeholders so a diacritizer can't overwrite them.

        Returns the protected text and a function that restores the overrides.
        """
        if self._pattern is None:
            return text, lambda t: t

        replacements: list[str] = []

        def _swap(m: re.Match[str]) -> str:
            replacements.append(self._entries[m["word"]])
            return m["prefix"] + chr(_PLACEHOLDER_BASE + len(replacements) - 1)

        protected = self._pattern.sub(_swap, text)

        def restore(t: str) -> str:
            for i, value in enumerate(replacements):
                t = t.replace(chr(_PLACEHOLDER_BASE + i), value)
            return t

        return protected, restore
