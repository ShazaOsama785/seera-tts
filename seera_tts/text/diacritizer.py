"""Optional automatic diacritization (تشكيل).

Whether full tashkeel helps depends on the TTS engine: XTTS-v2 was trained mostly
on undiacritized Arabic, so A/B test "none" vs "mishkal" on your own chapters.
The lexicon is applied either way and always wins for names it knows.
"""

from __future__ import annotations

import re
from typing import Protocol

_MISHKAL_MARKER = "\x01"  # mishkal replaces . ! : with this control char
_RESTORABLE = re.compile(r"[.!:]")


class Diacritizer(Protocol):
    def __call__(self, text: str) -> str: ...


class NoOpDiacritizer:
    def __call__(self, text: str) -> str:
        return text


class MishkalDiacritizer:
    def __init__(self) -> None:
        try:
            import mishkal.tashkeel
        except ImportError as exc:
            raise ImportError("Install with: pip install mishkal") from exc
        self._vocalizer = mishkal.tashkeel.TashkeelClass()

    def __call__(self, text: str) -> str:
        return _clean_mishkal(text, self._vocalizer.tashkeel(text))


def _clean_mishkal(original: str, output: str) -> str:
    """Put back the punctuation mishkal removed and tidy its spacing."""
    puncts = iter(_RESTORABLE.findall(original))
    output = re.sub(_MISHKAL_MARKER, lambda _: next(puncts, "."), output)
    output = re.sub(r"\s+([،,.؛:!؟?])", r"\1", output)  # no space before punctuation
    output = re.sub(r"([.!:])(?=\S)", r"\1 ", output)    # one space after it
    return re.sub(r"\s+", " ", output).strip()


_REGISTRY = {"none": NoOpDiacritizer, "mishkal": MishkalDiacritizer}


def build_diacritizer(name: str) -> Diacritizer:
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown diacritizer {name!r}. Options: {sorted(_REGISTRY)}") from None