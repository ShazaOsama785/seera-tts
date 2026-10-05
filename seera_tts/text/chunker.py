"""Split narration into speakable units.

`SentenceStream` is the key to real-time: it accepts text deltas as an LLM
streams them and emits each sentence the moment it is complete, so audio for
sentence 1 plays while the LLM is still writing sentence 3.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from .quran import surah_number

logger = logging.getLogger(__name__)

_TERMINATORS = frozenset(".!؟?…؛\n")
_QURAN_OPEN, _QURAN_CLOSE = "﴿{", "﴾}"  # sources use ﴿...﴾ or {...} for verses
# Verse + optional reference:  { ... } [ العلق: 1: 3 ]  /  [العلق: 1-3]  /  [96: 1]
_QURAN_SPAN = re.compile(
    r"(?P<verse>[﴿{][^﴾}]*[﴾}])"
    r"(?:\s*\[\s*(?P<surah>[^\]:]+?)\s*:\s*(?P<start>\d{1,3})"
    r"(?:\s*[:\-–]\s*(?P<end>\d{1,3}))?\s*\])?"
)
_HAS_WORD = re.compile(r"\w")  # skip leftovers like ")،"
_CLAUSE_BREAK = re.compile(r"(?<=[،,:])\s+")

SegmentKind = Literal["speech", "quran"]


class SentenceStream:
    """Incremental sentence splitter. Never splits inside a verse or a decimal like 12.5."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        buf, sentences, start, depth, i = self._buffer, [], 0, 0, 0

        while i < len(buf):
            ch = buf[i]
            if ch in _QURAN_OPEN:
                depth += 1
            elif ch in _QURAN_CLOSE:
                depth = max(0, depth - 1)
            elif ch in _TERMINATORS and depth == 0:
                if ch == "." and i > 0 and buf[i - 1].isdigit():
                    if i + 1 == len(buf):
                        break  # "12." might become "12.5" - wait for the next delta
                    if buf[i + 1].isdigit():
                        i += 1
                        continue
                end = i + 1
                while end < len(buf) and buf[end] in _TERMINATORS:  # "؟!" or "..."
                    end += 1
                if sentence := buf[start:end].strip():
                    sentences.append(sentence)
                start = i = end
                continue
            i += 1

        self._buffer = buf[start:]
        return sentences

    def flush(self) -> list[str]:
        remainder, self._buffer = self._buffer.strip(), ""
        return [remainder] if remainder else []


def split_sentences(text: str) -> list[str]:
    stream = SentenceStream()
    return stream.feed(text) + stream.flush()


@dataclass(frozen=True)
class QuranRef:
    surah: int
    start: int
    end: int

    def ayahs(self) -> list[tuple[int, int]]:
        return [(self.surah, a) for a in range(self.start, self.end + 1)]


def split_quran(sentence: str) -> list[tuple[SegmentKind, str, QuranRef | None]]:
    """Separate Quranic text from narration so it is never machine-voiced."""
    pieces: list[tuple[SegmentKind, str, QuranRef | None]] = []
    pos = 0
    for m in _QURAN_SPAN.finditer(sentence):
        if _HAS_WORD.search(before := sentence[pos : m.start()].strip()):
            pieces.append(("speech", before, None))
        pieces.append(("quran", m["verse"], _parse_ref(m)))
        pos = m.end()
    if _HAS_WORD.search(tail := sentence[pos:].strip()):
        pieces.append(("speech", tail, None))
    return pieces


def _parse_ref(m: re.Match[str]) -> QuranRef | None:
    if not m["surah"]:
        return None
    surah = surah_number(m["surah"])
    if surah is None:
        logger.warning("Unknown surah name %r - verse will be shown without audio", m["surah"])
        return None
    start = int(m["start"])
    end = int(m["end"]) if m["end"] else start
    return QuranRef(surah, start, end) if end >= start else None


def split_long(text: str, max_chars: int) -> list[str]:
    """Break text over `max_chars` at clause boundaries (، , :), then at word boundaries."""
    if len(text) <= max_chars:
        return [text]

    clauses = [c for c in _CLAUSE_BREAK.split(text) if c]
    by_clause = len(clauses) > 1
    units = clauses if by_clause else text.split()

    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long(unit, max_chars) if by_clause else [unit])
            continue
        candidate = f"{current} {unit}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks