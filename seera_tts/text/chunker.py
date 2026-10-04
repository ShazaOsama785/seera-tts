"""Split narration into speakable units.

`SentenceStream` is the key to real-time: it accepts text deltas as an LLM
streams them and emits each sentence the moment it is complete, so audio for
sentence 1 plays while the LLM is still writing sentence 3.
"""

from __future__ import annotations

import re
from typing import Literal

_TERMINATORS = frozenset(".!؟?…؛\n")
_QURAN_OPEN, _QURAN_CLOSE = "﴿", "﴾"
_QURAN_SPAN = re.compile(r"﴿[^﴾]*﴾")
_CLAUSE_BREAK = re.compile(r"(?<=[،,:])\s+")

SegmentKind = Literal["speech", "quran"]


class SentenceStream:
    """Incremental sentence splitter. Never splits inside ﴿...﴾ or a decimal like 12.5."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        buf, sentences, start, depth, i = self._buffer, [], 0, 0, 0

        while i < len(buf):
            ch = buf[i]
            if ch == _QURAN_OPEN:
                depth += 1
            elif ch == _QURAN_CLOSE:
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


def split_quran(sentence: str) -> list[tuple[SegmentKind, str]]:
    """Separate Quranic text (inside ﴿ ﴾) from narration so it is never machine-voiced."""
    pieces: list[tuple[SegmentKind, str]] = []
    pos = 0
    for match in _QURAN_SPAN.finditer(sentence):
        if before := sentence[pos : match.start()].strip():
            pieces.append(("speech", before))
        pieces.append(("quran", match.group()))
        pos = match.end()
    if tail := sentence[pos:].strip(" ."):
        pieces.append(("speech", sentence[pos:].strip()))
    return pieces


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
