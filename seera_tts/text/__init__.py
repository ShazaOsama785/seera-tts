"""Text front-end: raw narration -> list of ready-to-speak segments."""

from __future__ import annotations

from dataclasses import dataclass

from .chunker import SegmentKind, SentenceStream, split_long, split_quran, split_sentences
from .diacritizer import Diacritizer, build_diacritizer
from .lexicon import PronunciationLexicon
from .normalizer import normalize

__all__ = [
    "Segment",
    "SentenceStream",
    "TextPreparer",
    "PronunciationLexicon",
    "build_diacritizer",
    "normalize",
    "split_sentences",
]


@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    display_text: str  # what the UI shows / highlights
    spoken_text: str  # what the engine actually receives


class TextPreparer:
    def __init__(self, lexicon: PronunciationLexicon, diacritizer: Diacritizer, max_segment_chars: int):
        self._lexicon = lexicon
        self._diacritize = diacritizer
        self._max_chars = max_segment_chars

    def prepare_sentence(self, sentence: str) -> list[Segment]:
        segments: list[Segment] = []
        for kind, piece in split_quran(sentence):
            if kind == "quran":
                segments.append(Segment("quran", piece, piece))
                continue
            clean = normalize(piece)
            if not clean:
                continue
            for part in split_long(clean, self._max_chars):
                segments.append(Segment("speech", part, self._to_spoken(part)))
        return segments

    def prepare(self, text: str) -> list[Segment]:
        return [seg for sentence in split_sentences(text) for seg in self.prepare_sentence(sentence)]

    def _to_spoken(self, text: str) -> str:
        protected, restore = self._lexicon.protect(text)
        return restore(self._diacritize(protected))
