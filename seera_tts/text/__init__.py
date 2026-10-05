"""Text front-end: raw narration -> list of ready-to-speak segments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunker import QuranRef, SegmentKind, SentenceStream, split_long, split_quran, split_sentences
from .diacritizer import Diacritizer, build_diacritizer
from .lexicon import PronunciationLexicon
from .normalizer import normalize
from .quran import QuranIndex

logger = logging.getLogger(__name__)

__all__ = [
    "QuranIndex",
    "QuranRef",
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
    quran_ref: QuranRef | None = None  # only for kind == "quran"


class TextPreparer:
    def __init__(
        self,
        lexicon: PronunciationLexicon,
        diacritizer: Diacritizer,
        max_segment_chars: int,
        quran_index: QuranIndex | None = None,
    ):
        self._quran = quran_index
        self._lexicon = lexicon
        self._diacritize = diacritizer
        self._max_chars = max_segment_chars

    def prepare_sentence(self, sentence: str) -> list[Segment]:
        segments: list[Segment] = []
        for kind, piece, ref in split_quran(sentence):
            if kind == "quran":
                verse = "﴿" + piece.strip("{}﴿﴾ ") + "﴾"  # one display style for both formats
                segments.append(Segment("quran", verse, verse, self._identify(verse, ref)))
                continue
            clean = normalize(piece)
            if not clean:
                continue
            for part in split_long(clean, self._max_chars):
                segments.append(Segment("speech", part, self._to_spoken(part)))
        return segments

    def prepare(self, text: str) -> list[Segment]:
        return [seg for sentence in split_sentences(text) for seg in self.prepare_sentence(sentence)]

    def _identify(self, verse: str, written_ref: QuranRef | None) -> QuranRef | None:
        """The verse text is the ground truth; a written reference is the fallback."""
        found = self._quran.find(verse) if self._quran else None
        if found is None:
            return written_ref
        ref = QuranRef(*found)
        if written_ref and written_ref != ref:
            logger.warning("Reference %s does not match verse text; using %s", written_ref, ref)
        return ref

    def _to_spoken(self, text: str) -> str:
        protected, restore = self._lexicon.protect(text)
        return restore(self._diacritize(protected))