"""Narration pipeline: text (full or streamed from an LLM) -> stream of events + audio."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .audio import pcm16_to_wav, silence, to_pcm16
from .config import TTSConfig
from .engines import TTSEngine, build_engine
from .text import PronunciationLexicon, QuranIndex, Segment, SentenceStream, TextPreparer, build_diacritizer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ events
@dataclass(frozen=True)
class SegmentStarted:
    """A spoken segment is about to play - the UI can highlight `text`."""

    index: int
    text: str


@dataclass(frozen=True)
class QuranSegment:
    """Quranic text is never machine-voiced. The client shows it and plays the recitation."""

    index: int
    text: str
    audio_urls: tuple[str, ...] = ()  # one per ayah; empty if the verse wasn't identified


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes  # 16-bit mono PCM at pipeline.sample_rate


StreamEvent = SegmentStarted | QuranSegment | AudioChunk


# ---------------------------------------------------------------- pipeline
class NarrationPipeline:
    def __init__(self, engine: TTSEngine, preparer: TextPreparer, config: TTSConfig):
        self.engine = engine
        self.cfg = config
        self._preparer = preparer
        self._pause = to_pcm16(silence(config.sentence_pause_ms, engine.sample_rate))
        self._wav_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()

    @classmethod
    def from_config(cls, config: TTSConfig) -> "NarrationPipeline":
        """Build everything from the config: Quran index, lexicon, diacritizer, engine."""
        quran_index = None
        if config.quran_text_path.exists():
            quran_index = QuranIndex.from_tanzil(config.quran_text_path)
        else:
            logger.warning("No Quran text at %s - verses need written references.", config.quran_text_path)
        preparer = TextPreparer(
            lexicon=PronunciationLexicon.from_json(config.lexicon_path),
            diacritizer=build_diacritizer(config.diacritizer),
            max_segment_chars=config.max_segment_chars,
            quran_index=quran_index,
        )
        engine = build_engine(config.engine, config)
        engine.load()
        return cls(engine, preparer, config)

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    def stream(self, text: str | Iterable[str], voice: str | None = None) -> Iterator[StreamEvent]:
        """Stream events for a full text, or for text deltas arriving from an LLM.

        Audio for each sentence starts as soon as that sentence is complete.
        """
        voice = voice or self.cfg.default_voice
        deltas = [text] if isinstance(text, str) else text
        sentences = SentenceStream()
        index = 0

        def speak(sentence: str) -> Iterator[StreamEvent]:
            nonlocal index
            for segment in self._preparer.prepare_sentence(sentence):
                yield from self._render(segment, index, voice)
                index += 1

        for delta in deltas:
            for sentence in sentences.feed(delta):
                yield from speak(sentence)
        for sentence in sentences.flush():
            yield from speak(sentence)

    def synthesize_wav(self, text: str, voice: str | None = None) -> bytes:
        """Whole narration as a WAV file (LRU-cached - chapter narrations repeat a lot)."""
        key = (text, voice or self.cfg.default_voice)
        if key in self._wav_cache:
            self._wav_cache.move_to_end(key)
            return self._wav_cache[key]

        pcm = b"".join(e.pcm for e in self.stream(text, voice) if isinstance(e, AudioChunk))
        wav = pcm16_to_wav(pcm, self.sample_rate)
        self._wav_cache[key] = wav
        if len(self._wav_cache) > self.cfg.wav_cache_size:
            self._wav_cache.popitem(last=False)
        return wav

    def _recitation_urls(self, segment: Segment) -> tuple[str, ...]:
        if segment.quran_ref is None:
            return ()
        base, reciter = self.cfg.quran_audio_base.rstrip("/"), self.cfg.quran_reciter
        return tuple(f"{base}/{reciter}/{s:03d}{a:03d}.mp3" for s, a in segment.quran_ref.ayahs())

    def _render(self, segment: Segment, index: int, voice: str) -> Iterator[StreamEvent]:
        if segment.kind == "quran":
            yield QuranSegment(index, segment.display_text, self._recitation_urls(segment))
            return
        yield SegmentStarted(index, segment.display_text)
        for chunk in self.engine.stream(segment.spoken_text, voice):
            yield AudioChunk(to_pcm16(chunk))
        yield AudioChunk(self._pause)