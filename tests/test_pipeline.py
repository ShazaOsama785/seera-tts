from pathlib import Path

import numpy as np

from seera_tts.config import TTSConfig
from seera_tts.engines import TTSEngine
from seera_tts.pipeline import AudioChunk, NarrationPipeline, QuranSegment, SegmentStarted
from seera_tts.text import QuranIndex, TextPreparer
from seera_tts.text.diacritizer import NoOpDiacritizer
from seera_tts.text.lexicon import PronunciationLexicon

MINI = Path(__file__).parent / "data" / "quran_mini.txt"


class FakeEngine(TTSEngine):
    """Stands in for XTTS: 3 tiny chunks per call, and remembers what it was asked to say."""

    sample_rate = 24_000

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def load(self):
        pass

    def voices(self):
        return ["narrator"]

    def stream(self, text, voice):
        self.calls.append((text, voice))
        for _ in range(3):
            yield np.full(240, 0.1, dtype=np.float32)


def make_pipeline(**config_overrides) -> tuple[NarrationPipeline, FakeEngine]:
    engine = FakeEngine()
    cfg = TTSConfig(default_voice="narrator", **config_overrides)
    preparer = TextPreparer(
        PronunciationLexicon({"يثرب": "يَثْرِب"}), NoOpDiacritizer(), 150, QuranIndex.from_tanzil(MINI)
    )
    return NarrationPipeline(engine, preparer, cfg), engine


def kinds(events):
    return [type(e).__name__ for e in events]


# -------------------------------------------------------------- event order
def test_events_for_one_sentence():
    pipeline, _ = make_pipeline()
    events = list(pipeline.stream("هاجر النبي إلى يثرب."))
    # announce -> 3 audio chunks -> 1 pause chunk
    assert kinds(events) == ["SegmentStarted"] + ["AudioChunk"] * 4
    assert events[0] == SegmentStarted(0, "هاجر النبي إلى يثرب.")


def test_engine_gets_spoken_text_and_ui_gets_display_text():
    pipeline, engine = make_pipeline()
    events = list(pipeline.stream("هاجر النبي إلى يثرب."))
    assert engine.calls == [("هاجر النبي إلى يَثْرِب.", "narrator")]  # lexicon applied
    assert events[0].text == "هاجر النبي إلى يثرب."  # clean text for highlighting


def test_indexes_count_up_across_sentences():
    pipeline, _ = make_pipeline()
    starts = [e for e in pipeline.stream("الجملة الأولى. الجملة الثانية. الثالثة.") if isinstance(e, SegmentStarted)]
    assert [s.index for s in starts] == [0, 1, 2]


# ---------------------------------------------------------------- the Quran
def test_quran_is_never_sent_to_the_engine():
    pipeline, engine = make_pipeline()
    verse = "{ اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ خَلَقَ الإِنسَانَ مِنْ عَلَقٍ اقْرَأْ وَرَبُّكَ الأَكْرَمُ }"
    events = list(pipeline.stream(f"فقال: {verse} [ العلق: 1: 3 ] فرجع."))
    quran = next(e for e in events if isinstance(e, QuranSegment))
    assert quran.audio_urls == tuple(
        f"https://everyayah.com/data/Alafasy_128kbps/09600{a}.mp3" for a in (1, 2, 3)
    )
    assert all("اقْرَأْ" not in text for text, _ in engine.calls)
    assert kinds(events).count("SegmentStarted") == 2  # the speech before and after


def test_reciter_comes_from_config():
    pipeline, _ = make_pipeline(quran_reciter="Husary_64kbps")
    quran = next(e for e in pipeline.stream("{ اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ }") if isinstance(e, QuranSegment))
    assert quran.audio_urls == ("https://everyayah.com/data/Husary_64kbps/096001.mp3",)


# ------------------------------------------------------- LLM streaming mode
def test_deltas_give_same_result_as_full_text():
    text = "خرج النبي من مكة. ووصل إلى المدينة."
    full, _ = make_pipeline()
    streamed, _ = make_pipeline()
    deltas = ["خرج الن", "بي من مك", "ة. ووصل إ", "لى المدينة."]
    assert list(full.stream(text)) == list(streamed.stream(deltas))


def test_audio_starts_before_all_text_arrives():
    pipeline, _ = make_pipeline()
    received = []

    def llm():
        for delta in ["الجملة الأولى. ", "الجملة الثانية."]:
            received.append(delta)
            yield delta

    first_audio = next(e for e in pipeline.stream(llm()) if isinstance(e, AudioChunk))
    assert first_audio and received == ["الجملة الأولى. "]  # 2nd delta not even requested yet


# ----------------------------------------------------------------- voice/wav
def test_voice_override_and_default():
    pipeline, engine = make_pipeline()
    list(pipeline.stream("جملة.", voice="other"))
    list(pipeline.stream("جملة."))
    assert [v for _, v in engine.calls] == ["other", "narrator"]


def test_synthesize_wav_is_cached():
    pipeline, engine = make_pipeline()
    first = pipeline.synthesize_wav("جملة واحدة.")
    second = pipeline.synthesize_wav("جملة واحدة.")
    assert first[:4] == b"RIFF" and first == second
    assert len(engine.calls) == 1  # second call came from the cache, no GPU work