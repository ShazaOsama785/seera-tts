from pathlib import Path

import pytest

from seera_tts.config import TTSConfig
from seera_tts.text import QuranIndex, QuranRef, TextPreparer
from seera_tts.text.chunker import split_quran
from seera_tts.text.diacritizer import NoOpDiacritizer
from seera_tts.text.lexicon import PronunciationLexicon
from seera_tts.text.quran import SURAH_NAMES, surah_number

MINI = Path(__file__).parent / "data" / "quran_mini.txt"


def make_preparer(with_index: bool = True) -> TextPreparer:
    index = QuranIndex.from_tanzil(MINI) if with_index else None
    return TextPreparer(PronunciationLexicon({}), NoOpDiacritizer(), 150, index)


# ------------------------------------------------------------ surah names
def test_surah_lookup():
    assert len(SURAH_NAMES) == 114
    for name in ["العلق", "سورة العلق", "عَلَق", "٩٦", "96", "اقرأ"]:
        assert surah_number(name) == 96
    assert surah_number("الاسراء") == surah_number("بني إسرائيل") == 17
    assert surah_number("ال عمران") == 3
    assert surah_number("غير موجودة") is None and surah_number("200") is None


# ------------------------------------------------- reading the source format
def test_split_quran_reads_source_reference():
    text = "ثم أرسلني فقال: { اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ } [ العلق: 1: 3 ] )، فرجع."
    pieces = split_quran(text)
    assert [p[0] for p in pieces] == ["speech", "quran", "speech"]
    assert pieces[1][2] == QuranRef(96, 1, 3)
    assert split_quran("﴿ثاني اثنين﴾ [التوبة: 40]")[0][2] == QuranRef(9, 40, 40)
    assert split_quran("{ آية } [سورة غير موجودة: 5]")[0][2] is None


def test_preparer_source_format_end_to_end():
    segs = make_preparer(with_index=False).prepare("فقال: { اقْرَأْ } [ العلق: 1: 3 ] )، فرجع بها.")
    assert [(s.kind, s.display_text) for s in segs] == [
        ("speech", "فقال:"), ("quran", "﴿اقْرَأْ﴾"), ("speech", "فرجع بها.")
    ]
    assert segs[1].quran_ref == QuranRef(96, 1, 3)


# --------------------------------------------------- finding verses by text
def test_index_finds_verse_by_text():
    q = QuranIndex.from_tanzil(MINI)
    verse = "{ اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ خَلَقَ الإِْنسَانَ مِنْ عَلَقٍ اقْرَأْ وَرَبُّكَ الأَْكْرَمُ }"
    assert q.find(verse) == (96, 1, 3)
    assert q.find("﴿إِذْ يَقُولُ لِصَاحِبِهِ لَا تَحْزَنْ إِنَّ اللَّهَ مَعَنَا﴾") == (9, 40, 40)  # part of an ayah
    assert q.find("﴿اقرأ﴾") is None  # too short to be sure
    assert q.find("نص ليس من القرآن الكريم أبدا") is None


def test_preparer_identifies_verse_without_reference():
    segs = make_preparer().prepare("فقال: ﴿اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ﴾ فرجع.")
    assert segs[1].quran_ref == QuranRef(96, 1, 1)


def test_text_beats_wrong_written_reference():
    segs = make_preparer().prepare("فقال: { اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ } [البقرة: 5]")
    assert segs[1].quran_ref == QuranRef(96, 1, 1)


# ----------------------------------------------------------- the real file
def test_real_quran_file_is_complete():
    path = TTSConfig().quran_text_path
    if not path.exists():
        pytest.skip("download quran-simple-clean.txt from tanzil.net first")
    assert len(QuranIndex.from_tanzil(path)) == 6236