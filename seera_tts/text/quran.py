"""Surah name -> number lookup, tolerant of spelling variants in source texts."""

from __future__ import annotations

import bisect
import re

SURAH_NAMES = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف", "الأنفال",
    "التوبة", "يونس", "هود", "يوسف", "الرعد", "إبراهيم", "الحجر", "النحل", "الإسراء",
    "الكهف", "مريم", "طه", "الأنبياء", "الحج", "المؤمنون", "النور", "الفرقان", "الشعراء",
    "النمل", "القصص", "العنكبوت", "الروم", "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر",
    "يس", "الصافات", "ص", "الزمر", "غافر", "فصلت", "الشورى", "الزخرف", "الدخان",
    "الجاثية", "الأحقاف", "محمد", "الفتح", "الحجرات", "ق", "الذاريات", "الطور", "النجم",
    "القمر", "الرحمن", "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة", "الصف",
    "الجمعة", "المنافقون", "التغابن", "الطلاق", "التحريم", "الملك", "القلم", "الحاقة",
    "المعارج", "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان", "المرسلات",
    "النبأ", "النازعات", "عبس", "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج",
    "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد", "الشمس", "الليل", "الضحى", "الشرح",
    "التين", "العلق", "القدر", "البينة", "الزلزلة", "العاديات", "القارعة", "التكاثر",
    "العصر", "الهمزة", "الفيل", "قريش", "الماعون", "الكوثر", "الكافرون", "النصر",
    "المسد", "الإخلاص", "الفلق", "الناس",
]

# Other names used in classical books
_ALIASES = {
    "براءة": 9, "بني إسرائيل": 17, "المؤمن": 40, "حم السجدة": 41, "القتال": 47,
    "الدهر": 76, "عم": 78, "التطفيف": 83, "الانشراح": 94, "اقرأ": 96, "الزلزال": 99,
    "اللهب": 111, "تبت": 111,
}

_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")


def _key(name: str) -> str:
    """Normalize a surah name so spelling variants compare equal."""
    name = _DIACRITICS.sub("", name)
    name = re.sub(r"[أإآٱ]", "ا", name).replace("ة", "ه").replace("ى", "ي")
    name = re.sub(r"^سوره\s+", "", name.strip())
    return re.sub(r"\s+", " ", name)


def _build_index() -> dict[str, int]:
    index: dict[str, int] = {}
    named = [(n, i + 1) for i, n in enumerate(SURAH_NAMES)] + list(_ALIASES.items())
    for name, number in named:
        key = _key(name)
        index[key] = number
        if key.startswith("ال") and len(key) > 3:
            index.setdefault(key[2:], number)  # "علق" also matches العلق
    return index


_INDEX = _build_index()


def surah_number(name: str) -> int | None:
    """'العلق', 'سورة العلق', 'علق', '96', '٩٦' -> 96. Unknown -> None."""
    name = name.strip()
    if name.isdigit():
        n = int(name)
        return n if 1 <= n <= 114 else None
    return _INDEX.get(_key(name))


# ------------------------------------------------------------ text matching
_NON_LETTERS = re.compile(r"[^\u0621-\u064a]")


def _letters(text: str) -> str:
    """Letters only: no tashkeel, spaces, or punctuation; unified alef/ya/ta marbuta."""
    text = _DIACRITICS.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return _NON_LETTERS.sub("", text)


class QuranIndex:
    """Finds which surah/ayahs a quoted verse comes from, by its text alone."""

    def __init__(self, ayahs: list[tuple[int, int, str]]):
        self._starts: list[int] = []
        self._keys: list[tuple[int, int]] = []
        parts, pos = [], 0
        for surah, ayah, text in ayahs:
            key = _letters(text)
            self._starts.append(pos)
            self._keys.append((surah, ayah))
            parts.append(key)
            pos += len(key)
        self._text = "".join(parts)

    @classmethod
    def from_tanzil(cls, path) -> "QuranIndex":
        """Tanzil 'Text (with aya numbers)' file: one 'surah|ayah|text' per line."""
        ayahs = []
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            surah, ayah, text = line.split("|", 2)
            ayahs.append((int(surah), int(ayah), text))
        return cls(ayahs)

    def __len__(self) -> int:
        return len(self._keys)

    def find(self, verse: str, min_letters: int = 12) -> tuple[int, int, int] | None:
        """(surah, first_ayah, last_ayah), or None if not found, too short, or ambiguous."""
        key = _letters(verse)
        if len(key) < min_letters:
            return None  # short phrases like "الحمد لله" appear in many places
        first = self._text.find(key)
        if first == -1 or self._text.find(key, first + 1) != -1:
            return None  # not in the Quran text, or appears more than once
        i = bisect.bisect_right(self._starts, first) - 1
        j = bisect.bisect_right(self._starts, first + len(key) - 1) - 1
        (s1, a1), (s2, a2) = self._keys[i], self._keys[j]
        return (s1, a1, a2) if s1 == s2 else None