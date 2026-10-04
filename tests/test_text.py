import pytest

def test_mishkal_keeps_punctuation():
    pytest.importorskip("mishkal")
    from seera_tts.text.diacritizer import MishkalDiacritizer
    out = MishkalDiacritizer()("خرج النبي. ثم عاد إلى مكة؟ نعم!")
    assert out.endswith("!") and ". " in out and "؟" in out
    assert "\x01" not in out