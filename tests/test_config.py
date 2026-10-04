from seera_tts.config import TTSConfig


def test_defaults():
    cfg = TTSConfig()
    assert cfg.diacritizer == "mishkal"
    assert cfg.max_segment_chars == 85


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SEERA_TTS_SPEED", "0.9")
    monkeypatch.setenv("SEERA_TTS_USE_DEEPSPEED", "false")
    cfg = TTSConfig.from_env()
    assert cfg.speed == 0.9 and isinstance(cfg.speed, float)
    assert cfg.use_deepspeed is False