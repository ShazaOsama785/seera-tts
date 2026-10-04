"""Central configuration. Every field can be overridden by an env var: SEERA_TTS_<FIELD_NAME>."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class TTSConfig:
    # Model
    engine: str = "xtts"
    model_repo: str = "coqui/XTTS-v2"
    model_dir: Path = Path("./models/xtts_v2")
    device: str = "cuda"
    use_deepspeed: bool = False
    language: str = "ar"

    # Voices: voices/<name>.wav or voices/<name>/*.wav (several clips = more stable voice)
    voices_dir: Path = Path("./voices")
    default_voice: str = "narrator"

    # Generation (lower temperature = steadier, calmer narration)
    temperature: float = 0.65
    top_p: float = 0.85
    repetition_penalty: float = 10.0
    speed: float = 0.95
    stream_chunk_size: int = 20  # smaller = lower first-audio latency, more overhead

    # Text processing
    max_segment_chars: int = 85  # XTTS's Arabic limit is 166; tashkeel ~doubles length
    diacritizer: str = "mishkal"  # "none" | "mishkal"
    lexicon_path: Path = _PACKAGE_DIR / "data" / "lexicon.json"    # Text processing
    

    # Audio
    sample_rate: int = 24_000  # XTTS output rate
    sentence_pause_ms: int = 280  # storytelling pacing between sentences

    # Server
    wav_cache_size: int = 128
    max_text_chars: int = 8_000

    @classmethod
    def from_env(cls, prefix: str = "SEERA_TTS_") -> "TTSConfig":
        overrides = {}
        for f in fields(cls):
            raw = os.getenv(prefix + f.name.upper())
            if raw is None:
                continue
            default = getattr(cls, f.name)
            if isinstance(default, bool):
                overrides[f.name] = raw.strip().lower() in {"1", "true", "yes"}
            elif isinstance(default, Path):
                overrides[f.name] = Path(raw)
            else:
                overrides[f.name] = type(default)(raw)
        return cls(**overrides)
