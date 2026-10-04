from .base import TTSEngine

__all__ = ["TTSEngine", "build_engine"]


def build_engine(name: str, config) -> TTSEngine:
    """Engines import their heavy deps lazily, so tests and tooling stay light."""
    if name == "xtts":
        from .xtts import XTTSEngine

        return XTTSEngine(config)
    raise ValueError(f"Unknown engine {name!r}")
