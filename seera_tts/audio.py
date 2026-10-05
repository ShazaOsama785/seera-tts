"""Audio format helpers (stdlib only)."""

from __future__ import annotations

import io
import wave

import numpy as np


def to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> little-endian signed 16-bit PCM."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def silence(ms: int, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * ms / 1000), dtype=np.float32)


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()
