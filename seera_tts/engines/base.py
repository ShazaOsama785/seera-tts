"""Engine interface. Add a new model (F5-TTS, a cloud API, ...) by implementing this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class TTSEngine(ABC):
    sample_rate: int

    @abstractmethod
    def load(self) -> None:
        """Load weights and voices. Called once at startup."""

    @abstractmethod
    def voices(self) -> list[str]:
        """Names of the available voices."""

    @abstractmethod
    def stream(self, text: str, voice: str) -> Iterator[np.ndarray]:
        """Yield float32 mono audio chunks in [-1, 1] as soon as they are generated.

        Engines without native streaming can yield a single chunk; the pipeline
        still streams sentence by sentence.
        """

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        chunks = list(self.stream(text, voice))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
