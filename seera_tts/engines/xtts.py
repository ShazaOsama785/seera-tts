"""XTTS-v2 engine with true token-level streaming and voice cloning."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import TTSConfig
from .base import TTSEngine

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


class XTTSEngine(TTSEngine):
    def __init__(self, config: TTSConfig):
        self.cfg = config
        self.sample_rate = config.sample_rate
        self._model = None
        self._voices: dict[str, tuple] = {}  # name -> (gpt_cond_latent, speaker_embedding)
        self._lock = threading.Lock()  # one GPU, one generation at a time

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        model_dir = Path(snapshot_download(self.cfg.model_repo, local_dir=self.cfg.model_dir))
        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_dir), use_deepspeed=self.cfg.use_deepspeed)
        device = self.cfg.device if torch.cuda.is_available() else "cpu"
        if device != self.cfg.device:
            logger.warning("CUDA not available - running on CPU (not real-time).")
        self._model = model.to(device).eval()

        self._load_voices()
        self._warmup()
        logger.info("XTTS ready on %s with voices: %s", device, self.voices())

    def _load_voices(self) -> None:
        import torch

        for name, clips in _discover_voices(self.cfg.voices_dir).items():
            cache = self.cfg.voices_dir / f".{name}.latents.pt"
            if cache.exists() and cache.stat().st_mtime > max(c.stat().st_mtime for c in clips):
                self._voices[name] = tuple(torch.load(cache))
                continue
            gpt_latent, speaker_emb = self._model.get_conditioning_latents(
                audio_path=[str(c) for c in clips], max_ref_length=30, sound_norm_refs=True
            )
            torch.save((gpt_latent, speaker_emb), cache)
            self._voices[name] = (gpt_latent, speaker_emb)

        if not self._voices:
            raise RuntimeError(
                f"No voices found in {self.cfg.voices_dir}. Add e.g. voices/narrator.wav "
                "(6-30s of clean, calm MSA storytelling)."
            )

    def _warmup(self) -> None:
        """First CUDA call is slow; pay that cost at startup, not on the first user."""
        for _ in self.stream("بسم الله نبدأ.", next(iter(self._voices))):
            pass

    # -------------------------------------------------------------- inference
    def voices(self) -> list[str]:
        return sorted(self._voices)

    def stream(self, text: str, voice: str) -> Iterator[np.ndarray]:
        import torch

        if self._model is None:
            raise RuntimeError("Engine not loaded - call load() first.")
        if voice not in self._voices:
            raise KeyError(f"Unknown voice {voice!r}. Available: {self.voices()}")

        gpt_latent, speaker_emb = self._voices[voice]
        with self._lock, torch.inference_mode():
            for chunk in self._model.inference_stream(
                text,
                self.cfg.language,
                gpt_latent,
                speaker_emb,
                stream_chunk_size=self.cfg.stream_chunk_size,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                speed=self.cfg.speed,
                enable_text_splitting=False,  # the pipeline already splits
            ):
                yield chunk.squeeze().float().cpu().numpy()


def _discover_voices(voices_dir: Path) -> dict[str, list[Path]]:
    """voices/narrator.wav -> 'narrator'; voices/narrator/*.wav -> 'narrator' (multi-clip)."""
    found: dict[str, list[Path]] = {}
    if not voices_dir.exists():
        return found
    for entry in sorted(voices_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _AUDIO_EXTENSIONS:
            found.setdefault(entry.stem, []).append(entry)
        elif entry.is_dir():
            clips = sorted(p for p in entry.iterdir() if p.suffix.lower() in _AUDIO_EXTENSIONS)
            if clips:
                found[entry.name] = clips
    return found
