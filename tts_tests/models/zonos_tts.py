"""Zonos-v0.1 wrapper — emotion and prosody control TTS by Zyphra.

Zonos has no pip package; install via:
    pip install git+https://github.com/Zyphra/Zonos.git
"""

import logging
import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

logger = logging.getLogger(__name__)

PIP_EXTRA = "zonos"


def is_available() -> bool:
    try:
        import zonos  # noqa: F401
        return True
    except ImportError:
        return False


class ZonosTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Zonos-v0.1",
            model_id="zonos-v0.1",
            supports_voice_cloning=True,
            supports_emotions=True,
            available_voices=[],
            estimated_vram_gb=6.0,
            native_sample_rate=44100,
            description=(
                "1.6B param model with unique emotion and prosody controls. "
                "Quality: A — high quality with fine-grained expressive "
                "conditioning. Supports 5 languages (en/ja/zh/fr/de). "
                "Apache 2.0 licence."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from zonos.model import Zonos

        self._model = Zonos.from_pretrained(
            "Zyphra/Zonos-v0.1-transformer", device=device,
        )
        self._device = device

    def unload(self) -> None:
        del self._model
        self._model = None
        self._device = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        import torch
        import torchaudio

        start = time.perf_counter()

        if reference_audio:
            wav, sr = torchaudio.load(str(reference_audio))
            speaker = self._model.make_speaker_embedding(wav, sr)
        else:
            # Generate a default speaker embedding from silence as fallback
            logger.warning(
                "No reference audio provided; Zonos works best with a "
                "10-30s reference clip. Using default conditioning.",
            )
            speaker = self._model.make_speaker_embedding(
                torch.zeros(1, 16000), 16000,
            )

        cond_dict = self._model.prepare_conditioning(text=text, speaker=speaker)
        codes = self._model.generate(cond_dict)
        wavs = self._model.autoencoder.decode(codes).cpu()
        elapsed = time.perf_counter() - start

        audio = wavs.numpy().flatten().astype(np.float32)
        sr = 44100

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = ZonosTTS
