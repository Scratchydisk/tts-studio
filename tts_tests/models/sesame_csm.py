"""Sesame CSM-1B wrapper.

Sesame CSM has no pip package — install via:
    pip install git+https://github.com/SesameAILabs/csm.git
"""

import logging
import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

logger = logging.getLogger(__name__)

PIP_EXTRA = "sesame-csm"


def is_available() -> bool:
    try:
        import csm  # noqa: F401
        return True
    except ImportError:
        return False


class SesameCSM(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Sesame CSM-1B",
            model_id="sesame-csm-1b",
            supports_voice_cloning=False,
            available_voices=[],
            estimated_vram_gb=4.5,
            native_sample_rate=24000,
            description=(
                "Quality: A — 1.1B param conversational speech model by Sesame. "
                "Optimised for dialogue with voice cloning via speaker audio samples. "
                "English only. Apache 2.0 licence. "
                "Install: pip install git+https://github.com/SesameAILabs/csm.git"
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from csm import CSM

        self._model = CSM("sesame/csm-1b", device=device)
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

        start = time.perf_counter()

        kwargs = {
            "text": text,
            "speaker": 0,
            "context": [],
        }

        audio_tensor = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        audio = audio_tensor.cpu().numpy().flatten().astype(np.float32)
        sr = 24000

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = SesameCSM
