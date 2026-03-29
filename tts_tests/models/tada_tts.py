"""TADA-1B wrapper.

TADA by Hume AI has no standard pip package — install via:
    pip install git+https://github.com/HumeAI/tada.git
"""

import logging
import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

logger = logging.getLogger(__name__)

PIP_EXTRA = "tada"


def is_available() -> bool:
    try:
        import tada  # noqa: F401
        return True
    except ImportError:
        return False


class TadaTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="TADA-1B",
            model_id="tada-1b",
            supports_voice_cloning=True,
            available_voices=[],
            estimated_vram_gb=5.0,
            native_sample_rate=24000,
            description=(
                "Quality: A — 1B param zero-hallucination TTS by Hume AI. "
                "Supports long-form synthesis up to 700 seconds with fast RTF of 0.09. "
                "Voice cloning supported. MIT licence (+ Llama 3.2 Community License). "
                "Install: pip install git+https://github.com/HumeAI/tada.git"
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from tada import TADA

        self._model = TADA.from_pretrained("HumeAI/tada-1b", device=device)
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

        kwargs = {"text": text}
        if reference_audio:
            kwargs["speaker_audio"] = str(reference_audio)

        audio_output = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        # Convert output to numpy float32 (handles both tensor and array)
        if hasattr(audio_output, "cpu"):
            audio = audio_output.cpu().numpy().flatten().astype(np.float32)
        else:
            audio = np.array(audio_output).flatten().astype(np.float32)

        sr = 24000

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = TadaTTS
