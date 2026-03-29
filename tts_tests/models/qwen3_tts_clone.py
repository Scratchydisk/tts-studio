"""Qwen3-TTS-1.7B wrapper (Alibaba) — Base variant for voice cloning."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

PIP_EXTRA = "qwen3tts"


def is_available() -> bool:
    try:
        import qwen_tts  # noqa: F401
        return True
    except ImportError:
        return False


class Qwen3TTSClone(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Qwen3-TTS-1.7B Clone",
            model_id="qwen3-tts-1.7b-clone",
            supports_voice_cloning=True,
            available_voices=[],
            estimated_vram_gb=10.0,
            native_sample_rate=24000,
            description=(
                "1.7B param TTS model by Alibaba. "
                "Quality: S — top tier, streaming capable. "
                "Apache 2.0 licence. 10 languages "
                "(zh/en/ja/ko/de/fr/ru/pt/es/it). "
                "Base variant: voice cloning from reference audio only."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            MODEL_NAME, device_map=device,
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

        if not reference_audio or not reference_audio.exists():
            raise RuntimeError(
                "Qwen3-TTS Clone requires reference audio. "
                "Upload a reference audio clip in the UI."
            )

        start = time.perf_counter()

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            ref_audio=str(reference_audio),
            ref_text=reference_text,
            non_streaming_mode=True,
        )

        elapsed = time.perf_counter() - start

        audio = wavs[0].flatten().astype(np.float32)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = Qwen3TTSClone
