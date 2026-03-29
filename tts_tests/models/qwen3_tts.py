"""Qwen3-TTS-1.7B wrapper (Alibaba) — CustomVoice variant with preset speakers."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

VOICES = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

PIP_EXTRA = "qwen3tts"


def is_available() -> bool:
    try:
        import qwen_tts  # noqa: F401
        return True
    except ImportError:
        return False


class Qwen3TTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Qwen3-TTS-1.7B",
            model_id="qwen3-tts-1.7b",
            supports_voice_cloning=False,
            available_voices=VOICES,
            estimated_vram_gb=10.0,
            native_sample_rate=24000,
            description=(
                "1.7B param TTS model by Alibaba. "
                "Quality: S — top tier, streaming capable. "
                "Apache 2.0 licence. 10 languages "
                "(zh/en/ja/ko/de/fr/ru/pt/es/it). "
                "9 preset voices. CustomVoice variant (no cloning)."
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

        start = time.perf_counter()

        speaker = voice or VOICES[0]
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
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


MODEL_CLASS = Qwen3TTS
