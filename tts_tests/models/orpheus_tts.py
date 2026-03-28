"""Orpheus-3B TTS wrapper."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


def is_available() -> bool:
    try:
        import orpheus_tts  # noqa: F401
        return True
    except ImportError:
        return False


class OrpheusTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Orpheus-3B",
            model_id="orpheus-3b",
            supports_voice_cloning=False,
            available_voices=VOICES,
            estimated_vram_gb=7.0,
            native_sample_rate=24000,
            description=(
                "Fine-tuned 3B param model with high quality output. "
                "Requires vllm. Needs ~7GB VRAM."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from orpheus_tts import OrpheusModel
        self._model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
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

        voice = voice or "tara"
        start = time.perf_counter()

        audio_chunks = self._model.generate_speech(
            prompt=text,
            voice=voice,
        )

        chunks = []
        for chunk in audio_chunks:
            chunks.append(np.array(chunk, dtype=np.float32))

        if not chunks:
            raise RuntimeError("No audio generated")

        audio = np.concatenate(chunks)
        elapsed = time.perf_counter() - start

        return TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=len(audio) / 24000,
            generation_time=elapsed,
        )


MODEL_CLASS = OrpheusTTS
