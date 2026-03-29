"""Chatterbox TTS wrapper (Resemble AI)."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

PIP_EXTRA = "chatterbox"


def is_available() -> bool:
    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError:
        return False


class ChatterboxTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Chatterbox",
            model_id="chatterbox",
            supports_voice_cloning=True,
            supports_emotions=True,
            available_voices=[],
            estimated_vram_gb=4.0,
            native_sample_rate=24000,
            description=(
                "350M param voice cloning model by Resemble AI. "
                "Quality: A — excellent quality, fast, lightweight. "
                "MIT licence. 23 languages. Emotion exaggeration control. "
                "Note: requires PyTorch 2.6 — may conflict with newer versions."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from chatterbox.tts import ChatterboxTTS as ChatterboxEngine

        self._model = ChatterboxEngine.from_pretrained(device=device)
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
            kwargs["audio_prompt_path"] = str(reference_audio)

        wav = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        audio = wav.detach().cpu().numpy().flatten().astype(np.float32)
        sr = self._model.sr

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = ChatterboxTTS
