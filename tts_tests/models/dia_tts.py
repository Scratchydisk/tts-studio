"""Dia-1.6B TTS wrapper."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

PIP_EXTRA = "dia"


def is_available() -> bool:
    try:
        import dia  # noqa: F401
        return True
    except ImportError:
        return False


class DiaTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Dia-1.6B",
            model_id="dia-1.6b",
            supports_voice_cloning=False,
            supports_emotions=True,
            available_voices=["[S1]", "[S2]"],
            estimated_vram_gb=10.0,
            native_sample_rate=44100,
            description=(
                "Expressive conversational TTS with multi-speaker dialogue. "
                "Use [S1]/[S2] tags for speaker turns. Supports emotion tags like "
                "(laughs), (sighs), etc. Requires ~10GB VRAM (float32 only — "
                "produces garbage at float16/bfloat16)."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        from dia.model import Dia
        self._model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32")
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

        # If no speaker tags, wrap with [S1]
        if "[S1]" not in text and "[S2]" not in text:
            text = f"[S1] {text}"

        import gc
        import torch

        start = time.perf_counter()
        output = self._model.generate(text, verbose=False)
        elapsed = time.perf_counter() - start

        audio = np.array(output).astype(np.float32)

        # Free generation intermediates that Dia holds onto
        gc.collect()
        torch.cuda.empty_cache()

        return TTSResult(
            audio=audio,
            sample_rate=44100,
            duration=len(audio) / 44100,
            generation_time=elapsed,
        )


MODEL_CLASS = DiaTTS
