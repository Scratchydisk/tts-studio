"""OuteTTS-0.3-500M wrapper — pure language modelling approach to TTS."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

PIP_EXTRA = "outetts"


def is_available() -> bool:
    try:
        import outetts  # noqa: F401
        return True
    except ImportError:
        return False


class OuteTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="OuteTTS-0.3-500M",
            model_id="outetts-0.3-500m",
            supports_voice_cloning=True,
            available_voices=[],
            estimated_vram_gb=2.0,
            native_sample_rate=24000,
            description=(
                "500M param pure language modelling TTS. Quality: B — decent "
                "quality, very lightweight with GGUF support. Supports 6 "
                "languages (en/ja/ko/zh/fr/de). Apache 2.0 licence."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        import outetts

        interface = outetts.Interface(
            model_version="0.3",
            cfg=outetts.HFModelConfig_v3(
                model_path="OuteAI/OuteTTS-0.3-500M",
                tokenizer_path="OuteAI/OuteTTS-0.3-500M",
            ),
        )
        self._model = interface
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
            "temperature": 0.4,
            "max_length": 4096,
        }

        if reference_audio:
            speaker = self._model.create_speaker(str(reference_audio))
            kwargs["speaker"] = speaker

        output = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        audio = np.array(output.audio).flatten().astype(np.float32)
        sr = output.sr

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = OuteTTS
