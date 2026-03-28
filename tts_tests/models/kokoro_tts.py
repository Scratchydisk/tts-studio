"""Kokoro-82M TTS wrapper."""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def is_available() -> bool:
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False


class KokoroTTS(TTSModel):
    def __init__(self):
        self._pipeline = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Kokoro-82M",
            model_id="kokoro-82m",
            supports_voice_cloning=False,
            available_voices=VOICES,
            estimated_vram_gb=0.5,
            native_sample_rate=24000,
            description="Lightweight 82M param model with excellent clarity.",
        )

    def load(self, device: str = "cuda") -> None:
        from kokoro import KPipeline
        # Kokoro uses lang_code prefixes: 'a' = American English, 'b' = British English
        self._pipeline = KPipeline(lang_code="a")
        self._device = device

    def unload(self) -> None:
        self._pipeline = None
        self._device = None

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        voice = voice or "af_heart"
        start = time.perf_counter()

        # KPipeline returns a generator of (graphemes, phonemes, audio) tuples
        chunks = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice):
            if audio is not None:
                chunks.append(audio.numpy())

        if not chunks:
            raise RuntimeError("No audio generated")

        audio = np.concatenate(chunks)
        elapsed = time.perf_counter() - start

        return TTSResult(
            audio=audio.astype(np.float32),
            sample_rate=24000,
            duration=len(audio) / 24000,
            generation_time=elapsed,
        )


MODEL_CLASS = KokoroTTS
