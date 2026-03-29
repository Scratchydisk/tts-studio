"""Voxtral-4B TTS model definition.

This model is designed to run via a remote vLLM server.
Configure it in endpoints.json:
    {
        "voxtral-4b": {
            "url": "http://your-server:8000/v1",
            "model": "mistralai/Voxtral-4B-TTS-2603"
        }
    }

Start the server with:
    vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
"""

import time
from pathlib import Path

import numpy as np

from tts_tests.base import ModelInfo, TTSModel, TTSResult

VOICES = [
    # English
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    # French
    "fr_female",
    "fr_male",
    # Spanish
    "es_female",
    "es_male",
    # German
    "de_female",
    "de_male",
    # Italian
    "it_female",
    "it_male",
    # Portuguese
    "pt_female",
    "pt_male",
    # Dutch
    "nl_female",
    "nl_male",
    # Arabic
    "ar_male",
    # Hindi
    "hi_female",
    "hi_male",
]


def is_available() -> bool:
    # No local deps needed — runs via remote endpoint
    return False


class VoxtralTTS(TTSModel):
    def __init__(self):
        self._loaded = False

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Voxtral-4B",
            model_id="voxtral-4b",
            supports_voice_cloning=True,
            available_voices=VOICES,
            estimated_vram_gb=16.0,
            native_sample_rate=24000,
            description=(
                "Mistral's 4B param TTS model. 9 languages, 20 preset voices, "
                "voice cloning from ~3s of reference audio. Runs via vLLM worker "
                "(start from Models tab)."
            ),
        )

    def load(self, device: str = "cuda") -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        raise RuntimeError(
            "Voxtral requires a remote endpoint. "
            "Configure it in endpoints.json."
        )


MODEL_CLASS = VoxtralTTS
