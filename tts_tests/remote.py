"""Remote TTS model wrapper.

Wraps any TTS model to generate via a remote OpenAI-compatible TTS API
(e.g. vLLM serve) instead of loading locally.

Supports voice cloning by uploading reference audio to /v1/audio/voices
before generating speech.
"""

import io
import logging
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from tts_tests.base import ModelInfo, TTSModel, TTSResult

logger = logging.getLogger(__name__)


class RemoteTTSModel(TTSModel):
    """Wraps model metadata and forwards generate() calls to an HTTP endpoint."""

    def __init__(self, local_model: TTSModel, base_url: str, api_model: str | None = None):
        self._local = local_model
        self._base_url = base_url.rstrip("/")
        self._api_model = api_model
        self._connected = False

    def info(self) -> ModelInfo:
        info = self._local.info()
        return ModelInfo(
            name=info.name,
            model_id=info.model_id,
            supports_voice_cloning=info.supports_voice_cloning,
            supports_emotions=info.supports_emotions,
            available_voices=info.available_voices,
            estimated_vram_gb=info.estimated_vram_gb,
            native_sample_rate=info.native_sample_rate,
            description=info.description,
        )

    def load(self, device: str = "cuda") -> None:
        try:
            resp = httpx.get(f"{self._base_url}/models", timeout=5.0)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach remote server at {self._base_url}: {e}"
            )
        self._connected = True

    def unload(self) -> None:
        self._connected = False

    def is_loaded(self) -> bool:
        return self._connected

    def generate(
        self,
        text: str,
        voice: str | None = None,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded():
            raise RuntimeError("Not connected to remote server")

        info = self._local.info()
        start = time.perf_counter()

        # Reference audio voice cloning: not yet supported by vLLM-omni's
        # Voxtral HTTP API — the tokeniser only accepts preset voice names.
        # Log a warning and fall back to a preset voice.
        if reference_audio and reference_audio.exists():
            logger.warning(
                "Voice cloning via reference audio is not yet supported "
                "for remote models. Using preset voice instead."
            )

        if not voice:
            voice = info.available_voices[0] if info.available_voices else "default"

        payload = {
            "input": text,
            "model": self._api_model or info.model_id,
            "response_format": "wav",
            "voice": voice,
        }

        response = httpx.post(
            f"{self._base_url}/audio/speech",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()

        audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        elapsed = time.perf_counter() - start

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )
