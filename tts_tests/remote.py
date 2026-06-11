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
        # Try /models first (OpenAI standard), fall back to /health (Dia-TTS-Server)
        base = self._base_url
        server_root = base.rsplit("/v1", 1)[0] if "/v1" in base else base
        for url in [f"{base}/models", f"{server_root}/health"]:
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    self._connected = True
                    return
            except Exception:
                continue
        raise RuntimeError(
            f"Cannot reach remote server at {self._base_url}. "
            f"If you just started the server, it may still be loading the model — "
            f"check the server log in the Models tab and wait for 'Server is ready' before retrying."
        )

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

        if not voice:
            voice = info.available_voices[0] if info.available_voices else "default"

        # Try multipart upload with reference audio (supported by voxtral worker)
        if reference_audio and reference_audio.exists():
            response = self._generate_with_reference(text, voice, reference_audio)
        else:
            response = self._generate_json(text, voice, info)

        audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
        elapsed = time.perf_counter() - start

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )

    def _generate_json(self, text: str, voice: str, info) -> httpx.Response:
        """Standard JSON request (OpenAI-compatible)."""
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
        return response

    def _generate_with_reference(self, text: str, voice: str,
                                  reference_audio: Path) -> httpx.Response:
        """Multipart form request with reference audio for voice cloning."""
        with open(reference_audio, "rb") as f:
            files = {"reference_audio": (reference_audio.name, f, "audio/wav")}
            data = {
                "input": text,
                "voice": voice,
                "response_format": "wav",
            }
            response = httpx.post(
                f"{self._base_url}/audio/speech",
                data=data,
                files=files,
                timeout=300.0,
            )
        # If server doesn't support multipart (e.g. standard vLLM), fall back
        if response.status_code == 422:
            logger.warning(
                "Server does not support reference audio upload. "
                "Using preset voice instead."
            )
            from tts_tests.base import ModelInfo
            info = self._local.info()
            return self._generate_json(text, voice, info)
        response.raise_for_status()
        return response
