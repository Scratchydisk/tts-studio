"""Orpheus-3B standalone worker — runs in the vLLM venv.

Provides an HTTP API compatible with OpenAI's /v1/audio/speech endpoint.
Uses the orpheus-speech package which bundles vLLM inference and SNAC
audio decoding internally.

Usage (from the vLLM venv):
    pip install orpheus-speech
    python workers/orpheus_worker.py --port 8001 --gpu 0

TTS Studio manages this process via the Models tab.
"""

import argparse
import io
import logging
import os
import time
import wave

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("orpheus-worker")

MODEL_NAME = "canopylabs/orpheus-3b-0.1-ft"
SAMPLE_RATE = 24000
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Global state
_model = None


def load_model(gpu_memory_utilization: float = 0.85):
    """Load the Orpheus model."""
    global _model
    from orpheus_tts import OrpheusModel

    # Patch OrpheusModel to pass gpu_memory_utilization to vLLM
    _orig_setup = OrpheusModel._setup_engine

    def _patched_setup(self):
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    OrpheusModel._setup_engine = _patched_setup

    logger.info(
        "Loading model %s (gpu_memory_utilization=%.2f) — "
        "this will download weights on first run (~6 GB)...",
        MODEL_NAME, gpu_memory_utilization,
    )
    _model = OrpheusModel(model_name=MODEL_NAME)
    OrpheusModel._setup_engine = _orig_setup
    logger.info("Model loaded and ready.")


_request_counter = 0


def generate_speech(text: str, voice: str = "tara") -> tuple[bytes, int]:
    """Generate speech, returning (wav_bytes, sample_rate)."""
    global _request_counter
    _request_counter += 1
    request_id = f"req-{_request_counter:06d}"

    start = time.perf_counter()

    from orpheus_tts.engine_class import tokens_decoder_sync
    token_gen = _model.generate_tokens_sync(prompt=text, voice=voice, request_id=request_id)
    audio_chunks = tokens_decoder_sync(token_gen)

    # Collect all PCM chunks (int16 bytes at 24kHz)
    pcm_data = b""
    for chunk in audio_chunks:
        pcm_data += chunk

    if not pcm_data:
        raise RuntimeError("Model returned empty audio")

    elapsed = time.perf_counter() - start
    duration = len(pcm_data) / (SAMPLE_RATE * 2)  # 2 bytes per int16 sample
    logger.info("Generated %.1fs audio in %.1fs (%.1fx realtime)",
                duration, elapsed, duration / max(elapsed, 1e-6))

    # Wrap raw PCM in a WAV container
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    return buf.getvalue(), SAMPLE_RATE


def create_app():
    """Create the FastAPI application."""
    from fastapi import FastAPI, Form
    from fastapi.responses import Response

    app = FastAPI(title="Orpheus Worker")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model"}],
        }

    from fastapi import Request
    from pydantic import BaseModel

    class SpeechRequest(BaseModel):
        input: str
        voice: str = "tara"
        model: str = MODEL_NAME
        response_format: str = "wav"

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        """OpenAI-compatible speech endpoint — accepts JSON or form data."""
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            req = SpeechRequest(**data)
        else:
            form = await request.form()
            req = SpeechRequest(**form)

        wav_bytes, sr = generate_speech(req.input, voice=req.voice)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sr)},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Orpheus-3B worker server")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port to listen on (default: 8001)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    parser.add_argument("--gpu-mem", type=float, default=0.85,
                        help="GPU memory utilization for vLLM (default: 0.85)")
    args = parser.parse_args()

    # Pin to requested GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    load_model(gpu_memory_utilization=args.gpu_mem)

    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
