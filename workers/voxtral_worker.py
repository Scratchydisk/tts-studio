"""Voxtral-4B standalone worker — runs in the vLLM venv.

Provides an HTTP API compatible with OpenAI's /v1/audio/speech endpoint,
plus support for voice cloning via reference audio upload.

Usage (from the vLLM venv):
    python workers/voxtral_worker.py --port 8000 --gpu 0

TTS Studio manages this process via the Models tab.
"""

import argparse
import io
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voxtral-worker")

MODEL_NAME = "mistralai/Voxtral-4B-TTS-2603"

# Global state
_llm = None
_tokenizer = None


def load_model():
    """Load the Voxtral model and tokenizer."""
    global _llm, _tokenizer

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from vllm import SamplingParams  # noqa: F401
    from vllm_omni.entrypoints.omni import Omni

    logger.info("Loading tokenizer...")
    _tokenizer = MistralTokenizer.from_hf_hub(MODEL_NAME)

    logger.info("Loading model %s — this will download weights on first run (~8 GB)...", MODEL_NAME)
    _llm = Omni(model=MODEL_NAME)
    logger.info("Model loaded and ready.")


def generate_speech(text: str, voice: str = "neutral_female",
                    ref_audio_bytes: bytes | None = None) -> tuple[bytes, int]:
    """Generate speech, returning (wav_bytes, sample_rate).

    If ref_audio_bytes is provided, uses voice cloning instead of preset voice.
    """
    from mistral_common.protocol.speech.request import SpeechRequest
    from vllm import SamplingParams

    instruct_tokenizer = _tokenizer.instruct_tokenizer
    start = time.perf_counter()

    if ref_audio_bytes:
        tokenized = instruct_tokenizer.encode_speech_request(
            SpeechRequest(input=text, ref_audio=ref_audio_bytes)
        )
        inputs = {
            "prompt_token_ids": tokenized.tokens,
            "multi_modal_data": {
                "audio": [(
                    tokenized.audios[0].audio_array,
                    tokenized.audios[0].sampling_rate,
                )]
            },
        }
    else:
        tokenized = instruct_tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice=voice)
        )
        inputs = {
            "prompt_token_ids": tokenized.tokens,
            "additional_information": {"voice": [voice]},
        }

    sampling_params_list = [
        SamplingParams(max_tokens=4096),
        SamplingParams(max_tokens=4096),
    ]

    outputs = _llm.generate(inputs, sampling_params_list=sampling_params_list)

    # Extract audio bytes from output
    import soundfile as sf

    if hasattr(outputs, "audio"):
        audio_bytes = outputs.audio
    elif hasattr(outputs, "outputs"):
        audio_data = outputs.outputs[0]
        if hasattr(audio_data, "audio"):
            audio_bytes = audio_data.audio
        else:
            raise RuntimeError(f"Unexpected output format: {type(audio_data)}")
    else:
        raise RuntimeError(f"Unexpected output format: {type(outputs)}")

    # Read to get sample rate, then return raw wav
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    elapsed = time.perf_counter() - start
    logger.info("Generated %.1fs audio in %.1fs (%.1fx realtime)",
                len(audio) / sr, elapsed, len(audio) / sr / elapsed)

    # Re-encode as WAV
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="wav")
    return buf.getvalue(), sr


def create_app():
    """Create the FastAPI application."""
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import Response

    app = FastAPI(title="Voxtral Worker")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model"}],
        }

    @app.post("/v1/audio/speech")
    def speech(
        input: str = Form(...),
        voice: str = Form("neutral_female"),
        model: str = Form(MODEL_NAME),
        response_format: str = Form("wav"),
        reference_audio: UploadFile | None = File(None),
    ):
        """OpenAI-compatible speech endpoint with optional reference audio."""
        ref_bytes = None
        if reference_audio:
            ref_bytes = reference_audio.file.read()
            logger.info("Voice cloning with %d bytes of reference audio", len(ref_bytes))

        wav_bytes, sr = generate_speech(input, voice=voice, ref_audio_bytes=ref_bytes)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sr)},
        )

    # Also support JSON body for compatibility with existing remote.py
    from pydantic import BaseModel

    class SpeechRequestJSON(BaseModel):
        input: str
        voice: str = "neutral_female"
        model: str = MODEL_NAME
        response_format: str = "wav"

    @app.post("/v1/audio/speech/json")
    def speech_json(req: SpeechRequestJSON):
        """JSON body variant (no reference audio — use the form endpoint for cloning)."""
        wav_bytes, sr = generate_speech(req.input, voice=req.voice)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sr)},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Voxtral-4B worker server")
    parser.add_argument("--port", type=int, default=8100,
                        help="Port to listen on (default: 8100)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    args = parser.parse_args()

    # Pin to requested GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    load_model()

    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
