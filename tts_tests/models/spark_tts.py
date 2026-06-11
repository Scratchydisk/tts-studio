"""Spark-TTS-0.5B wrapper.

Spark-TTS has no pip package, so we clone the repo on first use.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tts_tests.base import ModelInfo, TTSModel, TTSResult
from tts_tests.config import CACHE_DIR

logger = logging.getLogger(__name__)

SPARK_REPO_URL = "https://github.com/SparkAudio/Spark-TTS.git"
SPARK_DIR = CACHE_DIR / "Spark-TTS"


def _ensure_repo() -> bool:
    """Clone the Spark-TTS repo if not present. Return True if available."""
    if SPARK_DIR.exists() and (SPARK_DIR / "cli" / "SparkTTS.py").exists():
        return True
    # A previous run may have left a partial/incomplete clone; git refuses to
    # clone into a non-empty dir (exit 128), so clear it first to self-heal.
    if SPARK_DIR.exists():
        import shutil
        shutil.rmtree(SPARK_DIR, ignore_errors=True)
    try:
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", SPARK_REPO_URL, str(SPARK_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Failed to clone Spark-TTS (exit %s): %s",
            e.returncode,
            (e.stderr or "").strip(),
        )
        return False
    except Exception as e:
        logger.warning("Failed to clone Spark-TTS: %s", e)
        return False

PIP_EXTRA = "sparktts"


def is_available() -> bool:
    try:
        import einops, omegaconf, transformers  # noqa: F401
        return _ensure_repo()
    except ImportError:
        return False


class SparkTTS(TTSModel):
    def __init__(self):
        self._model = None
        self._device = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Spark-TTS-0.5B",
            model_id="spark-tts-0.5b",
            supports_voice_cloning=True,
            available_voices=[],
            estimated_vram_gb=2.0,
            native_sample_rate=16000,
            description="Efficient 0.5B param model with voice cloning support.",
        )

    def load(self, device: str = "cuda") -> None:
        if str(SPARK_DIR) not in sys.path:
            sys.path.insert(0, str(SPARK_DIR))

        from huggingface_hub import snapshot_download
        from cli.SparkTTS import SparkTTS as SparkTTSEngine

        model_dir = snapshot_download("SparkAudio/Spark-TTS-0.5B")

        self._model = SparkTTSEngine(
            model_dir=model_dir,
            device=device,
        )
        if "cuda" in str(device) and torch.cuda.is_bf16_supported():
            # bf16 halves LLM weights and KV cache; the fp32 stack overflows
            # 4 GB cards into shared memory and generation crawls. bf16 rather
            # than fp16 — the Qwen backbone overflows fp16 logits into NaN,
            # tripping a device-side assert in sampling. BiCodec stays fp32 —
            # it only receives integer token tensors from the LLM.
            self._model.model.bfloat16()
            # wav2vec2 (~1.2 GB fp32) only tokenizes reference audio, and
            # extract_wav2vec2_features routes tensors via its .device — on
            # small cards park it on CPU so long generations stop spilling
            # VRAM into shared memory
            gpu_index = torch.device(device).index or 0
            total_gb = torch.cuda.get_device_properties(gpu_index).total_memory / 2**30
            if total_gb < 6:
                self._model.audio_tokenizer.feature_extractor.to("cpu")

        # Captioning reuses one reference clip for every segment, but the
        # engine re-runs wav2vec2 + BiCodec over it on each call. Memoise per
        # path — the returned token tensors are only read downstream.
        orig_tokenize = self._model.audio_tokenizer.tokenize
        cache: dict[str, tuple] = {}

        def cached_tokenize(audio_path):
            key = str(audio_path)
            if key not in cache:
                cache.clear()  # keep at most one reference resident
                cache[key] = orig_tokenize(audio_path)
            return cache[key]

        self._model.audio_tokenizer.tokenize = cached_tokenize
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
            kwargs["prompt_speech_path"] = str(reference_audio)
        if reference_text:
            kwargs["prompt_text"] = reference_text

        wav = self._model.inference(**kwargs)
        elapsed = time.perf_counter() - start

        audio = np.array(wav).flatten().astype(np.float32)
        sr = 16000

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=len(audio) / sr,
            generation_time=elapsed,
        )


MODEL_CLASS = SparkTTS
