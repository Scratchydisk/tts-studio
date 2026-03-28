"""Application configuration."""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "tts-studio"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
REFERENCE_AUDIO_DIR = Path(__file__).parent.parent / "reference_audio"
PROFILE_AUDIO_DIR = CACHE_DIR / "reference_audio"
ENDPOINTS_FILE = Path(__file__).parent.parent / "endpoints.json"

# Ensure directories exist
for d in (CACHE_DIR, OUTPUT_DIR, REFERENCE_AUDIO_DIR, PROFILE_AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_endpoints() -> dict[str, dict]:
    """Load remote endpoint config from endpoints.json.

    Returns a dict mapping model_id -> {"url": "...", "model": "..."}.
    The "model" field is optional and overrides the model name sent to the API.

    Example endpoints.json:
        {
            "voxtral-4b": {
                "url": "http://my-server:8000/v1",
                "model": "mistralai/Voxtral-4B-TTS-2603"
            },
            "orpheus-3b": {
                "url": "http://my-server:8001/v1"
            }
        }
    """
    if not ENDPOINTS_FILE.exists():
        return {}
    try:
        data = json.loads(ENDPOINTS_FILE.read_text())
        if not isinstance(data, dict):
            logger.warning("endpoints.json should be a JSON object, ignoring")
            return {}
        return data
    except Exception as e:
        logger.warning("Failed to read endpoints.json: %s", e)
        return {}
