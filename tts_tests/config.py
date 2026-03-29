"""Application configuration."""

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "tts-studio"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
REFERENCE_AUDIO_DIR = Path(__file__).parent.parent / "reference_audio"
PROFILE_AUDIO_DIR = REFERENCE_AUDIO_DIR
ENDPOINTS_FILE = Path(__file__).parent.parent / "endpoints.json"
SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

# Ensure directories exist
for d in (CACHE_DIR, OUTPUT_DIR, REFERENCE_AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _read_settings() -> dict:
    """Read settings.json."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to read settings.json: %s", e)
        return {}


def _write_settings(data: dict) -> None:
    """Write settings.json atomically."""
    import tempfile
    content = json.dumps(data, indent=4, ensure_ascii=False) + "\n"
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(SETTINGS_FILE)


def get_available_gpus() -> list[dict]:
    """Return a list of available GPUs with index and name."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            gpus.append({"index": i, "name": name})
    return gpus


def get_device(model_id: str | None = None) -> str:
    """Get the device for a model, respecting GPU settings.

    Checks (in order):
    1. Per-model GPU override in settings.json
    2. Default GPU in settings.json
    3. First available CUDA device, or CPU
    """
    if not torch.cuda.is_available():
        return "cpu"

    settings = _read_settings()
    gpu_settings = settings.get("gpu", {})

    # Per-model override
    if model_id:
        model_gpus = gpu_settings.get("models", {})
        if model_id in model_gpus:
            idx = model_gpus[model_id]
            if idx == "cpu":
                return "cpu"
            return f"cuda:{idx}"

    # Default GPU
    default = gpu_settings.get("default")
    if default is not None:
        if default == "cpu":
            return "cpu"
        return f"cuda:{default}"

    return "cuda"


def set_gpu_default(gpu_index: int | str) -> None:
    """Set the default GPU index (int) or 'cpu'."""
    settings = _read_settings()
    gpu_settings = settings.setdefault("gpu", {})
    gpu_settings["default"] = gpu_index
    _write_settings(settings)


def set_gpu_for_model(model_id: str, gpu_index: int | str | None) -> None:
    """Set a per-model GPU override. Pass None to clear the override."""
    settings = _read_settings()
    gpu_settings = settings.setdefault("gpu", {})
    model_gpus = gpu_settings.setdefault("models", {})
    if gpu_index is None:
        model_gpus.pop(model_id, None)
    else:
        model_gpus[model_id] = gpu_index
    # Clean up empty dicts
    if not model_gpus:
        gpu_settings.pop("models", None)
    _write_settings(settings)


def get_vllm_config() -> dict:
    """Get vLLM server configuration from settings.json."""
    settings = _read_settings()
    return settings.get("vllm", {})


def set_vllm_venv(venv_path: str) -> None:
    """Set the path to the vLLM virtual environment."""
    settings = _read_settings()
    vllm = settings.setdefault("vllm", {})
    vllm["venv"] = venv_path
    _write_settings(settings)


def get_vllm_servers() -> dict[str, dict]:
    """Get configured vLLM servers from settings.json."""
    settings = _read_settings()
    return settings.get("vllm", {}).get("servers", {})


def save_vllm_server(model_id: str, config: dict) -> None:
    """Save a vLLM server configuration."""
    settings = _read_settings()
    vllm = settings.setdefault("vllm", {})
    servers = vllm.setdefault("servers", {})
    servers[model_id] = config
    _write_settings(settings)


def delete_vllm_server(model_id: str) -> None:
    """Remove a vLLM server configuration."""
    settings = _read_settings()
    vllm = settings.get("vllm", {})
    servers = vllm.get("servers", {})
    servers.pop(model_id, None)
    if not servers:
        vllm.pop("servers", None)
    _write_settings(settings)


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
