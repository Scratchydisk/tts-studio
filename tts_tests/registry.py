"""Model registry — discovers available models and manages loading/unloading."""

import gc
import importlib
import logging
import pkgutil
from typing import Type

import torch

from tts_tests.base import ModelInfo, TTSModel

logger = logging.getLogger(__name__)

# Maps model_id -> (class, is_available, endpoint_config | None)
_registry: dict[str, tuple[Type[TTSModel], bool, dict | None]] = {}
_loaded_model: TTSModel | None = None


def register(cls: Type[TTSModel], available: bool = True, endpoint: dict | None = None) -> None:
    """Register a model class."""
    instance = cls()
    model_id = instance.info().model_id
    _registry[model_id] = (cls, available, endpoint)


def discover() -> None:
    """Auto-discover model modules in tts_tests.models."""
    _registry.clear()
    import tts_tests.models as models_pkg
    from tts_tests.config import load_endpoints

    endpoints = load_endpoints()

    for _importer, modname, _ispkg in pkgutil.iter_modules(models_pkg.__path__):
        try:
            mod = importlib.import_module(f"tts_tests.models.{modname}")
            if hasattr(mod, "MODEL_CLASS") and hasattr(mod, "is_available"):
                cls = mod.MODEL_CLASS
                instance = cls()
                model_id = instance.info().model_id
                endpoint = endpoints.get(model_id)

                # A model is available if it has a remote endpoint OR local deps
                local_available = mod.is_available()
                available = local_available or endpoint is not None

                register(cls, available, endpoint)
        except Exception as e:
            logger.warning("Failed to discover model %s: %s", modname, e)


def list_models() -> list[tuple[ModelInfo, bool]]:
    """Return (ModelInfo, is_available) for all registered models."""
    results = []
    for cls, available, _endpoint in _registry.values():
        results.append((cls().info(), available))
    return results


def is_remote(model_id: str) -> bool:
    """Check if a model is configured to use a remote endpoint."""
    if model_id in _registry:
        return _registry[model_id][2] is not None
    return False


def get_endpoint(model_id: str) -> dict | None:
    """Get the remote endpoint config for a model, if any."""
    if model_id in _registry:
        return _registry[model_id][2]
    return None


def get_loaded() -> TTSModel | None:
    return _loaded_model


def load_model(model_id: str, device: str = "cuda") -> TTSModel:
    """Load a model, unloading any currently loaded model first."""
    global _loaded_model
    if _loaded_model is not None:
        if _loaded_model.info().model_id == model_id and _loaded_model.is_loaded():
            return _loaded_model
        _loaded_model.unload()
        del _loaded_model
        _loaded_model = None
        gc.collect()
        torch.cuda.empty_cache()

    if model_id not in _registry:
        raise ValueError(f"Unknown model: {model_id}")

    cls, available, endpoint = _registry[model_id]
    if not available:
        raise RuntimeError(
            f"Model {model_id} dependencies are not installed and no remote endpoint configured. "
            f"Check pyproject.toml for the optional dependency group or add an endpoint to endpoints.json."
        )

    model = cls()

    if endpoint is not None:
        from tts_tests.remote import RemoteTTSModel
        model = RemoteTTSModel(
            local_model=model,
            base_url=endpoint["url"],
            api_model=endpoint.get("model"),
        )

    model.load(device)
    _loaded_model = model
    return model


def unload_current() -> None:
    """Unload the currently loaded model."""
    global _loaded_model
    if _loaded_model is not None:
        _loaded_model.unload()
        del _loaded_model
        _loaded_model = None
        gc.collect()
        torch.cuda.empty_cache()
