"""Shared UI helpers used across tabs."""

from urllib.parse import urlparse

from tts_tests import registry


def get_model_status_label(model_id: str) -> str:
    """Return a human-readable status label for a model."""
    installed = registry.is_installed(model_id)
    endpoint = registry.get_endpoint(model_id)

    if installed and endpoint:
        host = _endpoint_host(endpoint)
        return f"server: {host}"
    elif endpoint:
        host = _endpoint_host(endpoint)
        return f"server: {host}"
    elif installed:
        return "installed"
    else:
        return "not available"


def _endpoint_host(endpoint: dict) -> str:
    """Extract a short hostname from an endpoint config."""
    url = endpoint.get("url", "")
    try:
        parsed = urlparse(url)
        return parsed.hostname or "remote"
    except Exception:
        return "remote"


def get_icon_legend() -> str:
    """Return a Markdown string explaining all status icons used in the UI."""
    return (
        "\U0001f5e3 preset voices · "
        "\U0001f3a4 voice cloning · "
        "\U0001f310 remote server · "
        "\u26a0 not installed"
    )


def format_load_error(error: Exception) -> str:
    """Format a model loading error with helpful context."""
    msg = str(error)
    if "CUDA out of memory" in msg:
        return (
            f"Failed to load model: CUDA out of memory. "
            f"A model server may be using the GPU — check the Models tab "
            f"for running servers and stop any you don't need, then retry."
        )
    if "Cannot reach remote server" in msg:
        return f"Failed to load model: {msg}"
    return f"Failed to load model: {msg}"


def get_model_choices(include_unavailable: bool = True) -> list[tuple[str, str]]:
    """Return (display_label, model_id) for model dropdowns.

    Icons:
        \U0001f5e3 (speaking head) = has preset voices
        \U0001f3a4 (microphone) = supports voice cloning
        \U0001f310 (globe) = runs via remote server/API
        \u26a0 (warning) = not available

    Installed models with no remote endpoint get no extra suffix.
    """
    choices = []
    for info, available in registry.list_models():
        if not include_unavailable and not available:
            continue
        icons = ""
        if info.available_voices:
            icons += "\U0001f5e3 "
        if info.supports_voice_cloning:
            icons += "\U0001f3a4 "
        label = icons + info.name

        installed = registry.is_installed(info.model_id)
        endpoint = registry.get_endpoint(info.model_id)

        if not available:
            label += " \u26a0"
        elif endpoint:
            label += " \U0001f310"
        # else: installed locally, no endpoint — no suffix

        choices.append((label, info.model_id))
    return choices
