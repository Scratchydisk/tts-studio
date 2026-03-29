"""Model management tab — view model status, install models, assign GPUs, manage vLLM servers."""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import httpx
import torch

from tts_tests import registry
from tts_tests.config import (
    ENDPOINTS_FILE,
    _read_settings,
    _write_settings,
    get_available_gpus,
    get_device,
    get_vllm_config,
    get_vllm_servers,
    save_vllm_server,
    set_gpu_default,
    set_gpu_for_model,
    set_vllm_venv,
)
from tts_tests.ui.shared import get_icon_legend, get_model_status_label

_install_lock = threading.Lock()
_project_root = Path(__file__).resolve().parent.parent.parent
_needs_restart = False

# Track running vLLM server processes: model_id -> Popen
_vllm_processes: dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_gpu_info() -> str:
    """Build a Markdown summary of available GPUs."""
    gpus = get_available_gpus()
    if not gpus:
        return "**No GPUs detected** — models will run on CPU."
    lines = [
        "### GPUs",
        "| GPU | Name | VRAM Total | VRAM Used | VRAM Free |",
        "|-----|------|------------|-----------|-----------|",
    ]
    for gpu in gpus:
        idx = gpu["index"]
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        try:
            free, total_bytes = torch.cuda.mem_get_info(idx)
            used = (total_bytes - free) / 1024**3
            free_gb = free / 1024**3
        except Exception:
            reserved = torch.cuda.memory_reserved(idx) / 1024**3
            used = reserved
            free_gb = total - used
        lines.append(
            f"| {idx} | {gpu['name']} | {total:.1f} GB | {used:.1f} GB | {free_gb:.1f} GB |"
        )
    return "\n".join(lines)


def _build_model_table() -> str:
    """Build a Markdown table of all models with their status."""
    lines = [
        "| Model | Status | GPU | VRAM | Voices | Cloning | Description |",
        "|-------|--------|-----|------|--------|---------|-------------|",
    ]
    for info, _available in registry.list_models():
        status = get_model_status_label(info.model_id)
        device = get_device(info.model_id)
        if device == "cpu":
            gpu_label = "CPU"
        elif device.startswith("cuda:"):
            idx = int(device.split(":")[1])
            gpus = get_available_gpus()
            gpu_label = gpus[idx]["name"] if idx < len(gpus) else device
        else:
            gpu_label = "default"
        voices = str(len(info.available_voices)) if info.available_voices else "—"
        cloning = "Yes" if info.supports_voice_cloning else "No"
        lines.append(
            f"| **{info.name}** | {status} | {gpu_label} | {info.estimated_vram_gb:.1f} GB | "
            f"{voices} | {cloning} | {info.description} |"
        )
    return "\n".join(lines)


def _get_installable_models() -> list[tuple[str, str]]:
    """Return (display_label, model_id) for models that can be pip-installed."""
    choices = []
    for info, _available in registry.list_models():
        extra = registry.get_pip_extra(info.model_id)
        if not registry.is_installed(info.model_id) and extra:
            choices.append((f"{info.name}  (pip extra: {extra})", info.model_id))
    return choices


def _gpu_choices() -> list[tuple[str, int | str]]:
    """Return dropdown choices for GPU selection."""
    choices = []
    for gpu in get_available_gpus():
        choices.append((f"GPU {gpu['index']}: {gpu['name']}", gpu["index"]))
    choices.append(("CPU", "cpu"))
    return choices


def _model_choices_for_gpu() -> list[tuple[str, str]]:
    """Return (display_label, model_id) for all locally usable models."""
    choices = []
    for info, _available in registry.list_models():
        if registry.is_installed(info.model_id):
            choices.append((info.name, info.model_id))
    return choices


def _get_current_gpu_assignment(model_id: str) -> str:
    """Return current GPU assignment description for a model."""
    if not model_id:
        return ""
    device = get_device(model_id)
    gpus = get_available_gpus()
    if device == "cpu":
        return "Currently assigned to: **CPU**"
    elif device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        name = gpus[idx]["name"] if idx < len(gpus) else device
        return f"Currently assigned to: **GPU {idx}: {name}**"
    else:
        name = gpus[0]["name"] if gpus else "GPU"
        return f"Currently assigned to: **{name}** (default)"


def _install_model(model_id: str):
    """Install a model's dependencies via pip, streaming output."""
    if not model_id:
        yield "Please select a model to install."
        return

    extra = registry.get_pip_extra(model_id)
    if not extra:
        yield f"No pip extra defined for {model_id}. Check pyproject.toml or install manually."
        return

    if not _install_lock.acquire(blocking=False):
        yield "Another installation is already in progress. Please wait."
        return

    try:
        cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extra}]"]
        yield f"$ pip install -e \".[{extra}]\"\n\n"

        process = subprocess.Popen(
            cmd,
            cwd=str(_project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        output = f"$ pip install -e \".[{extra}]\"\n\n"
        for line in process.stdout:
            output += line
            yield output

        process.wait()
        if process.returncode == 0:
            global _needs_restart
            _needs_restart = True
            registry.discover()
            output += "\n--- Installation complete. Restart the app to use this model. ---"
        else:
            output += f"\n--- Installation failed (exit code {process.returncode}). ---"
        yield output
    finally:
        _install_lock.release()


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

# Known vLLM-compatible models with sensible defaults
_VLLM_MODEL_DEFAULTS = {
    "voxtral-4b": {
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "port": 8100,
        "gpu": 0,
        "args": [],
        "worker": "voxtral",
    },
    "qwen3-tts-1.7b": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "port": 8001,
        "gpu": 0,
        "args": ["--omni", "--gpu-memory-utilization", "0.95", "--enforce-eager"],
    },
}


def _seed_servers_from_endpoints() -> None:
    """Seed vLLM server configs from endpoints.json for any not already configured."""
    import json
    from urllib.parse import urlparse

    existing = get_vllm_servers()
    if not ENDPOINTS_FILE.exists():
        endpoints = {}
    else:
        try:
            endpoints = json.loads(ENDPOINTS_FILE.read_text())
        except Exception:
            endpoints = {}

    changed = False
    # Seed from endpoints.json
    for model_id, cfg in endpoints.items():
        if model_id in existing:
            continue
        url = cfg.get("url", "")
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port or 8000
        except Exception:
            continue
        # Only seed for localhost-ish endpoints (the TTS Studio server manages them)
        if host not in ("localhost", "127.0.0.1", "0.0.0.0"):
            continue
        hf_model = cfg.get("model", "")
        if not hf_model:
            continue
        # Use known defaults if available, otherwise basic config
        defaults = _VLLM_MODEL_DEFAULTS.get(model_id, {})
        save_vllm_server(model_id, {
            "model": hf_model,
            "port": port,
            "gpu": defaults.get("gpu", 0),
            "args": defaults.get("args", ["--omni", "--trust-remote-code", "--enforce-eager"]),
        })
        changed = True

    # Also seed known models that aren't configured anywhere yet
    for model_id, defaults in _VLLM_MODEL_DEFAULTS.items():
        if model_id not in existing and model_id not in get_vllm_servers():
            save_vllm_server(model_id, defaults)
            changed = True


def _get_vllm_python() -> str | None:
    """Return the python executable for the vLLM venv, or None if not configured."""
    config = get_vllm_config()
    venv_path = config.get("venv")
    if not venv_path:
        return None
    venv = Path(venv_path).expanduser()
    python = venv / "bin" / "python"
    if not python.exists():
        return None
    return str(python)


def _check_server_status(port: int) -> str:
    """Check if a vLLM server is responding on the given port."""
    try:
        resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=3.0)
        if resp.status_code == 200:
            return "running"
    except Exception:
        pass
    # Check if we have a tracked process
    for mid, proc in list(_vllm_processes.items()):
        srv = get_vllm_servers().get(mid, {})
        if srv.get("port") == port:
            if proc.poll() is None:
                return "starting"
            else:
                # Process exited
                del _vllm_processes[mid]
                return "stopped"
    return "stopped"


def _build_server_status() -> str:
    """Build Markdown showing vLLM server status."""
    servers = get_vllm_servers()
    if not servers:
        return ""
    lines = [
        "| Model | HF Model | Port | GPU | Status |",
        "|-------|----------|------|-----|--------|",
    ]
    for model_id, cfg in servers.items():
        port = cfg.get("port", "—")
        gpu = cfg.get("gpu", "—")
        hf_model = cfg.get("model", "—")
        status = _check_server_status(port) if isinstance(port, int) else "—"
        status_badge = {"running": "running", "starting": "starting...", "stopped": "stopped"}.get(status, status)
        lines.append(f"| {model_id} | {hf_model} | {port} | {gpu} | {status_badge} |")
    return "\n".join(lines)


def _start_vllm_server(model_id: str):
    """Start a vLLM or worker server for the given model, streaming log output."""
    servers = get_vllm_servers()
    if model_id not in servers:
        yield "Server not configured."
        return

    python = _get_vllm_python()
    if not python:
        yield "vLLM venv not configured or not found. Set the venv path first."
        return

    cfg = servers[model_id]
    hf_model = cfg.get("model")
    port = cfg.get("port", 8000)
    gpu = cfg.get("gpu", 0)
    extra_args = cfg.get("args", [])
    worker_type = cfg.get("worker")

    if not hf_model:
        yield "No HuggingFace model name configured for this server."
        return

    # Check if already running
    if _check_server_status(port) == "running":
        yield f"Server already running on port {port}."
        return

    # Free VRAM by unloading any locally loaded model before starting the server
    registry.unload_current()

    # Build command based on worker type
    if worker_type == "voxtral":
        worker_script = str(_project_root / "workers" / "voxtral_worker.py")
        cmd = [python, worker_script, "--port", str(port), "--gpu", str(gpu)]
        display_cmd = f"python workers/voxtral_worker.py --port {port} --gpu {gpu}"
    else:
        # Use the vllm-omni binary from the venv (the module has no __main__
        # block so -m invocation silently exits).
        vllm_bin = str(Path(python).parent / "vllm-omni")
        cmd = [
            vllm_bin, "serve", hf_model,
            "--port", str(port),
        ] + extra_args
        display_cmd = f"vllm-omni serve {hf_model} --port {port} {' '.join(extra_args)}"

    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}

    output = f"$ CUDA_VISIBLE_DEVICES={gpu} {display_cmd}\n\n"
    yield output

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    _vllm_processes[model_id] = process

    # Also ensure endpoints.json is updated
    _sync_endpoint(model_id, port, hf_model)

    # Stream output until server is ready or fails
    for line in process.stdout:
        output += line
        yield output
        # vLLM/uvicorn prints this when ready
        if "Application startup complete" in line or "Uvicorn running on" in line:
            output += "\n--- Server is ready. ---"
            yield output
            def _drain():
                for _ in process.stdout:
                    pass
            threading.Thread(target=_drain, daemon=True).start()
            return
        # Check if process died
        if process.poll() is not None:
            output += f"\n--- Server exited with code {process.returncode}. ---"
            _vllm_processes.pop(model_id, None)
            yield output
            return


def _stop_vllm_server(model_id: str) -> str:
    """Stop a running vLLM server."""
    proc = _vllm_processes.get(model_id)
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        _vllm_processes.pop(model_id, None)
        return f"Server for **{model_id}** stopped."

    # Try to find by port and kill
    servers = get_vllm_servers()
    cfg = servers.get(model_id, {})
    port = cfg.get("port")
    if port:
        try:
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True, text=True,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
            if pids:
                return f"Server on port {port} stopped (PID {', '.join(pids)})."
        except Exception:
            pass

    return f"No running server found for **{model_id}**."


def _sync_endpoint(model_id: str, port: int, hf_model: str) -> None:
    """Update endpoints.json with the vLLM server URL."""
    import json
    endpoints = {}
    if ENDPOINTS_FILE.exists():
        try:
            endpoints = json.loads(ENDPOINTS_FILE.read_text())
        except Exception:
            pass

    endpoints[model_id] = {
        "url": f"http://localhost:{port}/v1",
        "model": hf_model,
    }
    ENDPOINTS_FILE.write_text(
        json.dumps(endpoints, indent=4, ensure_ascii=False) + "\n",
    )
    # Re-discover so the model shows as available
    registry.discover()


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_model_management_tab():
    """Build the model management tab."""

    gr.Markdown(
        "Models run either on **the server hosting TTS Studio** (requires installing their Python "
        "dependencies) or on a **remote GPU server** via API "
        "(configured in `endpoints.json`). Some models support both."
    )
    gr.Markdown(get_icon_legend())

    # --- GPU info ---
    gpus = get_available_gpus()
    if gpus:
        gr.Markdown(value=_build_gpu_info)

    model_table = gr.Markdown(value=_build_model_table)

    # --- Install a model (most common action, so it's first) ---
    gr.Markdown("### Install a model")
    gr.Markdown(
        "Select a model below to install its dependencies into the current "
        "environment. Models from the **all** group (Kokoro, F5-TTS, Dia, "
        "Orpheus, Spark, OuteTTS) are included by default with `./run.sh`."
    )

    installable = _get_installable_models()
    with gr.Row():
        install_dropdown = gr.Dropdown(
            label="Model to install",
            choices=installable,
            value=None,
            interactive=True,
        )
        install_btn = gr.Button("Install", variant="primary")

    install_log = gr.Code(label="Install log", language="shell", lines=15, max_lines=15)

    restart_btn = gr.Button(
        "Restart App",
        variant="secondary",
        visible=False,
    )

    def _install_and_refresh(model_id):
        for output in _install_model(model_id):
            yield (
                output,
                gr.update(choices=_get_installable_models(), value=None),
                _build_model_table(),
                gr.update(visible=_needs_restart),
            )

    install_btn.click(
        fn=_install_and_refresh,
        inputs=[install_dropdown],
        outputs=[install_log, install_dropdown, model_table, restart_btn],
    )

    def _restart():
        os.execv(sys.executable, [sys.executable] + sys.argv)

    restart_btn.click(fn=_restart)

    # --- GPU assignment ---
    if len(gpus) > 1:
        gr.Markdown("### GPU assignment")
        gr.Markdown(
            "Assign models to specific GPUs. This lets you run lighter models "
            "on one GPU while keeping another free for other work."
        )

        gpu_opts = _gpu_choices()
        settings = _read_settings()
        current_default = settings.get("gpu", {}).get("default")

        with gr.Row():
            default_gpu_dd = gr.Dropdown(
                label="Default GPU",
                choices=gpu_opts,
                value=current_default if current_default is not None else gpu_opts[0][1],
                interactive=True,
            )

        def _save_default_gpu(gpu_index):
            set_gpu_default(gpu_index)
            return _build_model_table()

        default_gpu_dd.change(
            fn=_save_default_gpu,
            inputs=[default_gpu_dd],
            outputs=[model_table],
        )

        gr.Markdown("#### Per-model override")
        with gr.Row():
            model_gpu_dd = gr.Dropdown(
                label="Model",
                choices=_model_choices_for_gpu(),
                value=None,
                interactive=True,
            )
            model_gpu_select = gr.Dropdown(
                label="GPU",
                choices=[("Use default", "default")] + gpu_opts,
                value="default",
                interactive=True,
            )
            model_gpu_save = gr.Button("Save", variant="primary")

        model_gpu_status = gr.Markdown()

        model_gpu_dd.change(
            fn=_get_current_gpu_assignment,
            inputs=[model_gpu_dd],
            outputs=[model_gpu_status],
        )

        def _save_model_gpu(model_id, gpu_index):
            if not model_id:
                return "Please select a model.", _build_model_table()
            if gpu_index == "default":
                set_gpu_for_model(model_id, None)
                return f"Cleared GPU override for **{model_id}**.", _build_model_table()
            else:
                set_gpu_for_model(model_id, gpu_index)
                gpus_list = get_available_gpus()
                if gpu_index == "cpu":
                    label = "CPU"
                else:
                    label = next(
                        (f"GPU {g['index']}: {g['name']}" for g in gpus_list if g["index"] == gpu_index),
                        str(gpu_index),
                    )
                return f"**{model_id}** assigned to **{label}**.", _build_model_table()

        model_gpu_save.click(
            fn=_save_model_gpu,
            inputs=[model_gpu_dd, model_gpu_select],
            outputs=[model_gpu_status, model_table],
        )

    # --- vLLM server management ---
    gr.Markdown("### vLLM servers")
    gr.Markdown(
        "Serve TTS models via [vLLM](https://github.com/vllm-project/vllm) with the "
        "[vllm-omni](https://github.com/vllm-project/vllm-omni) extension. "
        "vLLM uses a **separate virtual environment** because it pins specific "
        "PyTorch versions.\n\n"
        "**One-time setup** (run in a terminal on the machine hosting TTS Studio):\n"
        "```bash\n"
        "python3 -m venv ~/vllm-env\n"
        "~/vllm-env/bin/pip install vllm\n"
        "~/vllm-env/bin/pip install git+https://github.com/vllm-project/vllm-omni.git\n"
        "~/vllm-env/bin/pip install mistral-common soundfile uvicorn fastapi python-multipart\n"
        "```\n"
        "The last line installs dependencies needed for the Voxtral worker "
        "(voice cloning support). Then set the venv path below."
    )

    # Seed vLLM server configs from endpoints.json on first run
    _seed_servers_from_endpoints()

    vllm_config = get_vllm_config()
    current_venv = vllm_config.get("venv", "")

    with gr.Row():
        vllm_venv_input = gr.Textbox(
            label="vLLM venv path",
            value=current_venv,
            placeholder="~/vllm-env",
            info="Path to a separate Python venv with vLLM installed.",
        )
        vllm_venv_save = gr.Button("Save", variant="secondary")

    vllm_venv_status = gr.Markdown()

    def _save_vllm_venv(path):
        if not path or not path.strip():
            return "Please enter a path."
        expanded = Path(path.strip()).expanduser()
        python = expanded / "bin" / "python"
        if not python.exists():
            return f"Warning: `{python}` not found. Make sure the venv exists."
        set_vllm_venv(path.strip())
        return f"vLLM venv set to `{path.strip()}`."

    vllm_venv_save.click(
        fn=_save_vllm_venv,
        inputs=[vllm_venv_input],
        outputs=[vllm_venv_status],
    )

    # Server status table
    server_status = gr.Markdown(value=_build_server_status)

    # Add/configure server
    gr.Markdown("#### Configure a server")
    with gr.Row():
        srv_model_id = gr.Textbox(
            label="Model ID (for endpoints.json)",
            placeholder="e.g. voxtral-4b",
        )
        srv_hf_model = gr.Textbox(
            label="HuggingFace model name",
            placeholder="e.g. mistralai/Voxtral-4B-TTS-2603",
        )
    with gr.Row():
        srv_port = gr.Number(label="Port", value=8000, precision=0, info="Port for the vLLM API server.")
        srv_gpu = gr.Number(label="GPU index", value=0, precision=0, info="Which GPU to run this server on.")
        srv_extra_args = gr.Textbox(
            label="Extra args",
            placeholder="--omni --trust-remote-code --enforce-eager",
            value="--omni --trust-remote-code --enforce-eager",
            info="Additional command-line arguments for vLLM.",
        )
    srv_save_btn = gr.Button("Save Server Config", variant="primary")
    srv_config_status = gr.Markdown()

    # Start / stop controls
    gr.Markdown("#### Start / stop")

    def _server_choices():
        servers = get_vllm_servers()
        return [(f"{mid} ({cfg.get('model', '?')})", mid) for mid, cfg in servers.items()]

    with gr.Row():
        srv_select = gr.Dropdown(
            label="Server",
            choices=_server_choices(),
            value=None,
            interactive=True,
        )
        srv_start_btn = gr.Button("Start", variant="primary")
        srv_stop_btn = gr.Button("Stop", variant="stop")
        srv_refresh_btn = gr.Button("Refresh Status", variant="secondary")

    # Wire save button (after srv_select is defined)
    def _save_server_config(model_id, hf_model, port, gpu, extra_args):
        if not model_id or not model_id.strip():
            return "Please enter a model ID.", _build_server_status(), gr.update()
        if not hf_model or not hf_model.strip():
            return "Please enter a HuggingFace model name.", _build_server_status(), gr.update()
        args = extra_args.split() if extra_args and extra_args.strip() else []
        save_vllm_server(model_id.strip(), {
            "model": hf_model.strip(),
            "port": int(port),
            "gpu": int(gpu),
            "args": args,
        })
        choices = _server_choices()
        return (
            f"Server config saved for **{model_id.strip()}**.",
            _build_server_status(),
            gr.update(choices=choices, value=model_id.strip()),
        )

    srv_save_btn.click(
        fn=_save_server_config,
        inputs=[srv_model_id, srv_hf_model, srv_port, srv_gpu, srv_extra_args],
        outputs=[srv_config_status, server_status, srv_select],
    )

    srv_log = gr.Code(label="Server log", language="shell", lines=15, max_lines=15)

    def _start_and_update(model_id):
        for output in _start_vllm_server(model_id):
            yield output, _build_server_status(), _build_model_table()

    srv_start_btn.click(
        fn=_start_and_update,
        inputs=[srv_select],
        outputs=[srv_log, server_status, model_table],
    )

    def _stop_and_update(model_id):
        msg = _stop_vllm_server(model_id)
        return msg, _build_server_status()

    srv_stop_btn.click(
        fn=_stop_and_update,
        inputs=[srv_select],
        outputs=[srv_log, server_status],
    )

    def _refresh():
        return _build_server_status(), gr.update(choices=_server_choices())

    srv_refresh_btn.click(
        fn=_refresh,
        outputs=[server_status, srv_select],
    )

    return model_table
