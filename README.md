# TTS Studio

A Gradio web app for testing open-source text-to-speech models and processing caption-to-speech video pipelines. Compare TTS models side by side, manage reusable voice profiles, and batch-process videos with spoken captions.

**Requires Python 3.11–3.12** for the full `.[all]` install (on 3.10, F5-TTS and Dia have conflicting numpy requirements). You don't need to install Python yourself — `run.sh` and `run.ps1` provision Python 3.12 automatically via [uv](https://docs.astral.sh/uv/), which is the only prerequisite. Several model dependencies do not yet support Python 3.13.

## Demos

- [Playground walkthrough](docs/playground-demo.md) — generating speech with Qwen3-TTS, narrated by the app itself

## Quick start

First install [uv](https://docs.astral.sh/uv/getting-started/installation/) (no sudo or existing Python needed):

```bash
# Linux / macOS / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

Then launch:

```bash
./run.sh        # Linux / WSL
.\run.ps1       # Windows PowerShell
```

This creates a virtual environment (Python 3.12, provisioned via uv), installs dependencies, and launches the web UI at `http://localhost:7860`. The server binds to `0.0.0.0` so it's accessible from other machines on the network at `http://<your-ip>:7860`.

The scripts detect your GPU and install the matching PyTorch build automatically: NVIDIA Blackwell cards (RTX 50-series, `sm_120`) get the CUDA 12.8 wheel, machines with no NVIDIA GPU get the CPU-only wheel, and everything else uses the default. If you install manually with `uv pip install -e ".[all]"` on a Blackwell card, you'll need to re-pin PyTorch afterwards (`uv pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade torch torchaudio`), since the model extras otherwise pull a build without `sm_120` kernels.

### CLI

```bash
# Launch the web UI
tts-studio serve

# Generate speech from the command line
tts-studio generate "Hello, world." --model kokoro-82m --voice bf_emma -o hello.wav

# Process a captioned video
tts-studio caption video.mp4 --model kokoro-82m --voice bf_emma

# Batch-process a directory of videos
tts-studio batch ./videos/ --profile emma
```

## MCP server

TTS Studio includes an [MCP server](docs/mcp-server.md) that lets AI assistants like Claude caption videos using your voice profiles. Start it with `tts-studio mcp` and connect from Claude Desktop or Claude Code. See the [MCP server docs](docs/mcp-server.md) for setup instructions.

## Supported models

Quality tiers: **S** = best-in-class, **A** = excellent, **B** = good, **C** = decent/niche.

| Model | Tier | Params | VRAM | Clone | Voices | Install |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-1.7B** | S | 1.7B | ~10 GB | No | 9 | `pip install -e ".[qwen3tts]"` |
| **Qwen3-TTS-1.7B Clone** | S | 1.7B | ~10 GB | Yes | — | `pip install -e ".[qwen3tts]"` |
| **Chatterbox** | A | 350M | ~4 GB | Yes | — | `pip install -e ".[chatterbox]"` |
| **Zonos-v0.1** | A | 1.6B | ~6 GB | Yes (10-30s) | — | `pip install -e ".[zonos]"` |
| **Sesame CSM-1B** | A | 1.1B | ~4.5 GB | Yes | — | `pip install -e ".[sesame-csm]"` |
| **TADA-1B** | A | 1B | ~5 GB | Yes | — | `pip install -e ".[tada]"` |
| **Kokoro-82M** | A | 82M | ~0.5 GB | No | 24 | included in `.[all]` |
| **F5-TTS** | A | — | ~2.5 GB | Yes | — | included in `.[all]` |
| **Voxtral-4B** | A | 4B | ~16 GB | Yes* | 20 (9 langs) | remote via vLLM |
| **Dia-1.6B** | B | 1.6B | ~10 GB | No | 2 | included in `.[all]` |
| **Orpheus-3B** | B | 3B | ~7 GB | No | 8 | remote via vLLM (see below) |
| **Spark-TTS-0.5B** | B | 0.5B | ~2 GB | Yes | — | included in `.[all]` |
| **OuteTTS-0.3-500M** | B | 500M | ~2 GB | Yes | — | included in `.[all]` |

\* Voxtral voice cloning only works with the local variant, not via the remote API.

Qwen3-TTS ships as two variants sharing the same `.[qwen3tts]` dependency: the **CustomVoice** variant has 9 preset voices but no cloning, while the **Base (Clone)** variant supports voice cloning from reference audio but has no presets.

Models in the `all` group are installed by default with `./run.sh`. Others need their extras group installed separately. Models that aren't installed show as "(not installed)" in the UI.

## Voice profiles

Profiles save a TTS configuration (model, voice, reference audio) under a friendly name for reuse across the UI and CLI.

Create a `profiles.json` in the project root:

```json
{
    "emma": {
        "model_id": "kokoro-82m",
        "voice": "bf_emma"
    },
    "cloned-sarah": {
        "model_id": "f5-tts",
        "reference_audio": "reference_audio/sarah.wav",
        "reference_text": "This is Sarah speaking naturally."
    }
}
```

Use a profile from the CLI:

```bash
tts-studio generate "Good morning." --profile emma -o morning.wav
```

## Caption-to-speech

Convert captioned videos into narrated versions. Provide a video file and a matching `.srt` subtitle file; the pipeline parses the captions, generates speech for each segment, and renders the output.

### Single video

```bash
tts-studio caption video.mp4 --srt video.srt --model kokoro-82m --voice bf_emma
```

### Batch processing

Place video files alongside matching `.srt` files (same filename stem) in a directory:

```bash
tts-studio batch ./videos/ --profile emma --pattern "*.webm"
```

### Caption configuration

Create a `caption_config.yaml` to customise TTS and output settings:

```yaml
tts:
  model_id: kokoro-82m
  voice: bf_emma
  speed: 1.0
  silence_before: 0.3
  silence_after: 0.5
output:
  format: mkv
```

**Requires:** `ffmpeg` installed on the server running TTS Studio.

## Remote models

Any model can be offloaded to a remote server running an OpenAI-compatible TTS API (e.g. vLLM). Create an `endpoints.json` file in the project root:

```json
{
    "voxtral-4b": {
        "url": "http://your-server:8000/v1",
        "model": "mistralai/Voxtral-4B-TTS-2603"
    }
}
```

Models with a configured endpoint show as "(remote)" in the UI. Models without local dependencies or a remote endpoint show as "(not installed)".

### Setting up a vLLM remote server

vLLM with the [vllm-omni](https://github.com/vllm-project/vllm-omni) extension can serve TTS models with an OpenAI-compatible API. Currently supported TTS models: **Voxtral-4B**, **Qwen3-TTS**, and **Fish Speech S2 Pro**.

#### 1. Install vLLM and vllm-omni

```bash
# Create a dedicated venv on the server
python3 -m venv ~/vllm-env && source ~/vllm-env/bin/activate

# Install vllm first, then vllm-omni (order matters — vllm-omni registers
# as a plugin and must be installed after vllm)
pip install vllm
pip install git+https://github.com/vllm-project/vllm-omni.git
```

If `--omni` is not recognised after installation, uninstall both and reinstall in order:

```bash
pip uninstall vllm vllm-omni -y
pip install vllm
pip install git+https://github.com/vllm-project/vllm-omni.git
```

#### 2. Download and serve a model

The first run downloads the model weights from HuggingFace automatically.

**Voxtral-4B** (requires >= 16 GB VRAM):

```bash
vllm serve mistralai/Voxtral-4B-TTS-2603 \
  --omni \
  --trust-remote-code \
  --enforce-eager
```

**Qwen3-TTS** (requires >= 8 GB VRAM):

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni
```

The server listens on port 8000 by default. Add `--port 8001` to change it.

#### 3. Multi-GPU servers

If you have multiple GPUs, force a specific one:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID vllm serve ...
```

vLLM runs one model per server process. To serve multiple models, start each on a different port.

#### 4. Configure TTS Studio to use the remote model

On the machine running TTS Studio, create `endpoints.json`:

```json
{
    "voxtral-4b": {
        "url": "http://your-server:8000/v1",
        "model": "mistralai/Voxtral-4B-TTS-2603"
    }
}
```

The `model` field must match the model name the server was started with.

#### 5. Test the connection

```bash
# Check the server is up
curl http://your-server:8000/v1/models

# Generate a test clip
curl -X POST http://your-server:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, this is a test.", "model": "mistralai/Voxtral-4B-TTS-2603", "voice": "neutral_female", "response_format": "wav"}' \
  --output test.wav
```

### Dia-1.6B remote server

Dia is not supported by vLLM. Use [Dia-TTS-Server](https://github.com/devnen/Dia-TTS-Server) instead:

```bash
git clone https://github.com/devnen/Dia-TTS-Server.git
cd Dia-TTS-Server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Dia requires >= 10 GB VRAM (float32 only — produces garbage at half precision). The server listens on port 8003 with an OpenAI-compatible API.

```json
{
    "dia-1.6b": {
        "url": "http://your-server:8003/v1",
        "model": "tts-1"
    }
}
```

### HuggingFace token

Some models (e.g. Orpheus) are gated on HuggingFace and require authentication to download. You can set your HuggingFace token in the **Models** tab under "HuggingFace token", or from the command line:

```bash
huggingface-cli login
```

The token is stored at `~/.cache/huggingface/token` and is shared across all virtual environments for the same user.

### Orpheus-3B remote server

Orpheus is a gated model — you must accept the license at [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) and set a HuggingFace token (see above) before first use.

Orpheus uses vLLM internally for inference and a SNAC decoder for audio. It is deliberately **not** part of the `.[all]` install: vLLM pins exact torch and recent transformers versions that conflict with other bundled models, so Orpheus always runs as a remote worker from its own venv. TTS Studio includes a bundled worker (`workers/orpheus_worker.py`) that can be started from the **Models** tab, or run manually:

```bash
# Install orpheus-speech into the vLLM venv
~/vllm-env/bin/pip install orpheus-speech

# Start the worker
~/vllm-env/bin/python workers/orpheus_worker.py --port 8001 --gpu 0
```

Orpheus requires ~7 GB VRAM. The worker exposes an OpenAI-compatible `/v1/audio/speech` endpoint.

```json
{
    "orpheus-3b": {
        "url": "http://your-server:8001/v1",
        "model": "canopylabs/orpheus-3b-0.1-ft"
    }
}
```

### Limitations

- **Voice cloning via reference audio** is not yet supported through vLLM's HTTP API for Voxtral. Use the Voxtral worker (start it from the Models tab) for voice cloning support.
- vLLM serves **one model per process**. For multiple models, run separate processes on different ports.

See [docs/remote-models.md](docs/remote-models.md) for more detailed setup notes.

## External integration

### Python API

```python
from tts_tests.api import caption_video, generate_tts

# Generate speech
audio, sr = generate_tts("Hello from Python.", model_id="kokoro-82m", voice="bf_emma")

# Caption a video using a saved profile
result = caption_video("video.mp4", profile="emma")
print(result.output_path)
```

### CLI for external processes

```bash
# Generate speech
tts-studio generate "Hello." --profile emma -o hello.wav

# Caption a video (called from another script)
tts-studio caption /path/to/video.mp4 --profile emma --format mkv
```

### Gradio API

When the web UI is running, the Gradio API is available at `http://localhost:7860/api`. Use the Gradio Python client or any HTTP client to call endpoints programmatically.

## Adding a new model

Create a new file in `tts_tests/models/` that exports:

- `MODEL_CLASS` — a subclass of `TTSModel` implementing `info()`, `load()`, `unload()`, `is_loaded()`, and `generate()`
- `is_available()` — a function returning `True` if the model's dependencies are installed

The model is auto-discovered on startup. See any existing model file for reference.

## Project structure

```
tts_tests/
  app.py              # Web UI entry point
  cli.py              # CLI entry point and argument parsing
  api.py              # Python API helpers
  base.py             # TTSModel abstract base class, TTSResult, ModelInfo
  config.py           # Paths, device detection, endpoint loading
  profiles.py         # Voice profile management (profiles.json)
  registry.py         # Model discovery, loading/unloading, remote wrapping
  remote.py           # RemoteTTSModel HTTP wrapper
  audio_utils.py      # Audio processing utilities
  models/             # One file per TTS model
  caption/            # Caption-to-speech pipeline
    config.py         # Caption/TTS configuration (YAML)
    srt_parser.py     # SRT subtitle parsing
    tts_bridge.py     # Bridge between caption segments and TTS registry
    timeline.py       # Timeline construction from segments
    render.py         # ffmpeg rendering
    pipeline.py       # Pipeline orchestration
    batch.py          # Batch directory processing
  ui/                 # Gradio interface (single generation, comparison)
endpoints.json        # Remote endpoint config (create from endpoints.json.example)
profiles.json         # Voice profiles (user-created)
run.sh                # Quick-start script
```

## Author's observations

Informal notes from testing these models. Your mileage may vary depending on hardware, reference audio quality, and text content.

- **Chatterbox** produces voice cloning quality roughly on par with Qwen3-TTS Clone, despite being a much smaller model (350M vs 1.7B). It's noticeably faster too. The trade-off is that Chatterbox has no preset voices — you always need a reference audio clip.
- **Qwen3-TTS** is the best all-rounder for preset voices. The Eric and Vivian presets are particularly natural. The Clone variant is solid for voice cloning but slower than Chatterbox at inference.
- **Kokoro** is excellent for its size — 82M parameters with surprisingly good quality and very fast inference. Best choice when VRAM is tight or you need high throughput.
- **Voxtral** has the widest language coverage (9 languages, 20 voices) but requires a beefy GPU (~16 GB) and runs best via vLLM on a dedicated server.
