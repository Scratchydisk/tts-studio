# TTS Studio

A Gradio web app for testing open-source text-to-speech models and processing caption-to-speech video pipelines. Compare TTS models side by side, manage reusable voice profiles, and batch-process videos with spoken captions.

## Quick start

```bash
./run.sh
```

This creates a virtual environment, installs dependencies, and launches the web UI at `http://localhost:7860`.

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

## Supported models

| Model | Parameters | VRAM | Voice cloning | Preset voices | Sample rate |
|---|---|---|---|---|---|
| **Kokoro-82M** | 82M | ~0.5 GB | No | 25 (American/British) | 24 kHz |
| **Spark-TTS-0.5B** | 0.5B | ~2 GB | Yes | No | 16 kHz |
| **F5-TTS** | — | ~2.5 GB | Yes | No | 24 kHz |
| **Orpheus-3B** | 3B | ~7 GB | No | 8 | 24 kHz |
| **Dia-1.6B** | 1.6B | ~10 GB | No | 2 (multi-speaker) | 44.1 kHz |
| **Voxtral-4B** | 4B | ~16 GB | Yes (local only) | 20 (9 languages) | 24 kHz |

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

**Requires:** `ffmpeg` on your system PATH.

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

### Voxtral remote server setup

```bash
pip install "vllm>=0.18.0" git+https://github.com/vllm-project/vllm-omni.git
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
```

**Note:** Voice cloning via reference audio is not yet supported through the vLLM HTTP API for Voxtral. Use the local variant (`voxtral-4b-local`) for voice cloning.

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
