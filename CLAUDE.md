# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TTS Studio is a Gradio web app + CLI for testing open-source text-to-speech models and processing caption-to-speech video pipelines. It supports 14+ TTS models with a unified interface, voice profiles, side-by-side comparison, and batch video narration.

## Commands

```bash
# Quick start (creates venv, installs deps, launches UI)
./run.sh

# Install in development mode (use uv — pip backtracks badly on this dep tree)
uv pip install -e ".[all]"

# Install a specific model's extras
uv pip install -e ".[chatterbox]"

# Launch web UI (default command)
tts-studio serve

# CLI generation
tts-studio generate "Hello" --model kokoro-82m --voice bf_emma -o hello.wav

# Caption a video
tts-studio caption video.mp4 --srt video.srt --profile emma

# Batch process
tts-studio batch ./videos/ --profile emma
```

There are no tests or linting configured yet.

## Architecture

### Entry points
- **CLI**: `tts_tests/cli.py` — argparse with subcommands (serve, generate, caption, batch, profiles). Entry point: `tts_tests.cli:main`
- **Web UI**: `tts_tests/app.py` → `tts_tests/ui/main.py` — Gradio app with 5 tabs
- **Python API**: `tts_tests/api.py` — `generate_tts()`, `caption_video()`, `caption_batch()`

### Model system
- **Base class**: `tts_tests/base.py` — `TTSModel` ABC with `info()`, `load()`, `unload()`, `generate()` → `TTSResult`
- **Registry**: `tts_tests/registry.py` — auto-discovers models from `tts_tests/models/`, each file exports `MODEL_CLASS` and `is_available()`. Only one model loaded at a time (auto-unloads previous, clears CUDA cache).
- **Remote wrapping**: `tts_tests/remote.py` — `RemoteTTSModel` wraps any model with HTTP calls to OpenAI-compatible `/v1/audio/speech` endpoints. Configured via `endpoints.json` in project root.
- **Model files**: `tts_tests/models/{model_id}.py` — one file per model

### Caption-to-speech pipeline (`tts_tests/caption/`)
4-stage resumable pipeline: **parse** (SRT → segments) → **tts** (segments → WAV files) → **timeline** (timing/freeze calculation) → **render** (ffmpeg mux). Intermediates saved to `{video}_intermediates/` for resumption via `--from-stage`.

### Voice profiles
`tts_tests/profiles.py` manages `profiles.json` — saves model + voice + reference audio configs under friendly names for reuse across CLI and UI.

### Key data types
- `TTSResult`: audio (float32 ndarray), sample_rate, duration, generation_time
- `ModelInfo`: name, model_id, supports_voice_cloning, available_voices, estimated_vram_gb, native_sample_rate
- `VoiceProfile`: model_id, voice, reference_audio, reference_text

### UI tabs (`tts_tests/ui/`)
Playground (single generation), Compare (side-by-side), Voice Profiles (CRUD), Caption Video, Batch Caption. Cross-tab state wiring for "Send to Profile" flow.

## Adding a new model

Create `tts_tests/models/{model_id}.py` exporting `MODEL_CLASS` (TTSModel subclass) and `is_available()` (returns True if deps installed). Auto-discovered on startup. Add optional deps group to `pyproject.toml`.

## External dependencies

- **ffmpeg** on PATH required for caption pipeline (video rendering, ffprobe for FPS/duration)
- **CUDA** optional but expected for local models
- Python >= 3.11 for the `all` extra (run.sh/run.ps1 provision 3.12 via uv); base package works on 3.10
- PyTorch >= 2.0
- Orpheus is excluded from `all` (vllm dependency conflicts) — runs as a remote worker via `endpoints.json`

## Configuration files

- `endpoints.json` — remote model endpoints (OpenAI-compatible API URLs)
- `profiles.json` — saved voice profiles
- `caption_config.yaml` — caption pipeline TTS/output settings
