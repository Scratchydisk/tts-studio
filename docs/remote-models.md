# Running TTS Models on a Remote Server

Some models need more VRAM than your local GPU provides. This guide covers serving TTS models on a remote machine and connecting them to TTS Studio.

## How it works

TTS Studio can connect to any server that implements the OpenAI-compatible `/v1/audio/speech` endpoint. You configure the connection in `endpoints.json`:

```json
{
    "model-id": {
        "url": "http://your-server:8000/v1",
        "model": "org/model-name"
    }
}
```

The `model-id` must match the model's ID in TTS Studio (e.g. `voxtral-4b`, `dia-1.6b`). The model then appears as "(remote)" in the UI.

---

## Voxtral-4B via vLLM

**Requirements:** GPU with >= 16GB VRAM, CUDA

[Voxtral](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) is served via [vLLM-omni](https://github.com/vllm-project/vllm-omni), an extension of vLLM for multi-modal models.

### Server setup

```bash
# Create a venv
python3 -m venv ~/vllm-voxtral && source ~/vllm-voxtral/bin/activate

# Install vllm first, then vllm-omni (order matters)
pip install vllm
pip install git+https://github.com/vllm-project/vllm-omni.git

# Start the server
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --trust-remote-code --enforce-eager
```

The server listens on port 8000 by default. Add `--port 8001` to change it.

**Troubleshooting:**
- If `--omni` is not recognised, vllm-omni wasn't installed after vllm. Uninstall both and reinstall in order.
- If you have multiple GPUs, set `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID` before the command.

### Client config

Add to `endpoints.json` on the machine running TTS Studio:

```json
{
    "voxtral-4b": {
        "url": "http://your-server:8000/v1",
        "model": "mistralai/Voxtral-4B-TTS-2603"
    }
}
```

### Available voices

Voxtral has 20 preset voices across 9 languages: `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male`, `fr_female`, `fr_male`, `es_female`, `es_male`, `de_female`, `de_male`, `it_female`, `it_male`, `pt_female`, `pt_male`, `nl_female`, `nl_male`, `ar_male`, `hi_female`, `hi_male`.

### Limitations

Voice cloning via reference audio is **not yet supported** through the vLLM HTTP API. The `/v1/audio/voices` upload endpoint exists but the tokeniser rejects custom voice names. Use the local variant (`voxtral-4b-local`) for voice cloning if your GPU has enough VRAM.

---

## Dia-1.6B via Dia-TTS-Server

**Requirements:** GPU with >= 10GB VRAM (float32 required — produces garbage at float16/bfloat16)

[Dia](https://github.com/nari-labs/dia) is not supported by vLLM. Instead, use [Dia-TTS-Server](https://github.com/devnen/Dia-TTS-Server), a FastAPI server with an OpenAI-compatible endpoint.

### Server setup

```bash
git clone https://github.com/devnen/Dia-TTS-Server.git
cd Dia-TTS-Server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py
```

The server starts on port 8003 by default and exposes:
- `/v1/audio/speech` — OpenAI-compatible TTS endpoint
- `/docs` — interactive Swagger API documentation
- `/health` — health check
- Web UI at `http://your-server:8003`

**Docker alternative:**
```bash
docker compose up -d
```

### Client config

Add to `endpoints.json`:

```json
{
    "dia-1.6b": {
        "url": "http://your-server:8003/v1",
        "model": "tts-1"
    }
}
```

Note: Dia-TTS-Server uses `"model": "tts-1"` in its OpenAI-compatible API.

### Features

- 43 built-in voice presets
- Voice cloning via reference audio
- Multi-speaker dialogue with `[S1]`/`[S2]` tags
- Emotion tags: `(laughs)`, `(sighs)`, `(coughs)`, etc.
- Text chunking for long inputs

---

## Other models supported by vLLM-omni

[vLLM-omni](https://github.com/vllm-project/vllm-omni) also supports:

- **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-*`) — voice cloning, voice design, 24 kHz
- **Fish Speech S2 Pro** (`fishaudio/s2-pro`) — voice cloning, 44.1 kHz

These can be served with the same `vllm serve MODEL --omni` pattern as Voxtral and connected via `endpoints.json`.

---

## Running TTS Studio on the remote server

The simplest approach is to run the entire app on the server:

```bash
# Copy the project
rsync -av --exclude venv --exclude output --exclude __pycache__ /path/to/tts-studio/ server:~/tts-studio/

# On the server
cd ~/tts-studio && ./run.sh
```

Access the UI at `http://your-server:7860`. All models run locally on the server's GPU — no need for endpoints.json.

---

## Testing the connection

After configuring `endpoints.json`, verify the server is reachable:

```bash
# Check the server is up
curl http://your-server:8000/v1/models

# Test generation
curl -X POST http://your-server:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, this is a test.", "model": "mistralai/Voxtral-4B-TTS-2603", "voice": "neutral_female", "response_format": "wav"}' \
  --output test.wav
```

Or from the CLI:

```bash
tts-studio generate "Hello, this is a test." --model voxtral-4b
```
