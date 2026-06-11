# Systemd Services

Sample systemd unit files for running TTS Studio as system services.

## Setup

Copy the service files and reload systemd:

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
```

## Services

- **tts-studio-ui** — Gradio web UI on port 7860
- **tts-studio-mcp** — MCP server on port 8900

## Usage

```bash
# Start services
sudo systemctl start tts-studio-ui
sudo systemctl start tts-studio-mcp

# Enable on boot
sudo systemctl enable tts-studio-ui
sudo systemctl enable tts-studio-mcp

# Check status
systemctl status tts-studio-ui
systemctl status tts-studio-mcp

# View logs
journalctl -u tts-studio-ui -f
journalctl -u tts-studio-mcp -f

# Restart after code changes
sudo systemctl restart tts-studio-ui tts-studio-mcp
```

## Customisation

Edit the service files to match your setup:

- **User** — change `User=stewart` to your username
- **WorkingDirectory** / **ExecStart** — update paths if your install is elsewhere
- **GPU** — add `Environment=CUDA_VISIBLE_DEVICES=0` to pin to a specific GPU
- **Port** — append `--port 8080` to `ExecStart` to change the default port
