#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# uv provisions Python 3.12 without sudo and resolves dependencies in
# seconds, failing fast with a clear message on conflicting pins where
# pip backtracks for hours.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# Recreate the venv if missing, relocated, or on Python < 3.11
# (f5-tts caps numpy at 1.26.4 on <3.11, conflicting with dia's numpy>=2.2.4)
if [ ! -d "venv" ] || ! grep -q "$SCRIPT_DIR/venv" "venv/pyvenv.cfg" 2>/dev/null \
    || ! venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "Creating virtual environment with Python 3.12..."
    rm -rf venv
    uv venv venv --python 3.12
fi

source venv/bin/activate

EXTRAS="all,mcp"

# Pick the PyTorch wheel index that matches this machine's hardware. The default
# PyPI wheel (cu126) covers Maxwell..Hopper but has no Blackwell (sm_120) kernels,
# so a 50-series GPU silently fails at runtime; the cu128 wheel adds sm_120 but
# drops Maxwell. We only override the default in the two cases it gets wrong:
# Blackwell (needs cu128) and no NVIDIA GPU (lean CPU wheel). Empty = keep default.
detect_torch_index() {
    if command -v nvidia-smi &>/dev/null; then
        local cap major
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
        major="${cap%%.*}"
        if [ -n "$major" ] && [ "$major" -ge 12 ] 2>/dev/null; then
            echo "https://download.pytorch.org/whl/cu128"  # Blackwell (sm_120+)
        fi
        # GPU present but pre-Blackwell (or cap unknown): default wheel covers it
        return
    fi
    echo "https://download.pytorch.org/whl/cpu"  # no NVIDIA GPU
}

if ! command -v tts-studio &>/dev/null; then
    echo "Installing dependencies ($EXTRAS)..."
    uv pip install -e ".[$EXTRAS]"
    # The resolve above pulls the default PyPI torch (cu126) because the model
    # extras depend on torch transitively. Re-pin torch to the wheel that matches
    # this hardware AFTER the full install, so the resolver can't downgrade it
    # back. Empty index = the default wheel is already correct for this machine.
    TORCH_INDEX="$(detect_torch_index)"
    if [ -n "$TORCH_INDEX" ]; then
        echo "Re-pinning PyTorch for this hardware from $TORCH_INDEX ..."
        uv pip install --index-url "$TORCH_INDEX" --upgrade torch torchaudio
    fi
else
    echo "Dependencies already installed."
fi

echo "Starting MCP server on port 8900..."
tts-studio mcp &
MCP_PID=$!

echo "Starting TTS Studio UI..."
# Open browser once the server is actually ready (timeout after 60s)
(for i in $(seq 1 60); do
    curl -s -o /dev/null http://localhost:7860 && break
    sleep 1
done
xdg-open "http://localhost:7860" 2>/dev/null || true) &

# Run UI in foreground; kill MCP server on exit
trap "kill $MCP_PID 2>/dev/null" EXIT
exec tts-studio
