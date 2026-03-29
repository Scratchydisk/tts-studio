#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Prefer Python 3.12 (some model deps don't support 3.13 yet)
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: No suitable Python interpreter found." >&2
    exit 1
fi

if [ ! -d "venv" ] || ! grep -q "$SCRIPT_DIR/venv" "venv/pyvenv.cfg" 2>/dev/null; then
    echo "Creating virtual environment with $PYTHON..."
    rm -rf venv
    "$PYTHON" -m venv venv
fi

source venv/bin/activate

EXTRAS="all"

if ! command -v tts-studio &>/dev/null; then
    echo "Installing dependencies ($EXTRAS)..."
    pip install -q -e ".[$EXTRAS]"
else
    echo "Dependencies already installed."
fi

echo "Starting TTS Studio..."
# Open browser once the server is actually ready (timeout after 60s)
(for i in $(seq 1 60); do
    curl -s -o /dev/null http://localhost:7860 && break
    sleep 1
done
xdg-open "http://localhost:7860" 2>/dev/null || true) &
exec tts-studio
