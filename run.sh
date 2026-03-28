#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

EXTRAS="all"
if [[ "${1:-}" == "--voxtral-local" ]]; then
    EXTRAS="all,voxtral-local"
    shift
fi

if ! command -v tts-studio &>/dev/null; then
    echo "Installing dependencies ($EXTRAS)..."
    pip install -q -e ".[$EXTRAS]"
else
    echo "Dependencies already installed."
fi

echo "Starting TTS Studio..."
# Open browser once the server is actually ready
(while ! curl -s -o /dev/null http://localhost:7860; do sleep 1; done
 xdg-open "http://localhost:7860" 2>/dev/null || true) &
exec tts-studio
