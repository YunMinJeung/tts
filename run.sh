#!/bin/bash
# Qwen3-TTS 실행 스크립트
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/tts_app.py" "$@"
