#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export MEMBRANE_ENGINE="http://127.0.0.1:${PORT:-8080}"
exec ../.venv/bin/python -m uvicorn engine.godot_server:app --no-access-log --timeout-graceful-shutdown 2 --host 127.0.0.1 --port "${PORT:-8080}"
