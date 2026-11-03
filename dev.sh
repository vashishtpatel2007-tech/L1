#!/usr/bin/env bash
# Runs API + frontend together. Ctrl-C stops both cleanly.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

cleanup() { echo; echo "stopping…"; kill 0 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "==> API    http://localhost:8000  (docs at /docs)"
( cd apps/api && uvicorn main:app --reload --port 8000 --host 0.0.0.0 ) &

echo "==> WEB    http://localhost:5173"
( cd apps/web && npm run dev ) &

wait
