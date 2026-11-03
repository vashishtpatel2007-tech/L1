#!/usr/bin/env bash
# One command, cold machine -> running stack. Run this ONCE per machine.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> python env"
command -v uv >/dev/null || { echo "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv ] || uv venv --python 3.13 --system-site-packages .venv
source .venv/bin/activate
uv pip install -q -r apps/api/requirements.txt certifi

echo "==> fixing macOS CA certificates (silently breaks every HTTPS call otherwise)"
python - <<'PY'
import ssl, certifi, urllib.request
ctx = ssl.create_default_context(cafile=certifi.where())
try:
    urllib.request.urlopen("https://pypi.org", timeout=10, context=ctx)
    print("   TLS OK via certifi")
except Exception as e:
    print("   TLS still failing:", e)
PY

echo "==> node deps"
( cd apps/web && npm install --no-fund --no-audit --silent )

echo "==> env file"
[ -f .env ] || { cp .env.example .env; echo "   created .env -- add an LLM key if you have one"; }

echo "==> caching model weights locally (do this on GOOD wifi)"
bash scripts/prefetch_models.sh 2>&1 | grep -E "OK|FAILED|dim:|cached" || true

echo "==> seeding demo data"
( cd apps/api && uvicorn main:app --port 8000 >/tmp/_setup_api.log 2>&1 & echo $! > /tmp/_setup_api.pid )
curl -s --retry 30 --retry-connrefused --retry-delay 1 http://127.0.0.1:8000/health >/dev/null || true
curl -s -X POST http://127.0.0.1:8000/demo/seed -H 'Content-Type: application/json' -d '{"n":150}' >/dev/null || true
kill "$(cat /tmp/_setup_api.pid)" 2>/dev/null || true

cat <<'DONE'

==========================================
  SETUP COMPLETE
  Start everything:   ./dev.sh
  Frontend:           http://localhost:5173
  API docs:           http://localhost:8000/docs
  Demo logins:        admin@demo.in / demo1234
==========================================
DONE
