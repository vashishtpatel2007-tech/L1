#!/usr/bin/env bash
# Public HTTPS URL for your demo. Two uses, both worth real points:
#   1. Judges open it on their own phone -> feels like a shipped product.
#   2. Your laptop dies -> present from a teammate's machine, same URL.
#
# Serve the BUILT frontend from the API first so one URL covers everything:
#   cd apps/web && npm run build && cd ../.. && (cd apps/api && uvicorn main:app --port 8000)
set -euo pipefail
command -v cloudflared >/dev/null || { echo "brew install cloudflared"; exit 1; }
echo "==> opening tunnel to http://localhost:8000"
echo "    Look for the https://<random>.trycloudflare.com line below."
echo "    Turn it into a QR code so judges can scan it: qrencode -t ANSI '<url>'"
exec cloudflared tunnel --url http://localhost:8000
