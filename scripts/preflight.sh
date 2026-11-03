#!/usr/bin/env bash
# RUN THIS 30 MINUTES BEFORE YOU PRESENT. Catches the things that kill demos.
set -uo pipefail
cd "$(dirname "$0")/.."
B=${BASE:-http://127.0.0.1:8000}
pass=0; fail=0
chk() { # name, command
  if eval "$2" >/dev/null 2>&1; then echo "  PASS  $1"; pass=$((pass+1));
  else echo "  FAIL  $1"; fail=$((fail+1)); fi
}

echo "=== PREFLIGHT ==="
chk "API responding"            "curl -sf $B/health"
chk "database has records"      "[ \$(curl -sf $B/items/stats | jq -r .total) -gt 0 ]"
chk "AI health endpoint"        "curl -sf $B/ai/health"
chk "triage returns a category" "curl -sf -X POST $B/ai/triage -H 'Content-Type: application/json' -d '{\"text\":\"urgent sewage leak for 5 days, children sick\"}' | jq -e .category"
chk "vision model loaded"       "curl -sf $B/ai/vision/info | jq -e '.tier'"
chk "RAG index present"         "curl -sf $B/ai/health | jq -e '.capabilities.rag.has_index'"
chk "impact metrics compute"    "curl -sf $B/metrics/impact | jq -e .at_national_scale"
chk "frontend bundle built"     "[ -f apps/web/dist/index.html ]"
chk "model weights cached"      "[ -d \$HOME/.cache/torch/hub/checkpoints ]"

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "  READY TO PRESENT" || echo "  FIX THE FAILURES ABOVE BEFORE YOU PRESENT"
echo
echo "=== offline drill: pull your wifi NOW and re-run this ==="
echo "    Everything except the LLM must still pass. If it does, you are safe."
