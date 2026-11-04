#!/usr/bin/env bash
# Run before the mentor sign-off demo and again before judging.
set -uo pipefail
cd "$(dirname "$0")/.."
B=${ENGINE:-http://127.0.0.1:8080}
pass=0; fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1"; fail=$((fail+1)); fi; }

echo "=== ENGINE ==="
chk "engine responding"            "curl -sf $B/health"
chk "/state returns 6 fields only" "[ \$(curl -sf $B/state | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))') -eq 6 ]"
chk "/state has no object data"    "curl -sf $B/state | python3 -c \"
import json,sys
d=json.load(sys.stdin)
assert set(d)=={'state','x','y','heading','holding','retry_count'}, d
assert isinstance(d['holding'], bool)\""
chk "/look returns a JPEG"          "[ \$(curl -sf -o /dev/null -w '%{content_type}' $B/look) = 'image/jpeg' ]"
chk "/look is 640x480"              "curl -sf $B/look | python3 -c 'import sys,io; from PIL import Image; im=Image.open(io.BytesIO(sys.stdin.buffer.read())); assert im.size==(640,480), im.size'"
chk "/depth returns a number or null" "curl -sf $B/depth | python3 -c 'import json,sys; d=json.load(sys.stdin); assert set(d)=={\"distance\"}'"
chk "/pickup never names an object"  "curl -sf -X POST $B/pickup | python3 -c \"
import json,sys
d=json.load(sys.stdin)
assert set(d) <= {'success','reason','state'}, d\""

echo "=== AGENT ==="
chk "leakage gate on prompt+tools"  "python3 -c \"
import sys; sys.path.insert(0,'.')
from agent.loop import SYSTEM, TOOLS
from agent.firewall import assert_prompt_is_clean
assert_prompt_is_clean(SYSTEM, TOOLS)\""
chk "firewall strips engine fields"  "python3 -c \"
import sys; sys.path.insert(0,'.')
from agent.firewall import sanitize
r = sanitize('pickup', {'success': True, 'state': 'Carry', 'object_name': 'obj0'})
assert r == {'success': True, 'state': 'Carry'}, r\""

echo "=== ARTEFACTS ==="
chk "at least one run logged"        "ls ../runs/run_*/log.jsonl"
chk "leakage audit clean"            "python3 -m audit.check_leakage ../runs"
chk "clips meet 720p / 10-20s"       "python3 -m audit.check_clip ../runs"

echo
echo "  $pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
  echo "  READY"
else
  echo "  FIX THE FAILURES ABOVE"
fi
echo
echo "  Reminders that no script can check:"
echo "   - nobody touches keyboard or mouse during a judged run"
echo "   - clips show no team name, logo, IDE or code"
echo "   - three different commands have passed before asking for sign-off"
