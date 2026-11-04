#!/usr/bin/env python3
"""Post-hoc leakage audit over a completed run.

The firewall prevents leakage at send time. This proves it afterwards, from the
artefacts a judge will actually read. Run it on every run before submitting.

    python -m audit.check_leakage ../runs/run_mock4
    python -m audit.check_leakage ../runs            # every run

Exit 0 = clean. Non-zero = do not submit this run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Engine-internal vocabulary that must never appear in a log. Note that colour
# and shape WORDS are not banned: they legitimately appear in the user's command
# and in Claude's own reasoning, which it derived from pixels. What must never
# appear is the engine's private naming, its coordinates, or any admin call.
BANNED = [
    (re.compile(r"\bobj\d+\b", re.I), "engine object id"),
    (re.compile(r"\bgeom\d+\b", re.I), "engine geom id"),
    (re.compile(r"\bdropzone\b", re.I), "engine drop-zone geom name"),
    (re.compile(r"/admin/", re.I), "admin endpoint used during a run"),
    (re.compile(r"\bobject_(name|id|colour|color|shape|count|position)s?\b", re.I), "object data field"),
    (re.compile(r"\bdistance_to_(target|object)\b", re.I), "named-target distance"),
    (re.compile(r"\bsegmentation\b", re.I), "segmentation mask"),
    (re.compile(r"\bdrop_zone_(x|y|pos|coord)", re.I), "drop-zone coordinates"),
]

#: Exactly the keys each tool result may contain.
ALLOWED_RESULT_KEYS = {
    "turn": {"heading", "state"},
    "move": {"collided", "distance_travelled", "state", "retry_count"},
    "pickup": {"success", "reason", "state"},
    "drop": {"success", "in_zone", "state"},
    "finish": {"accepted"},
}

FRAME_RE = re.compile(r"^frames/step_\d{3,4}\.png$")


def audit_run(run_dir: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    log = run_dir / "log.jsonl"
    if not log.exists():
        return False, [f"no log.jsonl in {run_dir}"]

    raw = log.read_text()
    for pat, why in BANNED:
        for m in pat.finditer(raw):
            line = raw[: m.start()].count("\n") + 1
            problems.append(f"{why}: {m.group(0)!r} on line {line}")

    for i, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {i} is not valid JSON")
            continue
        if "step" not in d:
            continue
        tool = d.get("tool", "")
        res = d.get("result")
        allowed = ALLOWED_RESULT_KEYS.get(tool)
        if allowed is not None and isinstance(res, dict):
            extra = set(res) - allowed
            if extra:
                problems.append(f"step {d['step']} ({tool}) result has non-allowlisted keys: {sorted(extra)}")
        fp = d.get("frame_path")
        if fp and not FRAME_RE.match(fp):
            problems.append(f"step {d['step']} frame path {fp!r} is not frames/step_NNN.png "
                            f"(a descriptive filename is itself object data)")

    # every referenced frame must exist -- judges check that the images are real
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        fp = d.get("frame_path")
        if fp and not (run_dir / fp).exists():
            problems.append(f"missing frame file: {fp}")

    return (not problems), problems


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "../runs")
    runs = [target] if (target / "log.jsonl").exists() else sorted(
        p for p in target.glob("run_*") if (p / "log.jsonl").exists())
    if not runs:
        print(f"no runs found under {target}")
        return 2

    bad = 0
    for r in runs:
        ok, problems = audit_run(r)
        steps = sum(1 for line in (r / "log.jsonl").read_text().splitlines()
                    if line.strip() and "step" in json.loads(line))
        frames = len(list((r / "frames").glob("*.png"))) if (r / "frames").exists() else 0
        if ok:
            print(f"  CLEAN  {r.name}  ({steps} steps, {frames} frames)")
        else:
            bad += 1
            print(f"  LEAK   {r.name}")
            for p in problems[:12]:
                print(f"           - {p}")
    print()
    if bad:
        print(f"{bad} of {len(runs)} runs FAILED the leakage audit. Do not submit those runs.")
        return 1
    print(f"All {len(runs)} runs clean: no object data reached the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
