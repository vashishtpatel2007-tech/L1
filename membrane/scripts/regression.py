#!/usr/bin/env python3
"""Automated Phase 1 pass-rate measurement.

Runs a set of commands, resolves each to its intended object from ground truth,
executes the run, and checks the RIGHT object reached the drop zone. Reports
pass rate and step count against the oracle's optimum, which is the directness
evidence Phase 1 asks for.

    python scripts/regression.py --mode oracle                  # free, proves the engine
    python scripts/regression.py --mode claude --model claude-haiku-5-5   # cheap
    python scripts/regression.py --mode claude --limit 3        # Sonnet rehearsal

`fail_wrong_object` is detected here and nowhere else: the engine only knows
whether SOMETHING landed in the zone, so correctness needs the resolver.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.commands import COMMAND_SET, resolve, truth  # noqa: E402

ENGINE = "http://127.0.0.1:8080"
ROOT = Path(__file__).resolve().parents[1]


def get(p: str):
    with urllib.request.urlopen(f"{ENGINE}{p}", timeout=30) as r:
        return json.loads(r.read())


def run_oracle(target: str, command: str, obstacles: bool) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "oracle_run.py"),
           "--target", target, "--command", command]
    if obstacles:
        cmd.append("--obstacles")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (out.stderr or out.stdout)[-300:]}


def run_agent(command: str, model: str, mock: bool, max_steps: int,
              obstacles: bool, run_id: str) -> dict:
    cmd = [sys.executable, "-m", "agent.loop", "--command", command,
           "--max-steps", str(max_steps), "--run-id", run_id, "--model", model]
    if mock:
        cmd.append("--mock")
    if obstacles:
        cmd.append("--obstacles")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1800)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": (out.stderr or out.stdout)[-400:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "claude", "mock"], default="oracle")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--max-steps", type=int, default=35)
    ap.add_argument("--limit", type=int, default=0, help="only the first N commands")
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--current-scene", action="store_true",
                    help="score the scene already loaded instead of the fixed demo one")
    a = ap.parse_args()

    t = truth(fixed=not a.current_scene)
    cases = []
    for c in COMMAND_SET:
        r = resolve(c, t)
        if r["verdict"] == "unique":                 # ambiguous commands are not scorable
            cases.append((c, r["target"]))
    if a.limit:
        cases = cases[: a.limit]

    print(f"mode={a.mode} | {len(cases)} scorable commands"
          + (f" | model={a.model}" if a.mode == "claude" else ""))
    print(f"{'command':48} {'expected':15} {'delivered':15} {'steps':>6}  verdict")
    print("-" * 104)

    rows, passed = [], 0
    for i, (command, expected) in enumerate(cases, 1):
        t0 = time.time()
        if a.mode == "oracle":
            res = run_oracle(expected, command, a.obstacles)
            steps = res.get("optimal_steps", 0)
            outcome = res.get("engine_outcome", {})
        else:
            res = run_agent(command, a.model, a.mode == "mock", a.max_steps,
                            a.obstacles, f"reg{i:02d}")
            steps = res.get("steps", 0)
            outcome = res.get("engine_outcome", {})
        delivered = outcome.get("dropped_label")
        in_zone = outcome.get("result") == "pass"
        ok = bool(in_zone and delivered == expected)
        verdict = ("pass" if ok else
                   "fail_wrong_object" if in_zone and delivered else
                   outcome.get("result") or "fail_not_reached")
        passed += ok
        rows.append({"command": command, "expected": expected, "delivered": delivered,
                     "steps": steps, "verdict": verdict,
                     "seconds": round(time.time() - t0, 1),
                     "cost_usd": res.get("cost_usd", 0.0)})
        print(f"{command:48} {expected:15} {str(delivered):15} {steps:6d}  {verdict}")

    print("-" * 104)
    rate = passed / len(cases) if cases else 0.0
    total_cost = sum(r["cost_usd"] or 0 for r in rows)
    avg_steps = sum(r["steps"] for r in rows) / len(rows) if rows else 0
    print(f"PASS RATE {passed}/{len(cases)} = {rate * 100:.0f}%   avg steps {avg_steps:.1f}"
          + (f"   spend ${total_cost:.3f}" if total_cost else ""))

    out = ROOT.parent / "runs" / f"regression_{a.mode}.json"
    out.write_text(json.dumps({"mode": a.mode, "model": a.model if a.mode == "claude" else None,
                               "pass_rate": rate, "passed": passed, "total": len(cases),
                               "avg_steps": round(avg_steps, 2),
                               "total_cost_usd": round(total_cost, 4),
                               "rows": rows}, indent=2))
    print(f"written to {out}")
    return 0 if rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
