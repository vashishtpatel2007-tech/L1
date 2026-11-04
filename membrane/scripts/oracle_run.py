#!/usr/bin/env python3
"""Oracle run: drives the REAL seven endpoints using engine ground truth.

Two purposes:
  1. Proves the whole HTTP path end to end -- locate, approach, grasp, carry,
     release, clip -- without spending a cent of Claude budget.
  2. Produces the OPTIMAL step count and path length for a command, which is the
     baseline we score Claude's directness against. Phase 1 awards marks for
     going directly, so "Claude took 14 steps against an optimal 9" is a far
     better thing to show a judge than "Claude took 14 steps".

The oracle is NOT the submission. It never informs the agent.

    python scripts/oracle_run.py --target "red sphere"
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request

ENGINE = "http://127.0.0.1:8080"


def get(p: str):
    with urllib.request.urlopen(f"{ENGINE}{p}", timeout=30) as r:
        return r.read() if r.headers.get("content-type", "").startswith("image/") else json.loads(r.read())


def post(p: str, body: dict | None = None):
    req = urllib.request.Request(f"{ENGINE}{p}", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def bearing(fx: float, fy: float, tx: float, ty: float) -> float:
    """Heading that points from (fx,fy) at (tx,ty), clockwise-positive from +x."""
    return math.degrees(math.atan2(-(ty - fy), (tx - fx)))


def rel(target_abs: float, heading: float) -> float:
    return ((target_abs - heading + 180) % 360) - 180


def go_to(tx: float, ty: float, standoff: float, steps: list[str]) -> None:
    """Turn to face a point, then close to `standoff` metres, retrying on contact."""
    for _ in range(6):
        st = get("/state")
        d = math.hypot(tx - st["x"], ty - st["y"])
        if d <= standoff + 0.04:
            return
        want = rel(bearing(st["x"], st["y"], tx, ty), st["heading"])
        if abs(want) > 2.0:
            post("/turn", {"degrees": round(want, 2)}); steps.append("turn")
        r = post("/move", {"distance": round(d - standoff, 3)}); steps.append("move")
        if r["collided"]:
            post("/turn", {"degrees": 35}); steps.append("turn")
            post("/move", {"distance": 0.45}); steps.append("move")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help='engine label, e.g. "red sphere"')
    ap.add_argument("--command", default="")
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--clip", default="")
    a = ap.parse_args()

    post("/admin/reset", {"command": a.command or f"Bring the {a.target}",
                          "obstacles": a.obstacles, "record": True})
    truth = get("/admin/truth")
    obj = next((o for o in truth["objects"] if o["label"] == a.target), None)
    if obj is None:
        raise SystemExit(f"no object labelled {a.target!r}. Available: "
                         f"{[o['label'] for o in truth['objects']]}")

    steps: list[str] = []
    go_to(obj["pos"][0], obj["pos"][1], 0.42, steps)

    for _ in range(4):
        if post("/pickup").get("success"):
            steps.append("pickup")
            break
        post("/move", {"distance": 0.12}); steps.append("move")
    else:
        print(json.dumps({"ok": False, "stage": "pickup", "steps": len(steps)}, indent=2))
        return

    dz = truth["drop_zone"]
    go_to(dz[0], dz[1], 0.30, steps)
    drop = post("/drop"); steps.append("drop")

    out = get("/admin/outcome")
    report = {
        "ok": drop.get("in_zone", False),
        "target": a.target,
        "optimal_steps": len(steps),
        "step_sequence": steps,
        "engine_outcome": out,
    }
    if a.clip:
        report["clip"] = post(f"/admin/clip?out={a.clip}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
