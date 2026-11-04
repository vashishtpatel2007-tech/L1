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
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner  # noqa: E402

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


def drive_to(tx: float, ty: float, steps: list[str], tol: float = 0.08) -> bool:
    """Turn to face a point and close on it. One hop, no planning."""
    for _ in range(4):
        st = get("/state")
        d = math.hypot(tx - st["x"], ty - st["y"])
        if d <= tol:
            return True
        want = rel(bearing(st["x"], st["y"], tx, ty), st["heading"])
        if abs(want) > 2.0:
            post("/turn", {"degrees": round(want, 2)}); steps.append("turn")
        r = post("/move", {"distance": round(d, 3)}); steps.append("move")
        if not r["collided"]:
            return math.hypot(tx - get("/state")["x"], ty - get("/state")["y"]) <= tol + 0.25
        # Contact: slide around it rather than shoving.
        post("/turn", {"degrees": 40}); steps.append("turn")
        post("/move", {"distance": 0.40}); steps.append("move")
    return False


def go_to(tx: float, ty: float, standoff: float, steps: list[str],
          obstacles=None, skip=()) -> None:
    """Route to a standoff around (tx,ty), detouring around anything in the way.

    A straight run fails as soon as objects are not neatly spread: the line
    clips another object, or the robot ends up with some OTHER object nearer,
    and /pickup takes whatever is closest -- quietly fetching the wrong item.
    """
    st = get("/state")
    start = (st["x"], st["y"])

    if obstacles:
        tgt = next((o for o in obstacles if math.dist(o["pos"], (tx, ty)) < 1e-6), None)
        goal = (planner.approach_pose(tgt, obstacles, standoff, prefer_from=start)
                if tgt else (tx, ty))
        route = planner.plan(start, goal, obstacles,
                             skip=tuple(skip) + ((tgt["label"],) if tgt else ()))
    else:
        d = math.hypot(tx - start[0], ty - start[1]) or 1.0
        goal = (tx - (tx - start[0]) / d * standoff, ty - (ty - start[1]) / d * standoff)
        route = [goal]

    for wp in route:
        drive_to(wp[0], wp[1], steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help='engine label, e.g. "red sphere"')
    ap.add_argument("--command", default="")
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--clip", default="")
    a = ap.parse_args()

    reset = {"command": a.command or f"Bring the {a.target}", "record": True}
    if a.obstacles:                       # only name it when deliberately setting it
        reset["obstacles"] = True
    post("/admin/reset", reset)
    truth = get("/admin/truth")
    obj = next((o for o in truth["objects"] if o["label"] == a.target), None)
    if obj is None:
        raise SystemExit(f"no object labelled {a.target!r}. Available: "
                         f"{[o['label'] for o in truth['objects']]}")

    steps: list[str] = []
    others = truth["objects"]
    go_to(obj["pos"][0], obj["pos"][1], 0.40, steps, obstacles=others)

    # Waypoint routing leaves the robot facing its direction of TRAVEL, not the
    # target, and /pickup requires the object within +/-30 degrees of heading.
    # Driving straight at the target used to supply this for free.
    st = get("/state")
    want = rel(bearing(st["x"], st["y"], obj["pos"][0], obj["pos"][1]), st["heading"])
    if abs(want) > 2.0:
        post("/turn", {"degrees": round(want, 2)}); steps.append("turn")

    for attempt in range(5):
        if post("/pickup").get("success"):
            steps.append("pickup")
            break
        # Not in reach: re-aim and inch in.
        st = get("/state")
        want = rel(bearing(st["x"], st["y"], obj["pos"][0], obj["pos"][1]), st["heading"])
        if abs(want) > 4.0:
            post("/turn", {"degrees": round(want, 2)}); steps.append("turn")
        post("/move", {"distance": 0.14}); steps.append("move")
    else:
        print(json.dumps({"ok": False, "stage": "pickup", "steps": len(steps)}, indent=2))
        return

    # Carrying: the held item moves with us, so exclude it from the obstacles.
    dz = truth["drop_zone"]
    remaining = [o for o in others if o["label"] != a.target]
    # A released object lands CARRY_FWD (0.60 m) AHEAD of the robot, so driving
    # onto the zone centre throws it past the far edge. Stop one carry-length
    # short, facing in, and it comes to rest on the platform.
    st = get("/state")
    vx, vy = dz[0] - st["x"], dz[1] - st["y"]
    vn = math.hypot(vx, vy) or 1.0
    stand = (dz[0] - vx / vn * 0.60, dz[1] - vy / vn * 0.60)
    route = planner.plan((st["x"], st["y"]), stand, remaining)
    for wp in route:
        drive_to(wp[0], wp[1], steps, tol=0.22)
    st = get("/state")
    want = rel(bearing(st["x"], st["y"], dz[0], dz[1]), st["heading"])
    if abs(want) > 3.0:
        post("/turn", {"degrees": round(want, 2)}); steps.append("turn")
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
