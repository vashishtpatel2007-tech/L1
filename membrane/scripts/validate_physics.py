#!/usr/bin/env python3
"""Exhaustive physics and manipulation validation. No API, no network.

Runs the whole pick-and-place pipeline in-process across every shape, every
size and many random layouts, and reports where it breaks down rather than
giving a single pass/fail. Everything here is deterministic: the same seeds
produce the same report, so a change that costs reliability shows up as a
number moving.

    python scripts/validate_physics.py                 # standard sweep
    python scripts/validate_physics.py --scenes 80     # deeper
    python scripts/validate_physics.py --clutter both  # the hard layouts
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.sim import BACKOFF, CARRY_FWD, Sim           # noqa: E402
from engine.world import (DROP_ZONE_HALF, PICKUP_CONE_DEG,  # noqa: E402
                          PICKUP_REACH, ROBOT_RADIUS, random_scene)
import planner                                            # noqa: E402


def planar_radius(o) -> float:
    """Half-width the object presents horizontally at the robot's height."""
    if o.kind == "box":
        return o.size[0] * math.sqrt(2)        # worst case: a corner
    if o.kind == "cone":
        # Cones taper. At the robot's centre height the cone is narrower than
        # its base, which is why a cone is contacted closer than its radius.
        z = 0.20
        return max(0.03, o.size[0] * max(0.0, 1.0 - z / max(o.size[1], 1e-6)))
    return o.size[0]


@dataclass
class Outcome:
    delivered: int = 0
    wrong_object: int = 0
    no_grasp: int = 0
    missed_zone: int = 0
    collisions: int = 0
    steps: int = 0
    path: float = 0.0
    notes: list = field(default_factory=list)


def live_positions(sim: Sim) -> list[dict]:
    """Where the objects are NOW, not where they spawned.

    `scene.objects` holds spawn positions. Reading those to decide whether a
    grasp is safe was wrong in exactly the cases that matter: on a crowded
    platform the approach nudges objects, so the stale layout said the target
    was nearest while the simulated one had a rival in front of it. This is
    the validation harness reading the engine's own state directly, not
    anything the API exposes.
    """
    return [{"label": o.label,
             "pos": list(sim.data.xpos[sim.obj_bids[o.name]][:2])}
            for o in sim.scene.objects]


def face(sim: Sim, tx: float, ty: float) -> None:
    b = math.degrees(math.atan2(-(ty - sim.y), tx - sim.x))
    sim.turn(((b - sim.heading + 180) % 360) - 180)


def drive(sim: Sim, tx: float, ty: float, tol: float, out: Outcome,
          obstacles=None, skip=()) -> None:
    """Close on a point, re-planning after a bump rather than guessing.

    Turning a fixed 40 degrees and driving on costs another collision as often
    as it helps, and against a wall it simply scrapes along. Re-planning from
    where the robot actually is routes around whatever it just found.
    """
    for _ in range(6):
        d = math.hypot(tx - sim.x, ty - sim.y)
        if d <= tol:
            return
        face(sim, tx, ty)
        x0, y0 = sim.x, sim.y
        r = sim.move(d - tol * 0.5)
        out.steps += 2
        out.path += math.hypot(sim.x - x0, sim.y - y0)
        if not r["collided"]:
            return
        out.collisions += 1
        # Back away and try a different line. Re-planning from here was tried
        # and measured WORSE -- 92.7% -> 90.7%, collisions 1.57 -> 1.98 --
        # because a robot already in contact sits inside the inflated grid, so
        # the route it gets back starts blocked. A fixed turn after backing off
        # is less clever and more reliable.
        sim.turn(40)
        sim.move(0.35)
        out.steps += 2


def attempt(sim: Sim, label: str, out: Outcome) -> str:
    """One full fetch. Returns the outcome name."""
    target = next(o for o in sim.scene.objects if o.label == label)
    others = [{"label": o.label, "pos": list(o.pos), "kind": o.kind, "height": o.height}
              for o in sim.scene.objects]
    # Barriers are geometry the planner must route around too; leaving them out
    # plans straight through a wall.
    others += planner.barrier_circles(sim.scene.obstacles)

    standoff = max(0.36, ROBOT_RADIUS + planar_radius(target) + 0.07)
    goal = planner.approach_pose(
        {"label": label, "pos": list(target.pos)}, others, standoff,
        prefer_from=(sim.x, sim.y))
    if goal is None:
        # No pose isolates this object from its neighbours. Refusing is correct:
        # the brief counts fetching the wrong item as its own failure mode.
        return "no_grasp"
    for wp in planner.plan((sim.x, sim.y), goal, others, skip=(label,)):
        drive(sim, wp[0], wp[1], 0.10, out, obstacles=others, skip=(label,))

    face(sim, *target.pos)
    for _ in range(5):
        # Never grasp blind. The engine takes the nearest object in the cone,
        # so check that this is the wanted one before committing: arrival
        # tolerance and the nudge below can both bring a rival inside the cone,
        # and a wrong grab is its own scored failure.
        if planner.grasp_takes_target((sim.x, sim.y), sim.heading, label,
                                      live_positions(sim)):
            if sim.pickup()["success"]:
                break
        sim.move(0.10)
        face(sim, *target.pos)
        out.steps += 2
    else:
        return "no_grasp"

    held = sim.holding
    dz = sim.scene.drop_zone
    remaining = [o for o in others if o["label"] != label]
    vx, vy = dz[0] - sim.x, dz[1] - sim.y
    vn = math.hypot(vx, vy) or 1.0
    stand = (dz[0] - vx / vn * CARRY_FWD, dz[1] - vy / vn * CARRY_FWD)
    for wp in planner.plan((sim.x, sim.y), stand, remaining):
        drive(sim, wp[0], wp[1], 0.18, out, obstacles=remaining)
    face(sim, *dz)
    r = sim.drop()
    out.steps += 1

    if sim.outcome.dropped_label != label:
        return "wrong_object"
    return "delivered" if r["in_zone"] else "missed_zone"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=50)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--clutter", default="none",
                    choices=["none", "pickup", "drop", "both", "full"])
    ap.add_argument("--obstacles", action="store_true")
    a = ap.parse_args()

    by_kind: dict[str, Outcome] = collections.defaultdict(Outcome)
    by_size: dict[str, Outcome] = collections.defaultdict(Outcome)
    total = Outcome()
    failures: list[tuple] = []

    sim = None
    for seed in range(a.scenes):
        sc = random_scene(seed=seed, n_objects=a.objects,
                          clutter=a.clutter, with_obstacles=a.obstacles)
        if sim is not None:
            sim.close()
        sim = Sim(scene=sc, record_video=False)
        for o in sc.objects:
            sim.reset_state()
            run = Outcome()
            try:
                res = attempt(sim, o.label, run)
            except Exception as e:                       # a crash is a failure
                res = "no_grasp"
                failures.append((seed, o.label, f"EXCEPTION {type(e).__name__}: {e}"))
            size = o.label.split()[0]
            for bucket in (by_kind[o.kind], by_size[size], total):
                setattr(bucket, res, getattr(bucket, res) + 1)
                bucket.collisions += run.collisions
                bucket.steps += run.steps
                bucket.path += run.path
            if res != "delivered":
                failures.append((seed, o.label, res))
    if sim:
        sim.close()

    n = (total.delivered + total.wrong_object + total.no_grasp + total.missed_zone)
    print(f"\nPHYSICS & MANIPULATION VALIDATION")
    print(f"  {a.scenes} scenes x {a.objects} objects = {n} full pick-and-place attempts")
    print(f"  clutter={a.clutter}  obstacles={a.obstacles}\n")

    def row(name, o):
        t = o.delivered + o.wrong_object + o.no_grasp + o.missed_zone
        if not t:
            return
        print(f"    {name:10} {o.delivered:4d}/{t:<4d} {o.delivered/t*100:5.1f}%   "
              f"wrong {o.wrong_object:3d}  nograsp {o.no_grasp:3d}  missed {o.missed_zone:3d}  "
              f"collisions {o.collisions:4d}  avg_steps {o.steps/t:5.1f}")

    print("  by shape:")
    for k in sorted(by_kind):
        row(k, by_kind[k])
    print("  by size:")
    for k in sorted(by_size):
        row(k, by_size[k])
    print("\n  TOTAL:")
    row("all", total)

    rate = total.delivered / n if n else 0.0
    print(f"\n  delivery rate {rate*100:.1f}%   "
          f"collisions per attempt {total.collisions/max(n,1):.2f}   "
          f"avg path {total.path/max(n,1):.2f} m")
    if failures:
        c = collections.Counter(f[2].split(":")[0] for f in failures)
        print(f"  failure modes: {dict(c)}")
        for f in failures[:10]:
            print(f"    seed {f[0]:3d}  {f[1]:26} {f[2]}")
    return 0 if rate >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
