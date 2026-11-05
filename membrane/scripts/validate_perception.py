#!/usr/bin/env python3
"""Measure how well the camera-only detector sees the room. No API, no network.

Renders real frames from the engine and compares what the detector reports
against what the engine actually placed. The engine's truth is used HERE, in
the test harness, and never reaches the agent: this is how the detector is
scored, not how it works.

    python scripts/validate_perception.py              # standard sweep
    python scripts/validate_perception.py --scenes 40  # deeper
    python scripts/validate_perception.py --show       # per-object detail
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import perception                                   # noqa: E402
from engine.sim import Sim                                     # noqa: E402
from engine.world import random_scene                          # noqa: E402


def truth_for(sim) -> list[dict]:
    """What the engine actually placed, for scoring only."""
    out = []
    for o in sim.scene.objects:
        size, colour, kind = o.label.split()[0], o.label.split()[1], o.kind
        out.append({"label": o.label, "colour": colour, "shape": kind,
                    "size": size, "height": o.height,
                    "pos": (float(o.pos[0]), float(o.pos[1]))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--headings", type=int, default=8)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    seen_total = 0
    colour_ok = shape_ok = 0
    pos_err: list[float] = []
    height_err: list[float] = []
    shape_confusion: dict[tuple[str, str], int] = collections.Counter()
    missed: dict[str, int] = collections.Counter()

    sim = None
    for seed in range(a.scenes):
        sc = random_scene(seed=seed, n_objects=a.objects, clutter="none",
                          with_obstacles=False)
        if sim is not None:
            sim.close()
        sim = Sim(scene=sc, record_video=False)
        truth = truth_for(sim)

        sim.reset_state()
        for _ in range(a.headings):
            # Relative steps: /turn clips at +-180, so absolute bearings stop
            # advancing past half a circle and half the room is never surveyed.
            jpg = sim.look()
            pose = {"x": sim.x, "y": sim.y, "heading": sim.heading}
            dets = perception.detect(jpg, pose)

            # Match each detection to the nearest truth object of that colour.
            # Colour is the one cue that is unambiguous in this scene, so it
            # anchors the match and the other fields are then scored honestly.
            for d in dets:
                # Match on POSITION, not colour. Matching on colour first makes
                # the colour score circular -- a mis-named object simply fails
                # to match and is written off as a phantom, so colour reads
                # 100% however wrong it is.
                t = min(truth, key=lambda t: math.dist(t["pos"], d.world))
                err = math.dist(t["pos"], d.world)
                if err > 1.2:
                    missed["bad position"] += 1
                    continue
                seen_total += 1
                colour_ok += (d.colour == t["colour"])
                if d.colour != t["colour"]:
                    shape_confusion[(t["colour"], d.colour)] += 0  # keep key set
                    missed[f"colour {t['colour']}->{d.colour}"] += 1
                pos_err.append(err)
                height_err.append(abs(t["height"] - d.height))
                if d.shape == t["shape"]:
                    shape_ok += 1
                else:
                    shape_confusion[(t["shape"], d.shape)] += 1
                if a.show:
                    flag = "" if d.shape == t["shape"] else "  <-- shape"
                    print(f"  {t['label']:28} saw {d.colour} {d.shape:9} "
                          f"h={d.height:.2f} (true {t['height']:.2f})  "
                          f"pos err {err:.2f} m{flag}")
            sim.turn(360.0 / a.headings)
    if sim:
        sim.close()

    print("\nCAMERA-ONLY PERCEPTION")
    print(f"  {a.scenes} scenes x {a.headings} headings, {a.objects} objects each")
    print(f"  detections scored: {seen_total}\n")
    if not seen_total:
        print("  nothing detected at all")
        return 1
    print(f"  colour correct   {colour_ok/seen_total*100:5.1f}%")
    print(f"  shape correct    {shape_ok/seen_total*100:5.1f}%")
    print(f"  position error   median {sorted(pos_err)[len(pos_err)//2]:.3f} m   "
          f"mean {sum(pos_err)/len(pos_err):.3f} m   worst {max(pos_err):.3f} m")
    print(f"  height error     median {sorted(height_err)[len(height_err)//2]:.3f} m   "
          f"mean {sum(height_err)/len(height_err):.3f} m   worst {max(height_err):.3f} m")
    if shape_confusion:
        print("\n  shape confusions (true -> reported):")
        for (t, g), n in sorted(shape_confusion.items(), key=lambda kv: -kv[1]):
            print(f"    {t:9} -> {g:9}  {n}")
    if missed:
        print(f"\n  rejected: {dict(missed)}")
    return 0 if shape_ok / seen_total >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
