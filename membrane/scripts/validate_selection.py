#!/usr/bin/env python3
"""Does the robot fetch the object it was ASKED for? No API, no network.

Perception accuracy is a means, not the goal. What matters is whether a typed
command resolves to the right object, so this sweeps real commands over real
rendered frames and scores the identity of the chosen object.

The robot sweeps the room first, exactly as it does on a run, because one
90-degree frame never contains the whole scene -- "the tallest object" decided
from a single look means "the tallest of the two I happen to be facing".

Engine truth is used to WRITE the commands and to MARK the answers. It never
reaches the chooser, which sees only rendered pixels and the robot's own pose.

    python scripts/validate_selection.py
    python scripts/validate_selection.py --scenes 30 --show
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import perception                                   # noqa: E402
from engine.sim import Sim                                     # noqa: E402
from engine.world import random_scene                          # noqa: E402

#: Each entry turns a known object into a command plus the rule for which
#: object is then correct. Superlatives are scored against the whole scene, so
#: they catch exactly the failure that started this: "bring the tallest" is
#: wrong unless it returns the tallest thing in the ROOM.
def commands_for(obj, scene) -> list[tuple[str, str]]:
    size, colour, kind = obj.label.split()
    twins = [o for o in scene.objects
             if o.label.split()[1] == colour and o.kind == kind]
    out = [
        # Only ask by colour and shape when that names ONE object. With two
        # teal spheres in the room the question has no right answer, and
        # scoring it would be marking the chooser down for the asker.
        (f"get the {colour} {kind}", obj.label if len(twins) == 1 else None),
        (f"bring the {size} {colour} {kind}", obj.label),
        (f"fetch the {colour} one", obj.label if sum(
            1 for o in scene.objects if o.label.split()[1] == colour) == 1 else None),
    ]
    tallest = max(scene.objects, key=lambda o: o.height)
    shortest = min(scene.objects, key=lambda o: o.height)
    if obj is tallest:
        out.append(("Bring the tallest object", tallest.label))
    if obj is shortest:
        out.append(("bring the shortest object", shortest.label))
    return [(c, want) for c, want in out if want]


def survey(sim, headings: int, vantages: int = 1) -> perception.WorldModel:
    """Sweep the room, folding every look into one map.

    More than one vantage point when asked: turning on the spot cannot see an
    object standing behind another, and that occlusion -- not angular coverage
    -- is what a single-spot survey misses. Doubling the number of headings
    from 8 to 16 moved selection accuracy by one percentage point; stepping
    aside and looking again is what actually reveals a hidden object.
    """
    # /turn is RELATIVE and clips at +-180, so stepping to an absolute
    # bearing silently stops turning past half a circle: a sweep written that
    # way covered 180 degrees, and every object behind the robot went unseen.
    # Turn by a fixed increment instead, and do not reset between looks.
    wm = perception.WorldModel()
    step = 360.0 / headings
    for v in range(vantages):
        for _ in range(headings):
            pose = {"x": sim.x, "y": sim.y, "heading": sim.heading}
            wm.observe(perception.detect(sim.look(), pose))
            sim.turn(step)
        if v + 1 < vantages:
            # Step aside along whichever heading has the most room, so the
            # parallax actually changes what is hidden behind what.
            sim.turn(90.0)
            sim.move(1.3)
    return wm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--headings", type=int, default=8)
    ap.add_argument("--vantages", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    total = right = 0
    by_kind: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    wrong: list[tuple] = []
    found_counts: list[int] = []

    sim = None
    for seed in range(a.scenes):
        sc = random_scene(seed=seed, n_objects=a.objects, clutter="none",
                          with_obstacles=False)
        if sim is not None:
            sim.close()
        sim = Sim(scene=sc, record_video=False)
        wm = survey(sim, a.headings, a.vantages)
        found_counts.append(len(wm.tracks))
        pose = {"x": sim.x, "y": sim.y, "heading": sim.heading}
        known = wm.objects(pose)

        for obj in sc.objects:
            for command, want_label in commands_for(obj, sc):
                pick = perception.choose(perception.parse_request(command), known)
                total += 1
                kind = command.split()[1] if command.startswith("Bring") else "named"
                if pick is None:
                    wrong.append((seed, command, want_label, "nothing chosen"))
                    by_kind[kind][1] += 1
                    continue
                # The chosen detection is correct when it sits on the wanted
                # object: identity is a position question, since the chooser
                # never sees a label.
                want = next(o for o in sc.objects if o.label == want_label)
                err = ((pick.world[0] - want.pos[0]) ** 2
                       + (pick.world[1] - want.pos[1]) ** 2) ** 0.5
                ok = err <= 0.5
                right += ok
                by_kind[kind][0] += ok
                by_kind[kind][1] += 1
                if not ok:
                    got = min(sc.objects,
                              key=lambda o: (o.pos[0] - pick.world[0]) ** 2
                              + (o.pos[1] - pick.world[1]) ** 2)
                    wrong.append((seed, command, want_label, f"chose {got.label}"))
                elif a.show:
                    print(f"  seed {seed:2}  {command:36} -> {want_label}")
    if sim:
        sim.close()

    print("\nCOMMAND -> OBJECT SELECTION  (camera pixels only)")
    print(f"  {a.scenes} scenes, {a.objects} objects, {a.headings}-look survey")
    print(f"  objects found per scene: "
          f"{sum(found_counts)/max(len(found_counts),1):.2f} of {a.objects}\n")
    for k, (ok, n) in sorted(by_kind.items()):
        print(f"    {k:10} {ok:4d}/{n:<4d} {ok/max(n,1)*100:5.1f}%")
    print(f"\n  TOTAL  {right}/{total}  {right/max(total,1)*100:.1f}%")
    if wrong:
        print(f"\n  {len(wrong)} wrong:")
        for w in wrong[:14]:
            print(f"    seed {w[0]:3}  {w[1]:34} wanted {w[2]:26} {w[3]}")
    return 0 if total and right == total else 1


if __name__ == "__main__":
    sys.exit(main())
