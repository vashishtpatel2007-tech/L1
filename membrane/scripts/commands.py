#!/usr/bin/env python3
"""Command set and ground-truth resolver.

At hour 18 the judges publish three UNSEEN commands and each is run once. We
cannot know the wording, but we can know which *kinds* of command this scene
answers unambiguously -- and fix the scene where it does not.

A command is only fair if exactly one object satisfies it. This module resolves
a command against engine ground truth and reports ambiguity, so the scene can be
corrected before it costs a hidden-command run.

    python scripts/commands.py              # audit the whole command set
"""
from __future__ import annotations

import json
import math
import re
import sys
import urllib.request

ENGINE = "http://127.0.0.1:8080"

SHAPE_WORDS = {
    "sphere": "sphere", "ball": "sphere", "round": "sphere", "circular": "sphere",
    "cube": "box", "box": "box", "square": "box", "boxy": "box",
    "cylinder": "cylinder", "tube": "cylinder", "can": "cylinder",
    "cone": "cone", "conical": "cone", "pointy": "cone", "pyramid": "cone",
}
COLOUR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "white", "black"}


def truth(reset: bool = True, fixed: bool = False) -> dict:
    """Ground truth for a FRESH scene.

    Objects stay where previous runs dropped them, so a snapshot taken without
    resetting resolves distance-based commands against moved objects and yields
    the wrong expected target.

    `fixed` loads the demo scene so a regression number is comparable run to
    run. Without it the harness scores whatever scene is loaded, which is the
    right thing when checking a randomised layout but useless as a baseline.
    """
    if reset:
        body = {"command": "", "record": False}
        if fixed:
            body.update(fixed=True, obstacles=False, clutter="none", randomise=False)
        req = urllib.request.Request(
            f"{ENGINE}/admin/reset", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=60).read()
    with urllib.request.urlopen(f"{ENGINE}/admin/truth", timeout=30) as r:
        return json.loads(r.read())


def resolve(command: str, t: dict) -> dict:
    """Which object does this command mean? Returns matches and a verdict."""
    low = command.lower()
    objs = t["objects"]
    dz = t["drop_zone"]

    # "the green square" names the DROP ZONE, not an object. Remove the phrase
    # before parsing attributes -- discarding "box" afterwards also deleted the
    # shape asked for by an earlier "cube" in the same sentence.
    low_obj = re.sub(r"\bgreen (square|patch|zone|area|marker)\b", " ", low)

    want_shapes = {SHAPE_WORDS[w] for w in SHAPE_WORDS if re.search(rf"\b{w}\b", low_obj)}
    want_colours = {c for c in COLOUR_WORDS if re.search(rf"\b{c}\b", low_obj)}

    cands = objs
    if want_shapes:
        cands = [o for o in cands if o["kind"] in want_shapes]
    if want_colours:
        cands = [o for o in cands if o["colour"] in want_colours]

    # superlatives
    if re.search(r"\btallest\b", low_obj):
        cands = [max(cands or objs, key=lambda o: o["height"])]
    elif re.search(r"\b(shortest|smallest)\b", low_obj):
        cands = [min(cands or objs, key=lambda o: o["height"])]
    elif re.search(r"\bfarthest from\b|\bfurthest from\b", low):
        ref = dz
        cands = [max(cands or objs, key=lambda o: math.dist(o["pos"], ref))]
    elif re.search(r"\bnearest to\b|\bclosest to\b", low):
        cands = [min(cands or objs, key=lambda o: math.dist(o["pos"], dz))]

    return {
        "command": command,
        "matches": [o["label"] for o in cands],
        "verdict": ("unique" if len(cands) == 1 else
                    "AMBIGUOUS" if len(cands) > 1 else "NO MATCH"),
        "target": cands[0]["label"] if len(cands) == 1 else None,
    }


#: Commands we expect judges could plausibly use, including the brief's examples.
COMMAND_SET = [
    # the brief's own examples
    "Bring the round red thing",
    "Bring the tallest object",
    # compound colour+shape (unambiguous by construction)
    "Bring the red cube",
    "Bring the blue cube",
    "Bring the blue cylinder",
    "Bring the yellow cone",
    "Bring the red sphere",
    # single-attribute where only one object has it
    "Bring the cylinder",
    "Bring the cone",
    "Bring the yellow one",
    "Bring the pointy one",
    # superlatives
    "Bring the shortest object",
    "Bring the smallest object",
    # multi-view reasoning (stretch goal)
    "Bring the cube farthest from the green square",
    "Bring the object closest to the green square",
    # deliberately ambiguous -- we want these FLAGGED, not silently guessed
    "Bring the cube",
    "Bring the red thing",
    "Bring the blue thing",
    "Bring the ball",
]


def main() -> int:
    t = truth()
    print("SCENE")
    for o in sorted(t["objects"], key=lambda x: -x["height"]):
        print(f"  {o['label']:16} kind={o['kind']:9} colour={o['colour']:7} "
              f"height={o['height']:.2f}  pos=({o['pos'][0]:+.1f},{o['pos'][1]:+.1f})")
    print(f"  drop zone at ({t['drop_zone'][0]:+.1f},{t['drop_zone'][1]:+.1f})\n")

    heights = [o["height"] for o in t["objects"]]
    print(f"HEIGHTS DISTINCT: {len(set(heights)) == len(heights)}  {sorted(heights)}")
    colours = {}
    shapes = {}
    for o in t["objects"]:
        colours.setdefault(o["colour"], []).append(o["label"])
        shapes.setdefault(o["kind"], []).append(o["label"])
    print(f"SHARED COLOUR: {[v for v in colours.values() if len(v) > 1]}")
    print(f"SHARED SHAPE : {[v for v in shapes.values() if len(v) > 1]}\n")

    print("COMMAND RESOLUTION")
    amb = []
    for c in COMMAND_SET:
        r = resolve(c, t)
        mark = "  ok  " if r["verdict"] == "unique" else f" {r['verdict'][:4]} "
        print(f"  {mark} {c:48} -> {r['matches']}")
        if r["verdict"] != "unique":
            amb.append(c)
    print()
    print(f"{len(COMMAND_SET) - len(amb)}/{len(COMMAND_SET)} resolve to exactly one object.")
    if amb:
        print("Inherently ambiguous (expected for bare colour/shape):")
        for c in amb:
            print(f"  - {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
