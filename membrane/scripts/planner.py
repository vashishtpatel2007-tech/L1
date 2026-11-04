#!/usr/bin/env python3
"""Obstacle-aware path planning for the oracle baseline.

Driving straight at a target fails in two ways once objects are not neatly
spread out, and both showed up in a 60-attempt sweep over random scenes:

  * the straight line clips another object, so the robot collides on approach
  * the robot arrives at a standoff where some OTHER object is nearer, and
    `/pickup` attaches the nearest thing -- delivering the wrong object while
    still reporting success

This module solves both geometrically. It is used only by the oracle, which is
the optimal baseline we score the model against; the agent never sees it and it
reads no engine data the agent is denied, because the oracle is not the agent.

The algorithm is a tangent-graph detour: if a segment passes too close to a
circular obstacle, route via the two tangent-side waypoints and keep the shorter
option, recursing until the path is clear or the depth budget runs out.
"""
from __future__ import annotations

import math

ROBOT_RADIUS = 0.14
CLEARANCE = 0.12          # extra room so a grazing pass still misses


def _obj_radius(o: dict) -> float:
    """Planar radius of an object from its height and kind.

    Sizes are not exposed by /admin/truth, and height is a good proxy: every
    shape in the palette is roughly as wide as it is tall, except cones and
    cylinders which are narrower.
    """
    h = float(o.get("height", 0.3))
    return {"cone": 0.42, "cylinder": 0.38}.get(o.get("kind", ""), 0.52) * h


def _dist_point_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-12:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def blockers(a, b, obstacles, skip=()) -> list[dict]:
    """Objects whose body intrudes on the segment a->b, nearest first."""
    out = []
    for o in obstacles:
        if o["label"] in skip:
            continue
        need = ROBOT_RADIUS + _obj_radius(o) + CLEARANCE
        if _dist_point_segment(tuple(o["pos"]), a, b) < need:
            out.append((math.dist(a, tuple(o["pos"])), o))
    out.sort(key=lambda t: t[0])
    return [o for _, o in out]


def plan(start, goal, obstacles, skip=(), depth: int = 4) -> list[tuple[float, float]]:
    """Waypoints from `start` to `goal` that clear every obstacle.

    Returns the goal alone when the straight line is already clear.
    """
    hit = blockers(start, goal, obstacles, skip)
    if not hit or depth <= 0:
        return [goal]

    o = hit[0]
    cx, cy = o["pos"]
    need = ROBOT_RADIUS + _obj_radius(o) + CLEARANCE + 0.10

    # Step perpendicular to the path, on whichever side is shorter.
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    n = math.hypot(dx, dy) or 1.0
    px, py = -dy / n, dx / n

    best, best_len = None, float("inf")
    for side in (+1, -1):
        wp = (cx + px * need * side, cy + py * need * side)
        if max(abs(wp[0]), abs(wp[1])) > 3.6:          # would clip a wall
            continue
        if blockers(start, wp, obstacles, skip):
            continue
        length = math.dist(start, wp) + math.dist(wp, goal)
        if length < best_len:
            best, best_len = wp, length

    if best is None:
        return [goal]
    return [best] + plan(best, goal, obstacles, skip, depth - 1)


def approach_pose(target: dict, obstacles: list[dict], standoff: float,
                  prefer_from=None) -> tuple[float, float]:
    """Where to stand so that `target` is the NEAREST object.

    `/pickup` attaches whatever is closest within reach, so standing on the
    wrong side of a crowded pair silently fetches the wrong item. This tries
    bearings around the target and keeps the one with the largest margin
    between the target and its nearest rival.
    """
    tx, ty = target["pos"]
    others = [o for o in obstacles if o["label"] != target["label"]]
    best, best_margin = None, -1e9

    for i in range(24):
        ang = 2 * math.pi * i / 24
        sx, sy = tx + standoff * math.cos(ang), ty + standoff * math.sin(ang)
        if max(abs(sx), abs(sy)) > 3.6:
            continue
        d_target = math.hypot(sx - tx, sy - ty)
        d_other = min((math.hypot(sx - o["pos"][0], sy - o["pos"][1]) for o in others),
                      default=99.0)
        margin = d_other - d_target
        if prefer_from is not None:                    # mild bias toward a short drive
            margin -= 0.04 * math.dist((sx, sy), prefer_from)
        if margin > best_margin:
            best, best_margin = (sx, sy), margin

    return best or (tx + standoff, ty)
