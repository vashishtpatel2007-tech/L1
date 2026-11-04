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

    A barrier circle carries its own radius, since it is geometry rather than
    an object with a height to infer from.

    Sizes are not exposed by /admin/truth, and height is a good proxy: every
    shape in the palette is roughly as wide as it is tall, except cones and
    cylinders which are narrower.
    """
    if "_radius" in o:
        return float(o["_radius"])
    h = float(o.get("height", 0.3))
    return {"cone": 0.42, "cylinder": 0.38}.get(o.get("kind", ""), 0.52) * h


def barrier_circles(obstacles: list[tuple[float, float, float, float]],
                    spacing: float = 0.18) -> list[dict]:
    """Represent box barriers as a chain of overlapping circles.

    The planner reasons about circles, and a wall is not one. Leaving barriers
    out of the obstacle set entirely meant routes were planned straight through
    them: with a barrier in the room, delivery fell to 60.7% with 2.64
    collisions per attempt, against 100% and 0.07 on an open floor.

    Each box becomes circles of radius equal to its short half-width, laid
    along its long axis closely enough to overlap.
    """
    out: list[dict] = []
    for i, (cx, cy, hx, hy) in enumerate(obstacles):
        along_x = hx >= hy
        half_long, half_short = (hx, hy) if along_x else (hy, hx)
        n = max(2, int(2 * half_long / spacing) + 1)
        for k in range(n):
            t = -half_long + 2 * half_long * k / (n - 1)
            px, py = (cx + t, cy) if along_x else (cx, cy + t)
            out.append({"label": f"__barrier{i}_{k}", "pos": [px, py],
                        "kind": "barrier", "height": 0.6,
                        "_radius": half_short})
    return out


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



# --------------------------------------------------------------- grid A* --
GRID = 0.10          # metres per cell
ROOM = 4.0           # half-extent of the room


def _grid_dims() -> int:
    return int(2 * ROOM / GRID) + 1


def _to_cell(x: float, y: float) -> tuple[int, int]:
    n = _grid_dims()
    return (min(n - 1, max(0, int((x + ROOM) / GRID))),
            min(n - 1, max(0, int((y + ROOM) / GRID))))


def _to_world(i: int, j: int) -> tuple[float, float]:
    return (i * GRID - ROOM, j * GRID - ROOM)


def build_grid(obstacles, skip=()) -> list[bytearray]:
    """Occupancy inflated by the robot's radius, so a free cell is a cell the
    robot's CENTRE may occupy. Walls are included as the room boundary."""
    n = _grid_dims()
    grid = [bytearray(n) for _ in range(n)]
    margin = ROBOT_RADIUS + CLEARANCE

    for i in range(n):
        for j in range(n):
            x, y = _to_world(i, j)
            if abs(x) > ROOM - margin - 0.02 or abs(y) > ROOM - margin - 0.02:
                grid[i][j] = 1

    for o in obstacles:
        if o["label"] in skip:
            continue
        r = _obj_radius(o) + margin
        ox, oy = o["pos"]
        lo_i, lo_j = _to_cell(ox - r, oy - r)
        hi_i, hi_j = _to_cell(ox + r, oy + r)
        for i in range(lo_i, hi_i + 1):
            for j in range(lo_j, hi_j + 1):
                x, y = _to_world(i, j)
                if (x - ox) ** 2 + (y - oy) ** 2 <= r * r:
                    grid[i][j] = 1
    return grid


def _clear_line(grid, a, b) -> bool:
    """Is the straight segment a->b free on the inflated grid?"""
    steps = max(2, int(math.dist(a, b) / (GRID * 0.5)))
    for k in range(steps + 1):
        t = k / steps
        i, j = _to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if grid[i][j]:
            return False
    return True


def _nearest_free(grid, cell):
    """Snap onto the grid when a start or goal lands inside inflation.

    The goal routinely does: a standoff sits close enough to its object to be
    inside that object's inflated radius. Refusing to plan in that case would
    reject most legitimate approaches.
    """
    n = _grid_dims()
    i0, j0 = cell
    if not grid[i0][j0]:
        return cell
    for radius in range(1, 26):
        for di in range(-radius, radius + 1):
            for dj in (-radius, radius):
                for i, j in ((i0 + di, j0 + dj), (i0 + dj, j0 + di)):
                    if 0 <= i < n and 0 <= j < n and not grid[i][j]:
                        return (i, j)
    return None


def astar(grid, start, goal) -> list[tuple[float, float]] | None:
    """Shortest 8-connected path between two world points, or None."""
    import heapq

    n = _grid_dims()
    s_cell = _nearest_free(grid, _to_cell(*start))
    g_cell = _nearest_free(grid, _to_cell(*goal))
    if s_cell is None or g_cell is None:
        return None
    if s_cell == g_cell:
        return [goal]

    def h(c):
        return math.dist(c, g_cell)

    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_heap = [(h(s_cell), 0.0, s_cell)]
    came: dict = {}
    best = {s_cell: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == g_cell:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return [_to_world(*c) for c in path]
        if g > best.get(cur, 1e18):
            continue
        for di, dj in nbrs:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < n and 0 <= nj < n) or grid[ni][nj]:
                continue
            if di and dj and (grid[cur[0] + di][cur[1]] or grid[cur[0]][cur[1] + dj]):
                continue                     # no cutting a diagonal corner
            step = 1.4142 if di and dj else 1.0
            ng = g + step
            if ng < best.get((ni, nj), 1e18):
                best[(ni, nj)] = ng
                came[(ni, nj)] = cur
                heapq.heappush(open_heap, (ng + h((ni, nj)), ng, (ni, nj)))
    return None


def _smooth(grid, pts, goal, obstacles=None, skip=()) -> list[tuple[float, float]]:
    """Keep only the turns that matter: a grid path is a staircase, and
    following every cell would make the robot shuffle.

    Shortcuts are tested with `blockers`, the same predicate used to decide a
    route needs planning at all. Testing them against the grid instead let a
    shortcut pass within 0.35 m of a barrier that the clearance rule puts at
    0.38 m -- the difference being one cell of discretisation. A planner whose
    smoother disagrees with its own validator will always produce routes that
    look clear and are not.
    """
    if not pts:
        return [goal]
    pts = pts + [goal]

    def clear(a, b) -> bool:
        if obstacles is not None:
            return not blockers(a, b, obstacles, skip)
        return _clear_line(grid, a, b)

    out: list[tuple[float, float]] = []
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not clear(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out or [goal]


def plan(start, goal, obstacles, skip=(), depth: int = 4) -> list[tuple[float, float]]:
    """Waypoints from `start` to `goal` that clear every obstacle.

    Returns the goal alone when the straight line is already clear.
    """
    hit = blockers(start, goal, obstacles, skip)
    if not hit or depth <= 0:
        return [goal]

    # A single perpendicular hop clears one object but cannot get around an
    # extended barrier: a 2.6 m wall needs a 1.3 m lateral detour, not 0.38 m.
    # With a barrier in the room that cost 2.67 collisions per attempt and a
    # 69% delivery rate. Plan properly on a grid and fall back to the hop only
    # if no route exists at all.
    try:
        grid = build_grid(obstacles, skip)
        raw = astar(grid, start, goal)
        if raw:
            return _smooth(grid, raw, goal, obstacles, skip)
    except Exception:
        pass

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
    """Where to stand so that `target` is the NEAREST object AND the drive
    there does not pass through the target itself.

    Two failures this prevents:

    `/pickup` attaches whatever is closest within reach, so standing on the
    wrong side of a crowded pair silently fetches the wrong item.

    And a standoff chosen on the FAR side of the target puts the target between
    the robot and where it is going, so the approach drives straight into it.
    Measured over 150 attempts: every single collision was an approach hitting
    an object, 222 of them, with none against walls and none while carrying.
    Ruling out poses the robot cannot reach in a straight line removes them.
    """
    tx, ty = target["pos"]
    others = [o for o in obstacles if o["label"] != target["label"]]
    t_radius = _obj_radius(target)
    best, best_margin = None, -1e9

    for i in range(36):
        ang = 2 * math.pi * i / 36
        sx, sy = tx + standoff * math.cos(ang), ty + standoff * math.sin(ang)
        if max(abs(sx), abs(sy)) > 3.6:
            continue

        if prefer_from is not None:
            # Would the drive clip the target on the way? Treat the target as
            # an obstacle for this one test, which `plan` deliberately does not.
            gap = _dist_point_segment((tx, ty), tuple(prefer_from), (sx, sy))
            if gap < ROBOT_RADIUS + t_radius + 0.02:
                continue
            # ...or clip something else on the way.
            if blockers(tuple(prefer_from), (sx, sy), others):
                continue

        d_other = min((math.hypot(sx - o["pos"][0], sy - o["pos"][1]) for o in others),
                      default=99.0)
        margin = d_other - standoff
        if prefer_from is not None:
            margin -= 0.04 * math.dist((sx, sy), prefer_from)
        if margin > best_margin:
            best, best_margin = (sx, sy), margin

    if best is not None:
        return best

    # Nothing clean: stand on the side the robot is already on, which at worst
    # stops short rather than driving through the object.
    if prefer_from is not None:
        dx, dy = prefer_from[0] - tx, prefer_from[1] - ty
        n = math.hypot(dx, dy) or 1.0
        return (tx + dx / n * standoff, ty + dy / n * standoff)
    return (tx + standoff, ty)
