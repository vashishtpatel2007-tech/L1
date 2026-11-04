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


PICKUP_REACH = 0.5          # the engine's grasp radius
PICKUP_CONE = 30.0          # degrees either side of heading
REACH_MARGIN = 0.05         # stay this far inside the reach limit
RIVAL_MARGIN = 0.16         # how much nearer the target must be than any rival
ARRIVAL_SLACK = 0.10        # how far from the planned pose the robot may stop
CONE_SLACK = 8.0            # degrees of heading error to allow for


def approach_pose(target: dict, obstacles: list[dict], standoff: float,
                  prefer_from=None, strict: bool = True) -> tuple[float, float] | None:
    """Where to stand so the grasp takes the RIGHT object, or None to decline.

    `/pickup` attaches whatever is nearest inside the reach cone and does not
    ask which object was wanted, so standing in the wrong place fetches the
    wrong thing. Two requirements are therefore not negotiable:

      1. the target is within the engine's 0.5 m reach, with margin
      2. the target is NEARER than every other graspable object

    and a third is merely strongly preferred:

      3. the drive there does not pass through the target or anything else

    Barriers matter for (3) but never for (2): a wall cannot be picked up, and
    counting it as a rival made the function decline every object standing near
    one -- the walled room fell to 70% delivery on 40 refusals.

    Standoff shrinks until a pose satisfies the hard requirements, because
    standing closer is always allowed and never costs correctness. When even
    the closest ring leaves a rival nearer, the objects are genuinely
    inseparable and None is returned: the brief scores fetching the wrong item
    as its own failure, so declining is strictly better than guessing.
    """
    tx, ty = target["pos"]
    others = [o for o in obstacles if o["label"] != target["label"]]
    # Only graspable objects can steal the grasp. Barrier proxies cannot.
    rivals = [o for o in others if o.get("kind") != "barrier"]
    t_radius = _obj_radius(target)
    floor = ROBOT_RADIUS + t_radius + 0.03          # closer than this is contact

    rings = [standoff, standoff - 0.04, standoff - 0.08, max(floor + 0.02, 0.30)]
    rings = [r for r in rings if r >= floor]

    def candidates(ring: float):
        """Every pose on one ring that is in reach and out of contact."""
        for i in range(72):
            ang = 2 * math.pi * i / 72
            sx, sy = tx + ring * math.cos(ang), ty + ring * math.sin(ang)
            if max(abs(sx), abs(sy)) > 3.6:
                continue
            d_t = math.hypot(sx - tx, sy - ty)
            if d_t > PICKUP_REACH - REACH_MARGIN or d_t < floor:
                continue
            yield (sx, sy), d_t, steal_margin(sx, sy, d_t)

    def steal_margin(sx: float, sy: float, d_t: float) -> float:
        """How much nearer the target is than the nearest object that could
        steal the grasp from this pose.

        `/pickup` takes the nearest object that is BOTH inside the 0.5 m reach
        and inside the 30-degree heading cone, and the robot faces its target,
        so a rival off to the side is harmless no matter how close it is.
        Treating every direction as a threat made the planner decline perfectly
        good fetches -- 13 of 150 on an open floor -- while still missing the
        rivals that actually matter, which sit just inside the cone.

        A negative result means a rival wins and the grasp takes the wrong
        object.
        """
        tdx, tdy = (tx - sx) / d_t, (ty - sy) / d_t
        worst = 99.0
        for o in rivals:
            ox, oy = o["pos"]
            vx, vy = ox - sx, oy - sy
            d_r = math.hypot(vx, vy)
            if d_r < 1e-9 or d_r > PICKUP_REACH:
                continue                      # out of reach: cannot be taken
            cosang = (vx * tdx + vy * tdy) / d_r
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
            if ang > PICKUP_CONE + CONE_SLACK:
                continue                      # outside the cone: harmless
            worst = min(worst, d_r - d_t)
        return worst

    def bias(pose: tuple[float, float]) -> float:
        if prefer_from is None:
            return 0.0
        return 0.04 * math.dist(pose, prefer_from)

    def path_is_clear(pose: tuple[float, float], allow_target_brush: bool) -> bool:
        if prefer_from is None:
            return True
        if blockers(tuple(prefer_from), pose, others):
            return False
        if allow_target_brush:
            return True
        gap = _dist_point_segment((tx, ty), tuple(prefer_from), pose)
        return gap >= ROBOT_RADIUS + t_radius + 0.02

    # Pass 1 and 2: insist on the rival safety band. The second pass tolerates
    # brushing the target on the way in -- nudging the thing you are reaching
    # for is recoverable -- but never driving through anything else. Dropping
    # that second test too sent open-floor collisions from 8 to 46.
    # Rings are tried outermost first and the first that works wins. Scoring
    # every ring together instead looks tidier and is worse: a tighter ring
    # always shows a larger rival margin, so the robot ends up hugging its
    # target and open-floor collisions went from 9 to 61.
    for brush in (False, True):
        for ring in rings:
            best, best_score = None, -1e9
            for pose, d_t, d_o in candidates(ring):
                if d_o < d_t + RIVAL_MARGIN:
                    continue
                if not path_is_clear(pose, brush):
                    continue
                score = (d_o - d_t) - bias(pose)
                if score > best_score:
                    best, best_score = pose, score
            if best is not None:
                return best

    # Pass 3: the safety band is a comfort buffer, not the engine's rule. Ask
    # the question the engine actually asks -- is the target the nearest thing
    # at all? -- and accept any pose where it is. Refusing these cost 13 of 150
    # open-floor fetches that the grasp would have got right.
    # The margin still has to cover the drive: the robot stops within
    # ARRIVAL_SLACK of the planned pose, and a margin thinner than that can be
    # spent entirely on arrival error, which is how objects standing on the
    # platform were still being mixed up.
    for clear_path_only in (True, False):
        for ring in rings:
            best, best_score = None, -1e9
            for pose, d_t, d_o in candidates(ring):
                if d_o - d_t <= ARRIVAL_SLACK:
                    continue                  # a rival could win the grasp
                if clear_path_only and not path_is_clear(pose, True):
                    continue
                score = (d_o - d_t) - bias(pose)
                if score > best_score:
                    best, best_score = pose, score
            if best is not None:
                return best

    # Every reachable pose leaves another object nearer: the target cannot be
    # isolated, and any grasp from here would be fail_wrong_object.
    if strict:
        return None

    if prefer_from is not None:
        dx, dy = prefer_from[0] - tx, prefer_from[1] - ty
        n = math.hypot(dx, dy) or 1.0
    else:
        dx, dy, n = 1.0, 0.0, 1.0
    r = min(standoff, PICKUP_REACH - REACH_MARGIN)
    return (tx + dx / n * r, ty + dy / n * r)



def grasp_takes_target(robot: tuple[float, float], heading_deg: float,
                       target_label: str, objects: list[dict]) -> bool:
    """Would `/pickup` right now attach the object we actually want?

    Reimplements the engine's rule exactly -- nearest object that is inside the
    reach AND inside the heading cone -- against live positions, so it can be
    asked immediately before grasping. Planning a safe pose is not enough on
    its own: the robot stops within a tolerance of that pose, and a failed
    grasp is retried after nudging forward, either of which can quietly bring a
    rival inside the cone.

    Returns False when nothing is graspable at all, since there is then no
    grasp to be right about.
    """
    rx, ry = robot
    h = math.radians(heading_deg)
    fx, fy = math.cos(h), -math.sin(h)        # engine's clockwise-positive heading
    best, best_d = None, 1e9
    for o in objects:
        ox, oy = o["pos"]
        vx, vy = ox - rx, oy - ry
        d = math.hypot(vx, vy)
        if d > PICKUP_REACH or d < 1e-6:
            continue
        cosang = (vx * fx + vy * fy) / d
        if math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) > PICKUP_CONE:
            continue
        if d < best_d:
            best, best_d = o["label"], d
    return best == target_label
