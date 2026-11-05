"""A navigation copilot for the model-driven path, built from camera pixels.

The model is the brain: it reads the scene and decides which object to fetch.
But precise metric navigation -- stand here, not there; go around that, keep
half a metre of clearance -- is geometry, and a language model eyeballing one
640x480 frame does it badly. It misjudges how far an object is, drives forward
to grasp something it has not quite reached, and shoves it away or collides.

This copilot does the geometry, from the same pixels the model sees and nothing
else. It never reads engine state. Every object position below is measured off
the camera frame by `perception`, exactly as the model's own vision would be,
so it discloses nothing the brief withholds; it is an on-board vision-and-
planning aid, not a back channel to the simulator.

It does three things:

  report()       a short, camera-derived summary of what is in view and which
                 object best matches the request, to steer the model's choice.
  safe_action()  takes the move the model wants and, if the measured world says
                 it would collide or overshoot, replaces it with a planner step
                 that keeps 0.5 m of clearance -- redirecting around obstacles
                 and stopping at a clean grasp standoff.
  target_world() the committed target's measured position, so the loop can tell
                 when the robot has arrived.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import perception
from .navigation import marker_from_image
from .perception import Detection, Request, WorldModel

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import planner                                            # noqa: E402
except Exception:                                             # pragma: no cover
    planner = None

#: Keep at least this far from any object the robot is not fetching. The brief's
#: own safety margin.
CLEARANCE = 0.5
#: A clean standoff for grasping. Inside the 0.5 m reach with room to spare,
#: because the camera-measured target position carries about 0.12 m of error
#: and a standoff at the very edge of reach turns that error into a missed
#: grasp. 0.36 m keeps the true distance under 0.5 m even at the worst case.
GRASP_STANDOFF = 0.36


@dataclass
class Copilot:
    world: WorldModel = field(default_factory=WorldModel)
    request: Request | None = None
    target: dict | None = None          # the committed target, as a world track
    _committed_xy: tuple[float, float] | None = None
    drop_zone: tuple[float, float] | None = None   # camera-measured platform
    grasp_fails: int = 0                 # consecutive failed pickups at the target
    _last_xy: tuple[float, float] | None = None
    _turns_without_moving: int = 0
    _carry_route: list | None = None
    _carry_goal: tuple[float, float] | None = None
    _replan_carry: bool = False

    def _note_pose(self, pose: dict) -> bool:
        """Track whether the robot is actually moving. Returns True if it has
        turned several times in a row without the position changing -- a sign
        it is stuck hunting a heading and should drive instead."""
        here = (pose["x"], pose["y"])
        if self._last_xy is not None and math.dist(here, self._last_xy) < 0.03:
            self._turns_without_moving += 1
        else:
            self._turns_without_moving = 0
        self._last_xy = here
        return self._turns_without_moving >= 4

    def note_result(self, tool: str, result: dict) -> None:
        """Learn from what just happened, so a failed grasp changes the plan.

        Without this the copilot would read the same geometry next step and
        issue the same pickup, forever: a missed grasp spammed `pickup` 70-odd
        times until the budget ran out. A failure here makes the approach nudge
        and re-aim instead, and after a few, hand the problem to the model.
        """
        if tool == "pickup":
            if result.get("success"):
                self.grasp_fails = 0
            else:
                self.grasp_fails += 1
        if isinstance(result, dict) and result.get("collided"):
            self._replan_carry = True       # the planned route hit something

    # ---------------------------------------------------------------- see --
    def observe(self, jpg: bytes, pose: dict) -> None:
        """Fold this frame into the measured world. Camera pixels only."""
        try:
            self.world.observe(perception.detect(jpg, pose))
        except Exception:
            pass                        # perception is an aid; never fatal
        try:
            seen = marker_from_image(jpg, pose)
            if seen is not None:
                # Smooth it. A single frame's marker estimate jitters by a
                # tenth of a metre, and feeding that raw to the carry heading
                # made the robot hunt around the platform bearing without ever
                # settling. A running mesh of past estimates is steady.
                if self.drop_zone is None:
                    self.drop_zone = seen
                else:
                    self.drop_zone = (0.7 * self.drop_zone[0] + 0.3 * seen[0],
                                      0.7 * self.drop_zone[1] + 0.3 * seen[1])
        except Exception:
            pass

    def set_command(self, command: str) -> None:
        self.request = perception.parse_request(command)

    def _objects(self, pose: dict) -> list[Detection]:
        try:
            return self.world.objects(pose)
        except Exception:
            return []

    def choose_target(self, pose: dict) -> Detection | None:
        """The measured object that best matches the request, or None."""
        if self.request is None:
            return None
        seen = self._objects(pose)
        if not seen:
            return None
        try:
            return perception.choose(self.request, seen)
        except Exception:
            return None

    def target_world(self, pose: dict) -> tuple[float, float] | None:
        """Where the chosen target is, by camera measurement."""
        t = self.choose_target(pose)
        if t is None:
            return self._committed_xy
        self._committed_xy = t.world
        return t.world

    def _sightings(self, world_xy: tuple[float, float]) -> int:
        """How many frames this position has been seen in -- the confidence
        that it is a real object and not a one-frame artefact."""
        best, n = 1e9, 0
        for t in self.world.tracks:
            dd = math.dist(t["world"], world_xy)
            if dd < best:
                best, n = dd, t["n"]
        return n if best < 0.3 else 0

    # --------------------------------------------------------- carry leg --
    DROP_STANDOFF = 0.60                 # where the robot stops to release

    def carry_action(self, pose: dict) -> tuple[str, dict, str] | None:
        """Carry the item to the platform along one planned route.

        The route is the heart of it. Planning a full A* path to the platform
        ONCE and then following its fixed waypoints is what stops the carry leg
        oscillating: with concrete points to drive through, the heading never
        hunts, and the path is both direct (the planner\'s shortest) and clear
        (0.5 m from every object). It is not the retrace the robot used to do --
        that reversed the outbound path and doubled back through where the
        object was found; this goes straight to the platform.

        The route is replanned only when it must be: when the platform is first
        seen, when a collision or a stall says the world is not what was
        planned, or when the platform estimate has shifted a long way. Returns
        None until the platform has been seen, so the model searches for it.
        """
        if planner is None or self.drop_zone is None:
            return None
        here = (pose["x"], pose["y"])
        goal = self.drop_zone
        stuck = self._note_pose(pose)

        # A standoff point short of the platform centre, on the line from here,
        # so the robot stops in front of it rather than on it.
        gx, gy = goal
        gd = math.dist(here, goal) or 1.0
        stop = (gx - (gx - here[0]) / gd * self.DROP_STANDOFF,
                gy - (gy - here[1]) / gd * self.DROP_STANDOFF)

        # Arrived.
        if math.dist(here, goal) <= self.DROP_STANDOFF + 0.12:
            self._carry_route = None
            return ("drop", {}, "copilot: at the platform standoff; releasing")

        # (Re)plan when there is no route, the goal moved, we were bumped, or we
        # have stalled hunting a heading.
        need_plan = (self._carry_route is None
                     or self._carry_goal is None
                     or math.dist(self._carry_goal, goal) > 0.4
                     or self._replan_carry or stuck)
        if need_plan:
            self._replan_carry = False
            self._turns_without_moving = 0
            obstacles = self._obstacles(pose, exclude=None)
            try:
                route = planner.plan(here, stop, obstacles)
            except Exception:
                route = None
            # Drop the first point if it is essentially where we stand.
            route = [w for w in (route or [stop]) if math.dist(here, w) > 0.12]
            self._carry_route = route or [stop]
            self._carry_goal = goal

        # Follow the route: drive at the next waypoint, dropping it once reached.
        while self._carry_route and math.dist(here, self._carry_route[0]) <= 0.18:
            self._carry_route.pop(0)
        aim = self._carry_route[0] if self._carry_route else stop

        bearing = (((math.degrees(math.atan2(-(aim[1] - here[1]),
                    aim[0] - here[0])) - pose["heading"] + 180) % 360) - 180)
        if abs(bearing) > 16:
            return ("turn", {"degrees": round(bearing, 1)},
                    "copilot: turning onto the planned route to the platform")
        step = min(1.0, math.dist(here, aim))
        return ("move", {"distance": round(max(0.2, step), 2)},
                "copilot: carrying along the planned direct route, 0.5 m clear")

    # ------------------------------------------------------------- report --
    def report(self, pose: dict, holding: bool) -> str:
        """A camera-derived briefing for the model.

        Advisory, not an order: the model still decides. It names colours,
        shapes and sizes -- all of them read off the pixels, the same words the
        request itself uses -- and the bearing and distance of each object, so
        the model is choosing from measured geometry instead of guessing depth
        from one frame. It carries no engine identifiers.
        """
        seen = self._objects(pose)
        if not seen:
            return ("VISION (from your camera): no object is confidently in "
                    "view from here; turn to bring more of the room into frame.")

        lines = ["VISION (measured from your camera pixels, advisory):"]
        for d in sorted(seen, key=lambda o: o.distance):
            side = ("ahead" if abs(d.bearing) <= 10 else
                    f"{abs(d.bearing):.0f} deg {'right' if d.bearing > 0 else 'left'}")
            lines.append(
                f"  - a {d.colour} {d.shape}, about {d.distance:.1f} m away, {side}"
                f" (height ~{d.height:.2f} m)")

        if not holding and self.request is not None and not self.request.is_empty:
            pick = self.choose_target(pose)
            if pick is not None:
                where = ("ahead" if abs(pick.bearing) <= 10 else
                         f"{abs(pick.bearing):.0f} deg to your "
                         f"{'right' if pick.bearing > 0 else 'left'}")
                lines.append(
                    f"  => best match for the request: the {pick.colour} {pick.shape}, "
                    f"{where}, {pick.distance:.1f} m off. Turn to face it, then let "
                    f"your moves close in; it will say when to grasp.")
            else:
                lines.append("  => nothing in view matches the request yet; keep looking.")
        return "\n".join(lines)

    # -------------------------------------------------------- safe motion --
    def _obstacles(self, pose: dict, exclude: tuple[float, float] | None) -> list[dict]:
        """The measured objects as planner circles, minus the target.

        Each detection becomes a circle whose radius is half its measured
        width. The target is left out when approaching so the planner routes TO
        it rather than around it.
        """
        out = []
        for i, d in enumerate(self._objects(pose)):
            if exclude is not None and math.dist(d.world, exclude) < 0.25:
                continue
            out.append({"label": f"seen{i}", "pos": [d.world[0], d.world[1]],
                        "kind": d.shape, "height": d.height,
                        "_radius": max(0.10, d.width / 2.0)})
        return out

    def _clear_straight(self, here: tuple[float, float], fwd: tuple[float, float],
                        obstacles: list[dict]) -> float:
        """How far ahead the robot can drive before coming within CLEARANCE of
        any obstacle. The robot's own radius plus the brief's 0.5 m buffer.

        This is the hard cap on any forward move when the path is blocked: it is
        a property of the straight line itself, so no waypoint beyond an
        obstacle can ever license driving through it.
        """
        robot_r = 0.14
        limit = 12.0
        for o in obstacles:
            ox, oy = o["pos"]
            r = o.get("_radius", 0.2) + robot_r + CLEARANCE
            # Project the obstacle centre onto the heading ray.
            along = (ox - here[0]) * fwd[0] + (oy - here[1]) * fwd[1]
            if along <= 0:
                continue                 # behind the robot
            perp = abs((ox - here[0]) * (-fwd[1]) + (oy - here[1]) * fwd[0])
            if perp < r:
                # The ray passes within r of this obstacle; the robot must stop
                # before the near edge of that encounter.
                back = math.sqrt(max(0.0, r * r - perp * perp))
                limit = min(limit, max(0.0, along - back))
        return limit

    def safe_action(self, tool: str, args: dict, pose: dict,
                    holding: bool, depth: float | None) -> tuple[str, dict, str] | None:
        """Vet the model's action against the measured world.

        Only forward moves are touched, and only when the geometry says they are
        unsafe; everything else passes straight through, so the model stays in
        control. Returns a replacement (tool, args, note) or None.
        """
        if planner is None or tool != "move":
            return None
        try:
            distance = float(args.get("distance", 0) or 0)
        except (TypeError, ValueError):
            return None
        if distance <= 0:
            return None

        here = (pose["x"], pose["y"])
        heading = math.radians(pose["heading"])
        fwd = (math.cos(heading), -math.sin(heading))
        dest = (here[0] + fwd[0] * distance, here[1] + fwd[1] * distance)

        if holding:
            # Carrying: the goal is the platform, and NOTHING is excluded --
            # every object in the room is an obstacle to route around, keeping
            # the brief's 0.5 m clearance, on the shortest path to the drop.
            goal_xy = self.drop_zone
            obstacles = self._obstacles(pose, exclude=None)
        else:
            goal_xy = self.target_world(pose)
            obstacles = self._obstacles(pose, exclude=goal_xy)
        target_xy = goal_xy

        # Is anything in the way of the straight drive? This is decided FIRST,
        # because closing on the target (branch 1) must never take the robot
        # through an obstacle sitting between it and that target.
        blocked = bool(obstacles) and planner.blockers(here, dest, obstacles)

        # 1. About to drive into the thing being fetched, with a clear line to
        #    it: stop at a clean standoff and let the grasp happen instead of
        #    shoving it. Pickup only -- while carrying, stopping short is the
        #    navigator's job. Skipped when an obstacle is in the way, so the
        #    detour below handles it.
        if not holding and target_xy is not None and not blocked:
            d_to_target = math.dist(here, target_xy)
            if d_to_target - distance < GRASP_STANDOFF:
                room = max(0.0, d_to_target - GRASP_STANDOFF)
                if room < 0.12:
                    return ("pickup", {},
                            "copilot: at grasp range of the target, grasping instead "
                            "of driving into it")
                return ("move", {"distance": round(room, 2)},
                        f"copilot: closing to a {GRASP_STANDOFF:.2f} m grasp standoff "
                        f"rather than driving through the target")

        # 2. The straight move clips an obstacle: redirect along the planner's
        #    clear route, which keeps the 0.5 m margin. Two things must both
        #    hold -- turn toward the detour, AND never drive past the obstacle
        #    even toward a valid waypoint beyond it.
        if blocked:
            # How far can the robot go straight ahead before coming within
            # CLEARANCE of any obstacle? That caps every forward move here,
            # independently of what the planner returns, so a waypoint sitting
            # beyond the blocker can never license driving through it.
            clear_ahead = self._clear_straight(here, fwd, obstacles)

            goal = target_xy if target_xy is not None else dest
            try:
                route = planner.plan(here, goal, obstacles)
            except Exception:
                route = None
            if route:
                wx, wy = route[0]
                bearing = ((math.degrees(math.atan2(-(wy - here[1]), wx - here[0]))
                            - pose["heading"] + 180) % 360) - 180
                if abs(bearing) > 12 or clear_ahead < 0.25:
                    return ("turn", {"degrees": round(bearing, 1)},
                            "copilot: an obstacle is in the straight path; turning "
                            "onto a clear detour that keeps 0.5 m clearance")
                step = min(distance, math.dist(here, (wx, wy)), clear_ahead)
                return ("move", {"distance": round(max(0.2, step), 2)},
                        "copilot: edging along a detour around an obstacle, 0.5 m clear")
            # No route at all: do not drive into the blockage. Edge forward only
            # as far as is provably clear, otherwise turn to find a way.
            if clear_ahead >= 0.25:
                return ("move", {"distance": round(min(distance, clear_ahead), 2)},
                        "copilot: route blocked ahead; advancing only while clear")
            return ("turn", {"degrees": 45},
                    "copilot: the way ahead is fully blocked; turning to find a "
                    "clear line")

        # 3. Nothing in the way, but the depth ray sees something close: shorten
        #    so a misjudged distance cannot become a collision.
        if depth is not None and depth < distance + CLEARANCE and not holding:
            safe = max(0.2, depth - CLEARANCE)
            if safe < distance:
                return ("move", {"distance": round(safe, 2)},
                        f"copilot: only {depth:.2f} m of clearance ahead, shortening "
                        f"to keep 0.5 m")
        return None
