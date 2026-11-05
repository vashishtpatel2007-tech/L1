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
#: A clean standoff for grasping: inside the 0.5 m reach, outside contact.
GRASP_STANDOFF = 0.42


@dataclass
class Copilot:
    world: WorldModel = field(default_factory=WorldModel)
    request: Request | None = None
    target: dict | None = None          # the committed target, as a world track
    _committed_xy: tuple[float, float] | None = None

    # ---------------------------------------------------------------- see --
    def observe(self, jpg: bytes, pose: dict) -> None:
        """Fold this frame into the measured world. Camera pixels only."""
        try:
            self.world.observe(perception.detect(jpg, pose))
        except Exception:
            pass                        # perception is an aid; never fatal

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

        target_xy = self.target_world(pose) if not holding else self._committed_xy
        obstacles = self._obstacles(pose, exclude=target_xy if not holding else None)

        # 1. About to drive into the thing being fetched: stop at a clean
        #    standoff and let the grasp happen, instead of shoving it.
        if not holding and target_xy is not None:
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
        #    shortest clear route, which keeps the 0.5 m margin.
        if obstacles and planner.blockers(here, dest, obstacles):
            goal = target_xy if (not holding and target_xy is not None) else dest
            try:
                route = planner.plan(here, goal, obstacles)
            except Exception:
                route = None
            if route:
                wx, wy = route[0]
                bearing = ((math.degrees(math.atan2(-(wy - here[1]), wx - here[0]))
                            - pose["heading"] + 180) % 360) - 180
                if abs(bearing) > 12:
                    return ("turn", {"degrees": round(bearing, 1)},
                            "copilot: an obstacle is in the straight path; turning "
                            "onto a clear detour that keeps 0.5 m clearance")
                step = min(distance, math.dist(here, (wx, wy)))
                return ("move", {"distance": round(max(0.2, step), 2)},
                        "copilot: following a detour around an obstacle, 0.5 m clear")

        # 3. Nothing in the way, but the depth ray sees something close: shorten
        #    so a misjudged distance cannot become a collision.
        if depth is not None and depth < distance + CLEARANCE and not holding:
            safe = max(0.2, depth - CLEARANCE)
            if safe < distance:
                return ("move", {"distance": round(safe, 2)},
                        f"copilot: only {depth:.2f} m of clearance ahead, shortening "
                        f"to keep 0.5 m")
        return None
