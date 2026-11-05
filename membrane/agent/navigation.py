"""Return navigation from camera observations and the robot's own odometry.

No engine scene imports, named-object endpoints, or ground-truth coordinates.
The marker is measured before pickup, while the gripper cannot occlude it.
Successful outbound poses form a reversible route around obstacles.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


#: A 45 degree sweep covers the room in eight looks. The marker is visible from
#: only a minority of headings, so a half-turn survey leaves it unseen and the
#: carry leg falls back to the model, which then drops off the platform.
MAX_SURVEY_TURNS = 8
MAX_CARRY_SURVEY = 8


def relative_angle(target: float, heading: float) -> float:
    return (target - heading + 180.0) % 360.0 - 180.0


def ground_point(u: float, v: float, height: float = 0.0) -> tuple[float, float] | None:
    """Camera ray intersected with a horizontal plane; forward/right metres."""
    q, side = (v - 240.0) / 320.0, (u - 320.0) / 320.0
    sin, cos = math.sin(math.radians(10)), math.cos(math.radians(10))
    down = sin + q * cos
    if down <= 0.02:
        return None
    scale = (0.5 - height) / down
    return 0.18 + scale * (cos - q * sin), scale * side


def marker_from_image(jpg: bytes, pose: dict) -> tuple[float, float] | None:
    """Locate the delivery marker, entirely from RGB pixels.

    Works for both the raised dark platform with white rim (current scene)
    and the legacy two-tone green decal.
    """
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # 1. Dark platform (black slab 70 mm tall with white rim)
    dark = (r < 35) & (g < 35) & (b < 35)
    white = (r > 180) & (g > 180) & (b > 180)
    count, _, stats, cents = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 30 or w < 10 or x <= 2 or x + w >= 637 or y + h >= 477:
            continue
        # Verify white rim near the dark slab
        above_y = max(0, y - 10)
        if not np.any(white[above_y:y + 1, max(0, x - 2):min(640, x + w + 2)]):
            continue
        vb = y + h - 1
        u_center = int(cents[i][0])
        base = ground_point(u_center, vb, height=0.0)
        if base is None:
            continue
        # Platform center is 0.55 m behind its near edge
        forward, right = base[0] + 0.55, base[1]
        heading = math.radians(pose["heading"])
        return (pose["x"] + forward * math.cos(heading) - right * math.sin(heading),
                pose["y"] - forward * math.sin(heading) - right * math.cos(heading))

    # 2. Legacy green platform fallback
    face = (g > r + 35) & (g > b + 35)
    top = (g > r * 1.02) & (r > 90) & (r > b * 1.7) & (g > b * 1.8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(top.astype(np.uint8))
    for index in sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True):
        x, y, w, h, area = stats[index]
        if area < 45 or w < 12 or x <= 2 or x + w >= 637 or y + h >= 477:
            continue
        component = labels == index
        heights = []
        for u in range(x + 2, x + w - 2):
            rows = np.flatnonzero(component[:, u])
            if not len(rows):
                continue
            vt = int(rows[-1])
            below = np.flatnonzero(face[vt + 1:min(vt + 32, 480), u])
            if len(below) < 2:
                continue
            vb = vt + 1 + int(below[-1])
            base = ground_point(u, vb)
            if base is None:
                continue
            q = (vt - 240) / 320
            sin, cos = math.sin(math.radians(10)), math.cos(math.radians(10))
            slab_height = 0.5 - (base[0] - 0.18) * (sin + q * cos) / (cos - q * sin)
            if 0.015 < slab_height < 0.18:
                heights.append(slab_height)
        if len(heights) < 5:
            continue
        height = float(np.median(heights))
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0, :]
        points = [ground_point(float(u), float(v), height) for u, v in hull]
        if any(p is None for p in points):
            continue
        polygon = np.array(points, dtype=np.float32)
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) < 0.1:
            continue
        forward, right = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        if not 0.4 < forward < 9:
            continue
        heading = math.radians(pose["heading"])
        return (pose["x"] + forward * math.cos(heading) - right * math.sin(heading),
                pose["y"] - forward * math.sin(heading) - right * math.cos(heading))
    return None


@dataclass
class ReturnNavigator:
    goal: tuple[float, float] | None = None
    route: list[tuple[float, float]] = field(default_factory=list)
    return_route: list[tuple[float, float]] | None = None
    initial_heading: float | None = None
    prepared: bool = False
    survey_turns: int = 0
    carry_survey: int = 0
    centred: bool = False
    phase: str = "Finding delivery platform"
    last_position: tuple[float, float] | None = None
    stalled: int = 0

    def remember_pose(self, pose: dict) -> None:
        """Retain corners, merge collinear moves, never shortcut an unknown path."""
        point = (pose["x"], pose["y"])
        if self.route and math.dist(point, self.route[-1]) < 0.06:
            return
        if len(self.route) >= 2:
            a, b = self.route[-2:]
            ab, bc = (b[0] - a[0], b[1] - a[1]), (point[0] - b[0], point[1] - b[1])
            cross = abs(ab[0] * bc[1] - ab[1] * bc[0])
            if cross < 0.015 and ab[0] * bc[0] + ab[1] * bc[1] > 0:
                self.route[-1] = point
                return
        self.route.append(point)

    def prepare(self, jpg: bytes, pose: dict):
        """Find the delivery marker before a carried item can cover the camera."""
        self.remember_pose(pose)
        if self.prepared:
            return None
        if self.initial_heading is None:
            self.initial_heading = pose["heading"]
        seen = marker_from_image(jpg, pose)
        if seen is not None:
            self.goal = seen
            angle = self._angle(seen, pose)
            if abs(angle) > 5 and not self.centred:
                self.centred = True
                return "turn", {"degrees": round(angle, 2)}, "aligning the visible delivery marker for a clear measurement"
        if self.goal is not None:
            restore = relative_angle(self.initial_heading, pose["heading"])
            self.prepared = True
            self.phase = "Finding requested object"
            if abs(restore) > 2:
                return "turn", {"degrees": round(restore, 2)}, "delivery marker remembered from camera; resuming object search"
            return None
        if self.survey_turns >= MAX_SURVEY_TURNS:
            # Give up on the pre-measurement, do NOT end the run. The marker is
            # in view from only a minority of headings, and a robot that cannot
            # see it from where it happens to start is not a failed run -- it
            # just has to find the marker later, which the model can do. Raising
            # here burned the whole attempt before the object search began.
            self.prepared = True
            self.phase = "Finding requested object"
            return None
        self.survey_turns += 1
        return "turn", {"degrees": 45}, "looking for the delivery marker before picking up"

    @staticmethod
    def _angle(point, pose):
        bearing = math.degrees(math.atan2(-(point[1] - pose["y"]), point[0] - pose["x"]))
        return relative_angle(bearing, pose["heading"])

    def decide(self, jpg: bytes, pose: dict):
        if not pose["holding"]:
            return None
        if self.goal is None:
            seen = marker_from_image(jpg, pose)
            if seen is not None:
                self.goal = seen
                self.phase = "Approaching delivery platform"
            elif self.carry_survey < MAX_CARRY_SURVEY:
                self.carry_survey += 1
                self.phase = "Carrying, looking for the delivery marker"
                return ("turn", {"degrees": 45},
                        "searching for the delivery marker after pickup")
            else:
                # Full circle from the grasp pose still missed it. The model
                # can find the marker visually -- far better than abandoning
                # a successful grasp.
                self.phase = "Carrying, looking for the delivery marker"
                return None
        if pose["state"] == "Blocked":
            # Obstructed: back out and let the copilot re-route rather than
            # stranding the robot holding the item.
            self.phase = "Carrying, route blocked, recovering"
            return None

        # Head STRAIGHT for the delivery platform. The robot used to retrace
        # its outbound route in reverse -- list(reversed(self.route)) -- which
        # sent it back to where it first spotted the object before turning for
        # the platform, a visibly wrong dog-leg. The shortest path to the drop
        # zone does not pass through where the robot happened to come from; it
        # goes directly, and the copilot bends it around any obstacle in the
        # way while keeping the brief's 0.5 m clearance.
        here = (pose["x"], pose["y"])
        target, standoff = self.goal, 0.60
        self.phase = "Carrying straight to the delivery platform"
        distance = math.dist(here, target)
        angle = self._angle(target, pose)
        if abs(angle) > 2:
            return ("turn", {"degrees": round(angle, 2)},
                    "facing the delivery platform directly, by camera measurement")
        if abs(distance - standoff) <= 0.07:
            self.phase = "Placing object on platform"
            return ("drop", {},
                    "over the camera-measured delivery position; releasing")
        travel = max(-0.4, min(1.2, distance - standoff))
        if self.last_position is not None and math.dist(here, self.last_position) < 0.02:
            self.stalled += 1
        else:
            self.stalled = 0
        if self.stalled >= 3:
            # Not giving up -- hand control back to the model, which can see a
            # way the geometry here cannot.
            self.stalled = 0
            return None
        self.last_position = here
        return ("move", {"distance": round(travel, 3)},
                "driving straight toward the platform; the copilot keeps it clear")
