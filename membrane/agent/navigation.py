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
    """Locate the existing two-tone delivery marker, entirely from RGB pixels.

    Estimate the slab's height from its visible floor-contact edge, rather than
    treating its raised top as floor (which overestimates distance). Reject
    clipped/occluded markers instead of learning a false delivery position.
    """
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
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
        if self.survey_turns >= 8:
            raise RuntimeError("Delivery platform could not be measured from the camera; stopping before pickup.")
        self.survey_turns += 1
        return "turn", {"degrees": 45}, "surveying the room to remember the delivery marker before pickup"

    @staticmethod
    def _angle(point, pose):
        bearing = math.degrees(math.atan2(-(point[1] - pose["y"]), point[0] - pose["x"]))
        return relative_angle(bearing, pose["heading"])

    def decide(self, jpg: bytes, pose: dict):
        if not pose["holding"]:
            raise RuntimeError("Return navigation requires a confirmed pickup.")
        if self.goal is None:
            raise RuntimeError("No camera-observed delivery position; refusing an unverified drop.")
        if pose["state"] == "Blocked":
            raise RuntimeError("The previously clear return route is now obstructed; holding the item safely.")
        here = (pose["x"], pose["y"])
        if self.return_route is None:
            self.return_route = list(reversed(self.route))
        while self.return_route and math.dist(here, self.return_route[0]) <= 0.08:
            self.return_route.pop(0)
        if self.return_route:
            self.phase = "Returning along clear route"
            target, standoff = self.return_route[0], 0.0
        else:
            self.phase = "Approaching delivery platform"
            target, standoff = self.goal, 0.60
        distance = math.dist(here, target)
        angle = self._angle(target, pose)
        if abs(angle) > 2:
            return "turn", {"degrees": round(angle, 2)}, "aligning with the remembered route using measured robot heading"
        if not self.return_route and abs(distance - standoff) <= 0.07:
            self.phase = "Placing object on platform"
            return "drop", {}, "carried item is aligned over the camera-measured delivery position; releasing"
        travel = max(-0.4, min(1.2, distance - standoff))
        if self.last_position is not None and math.dist(here, self.last_position) < 0.02:
            self.stalled += 1
        else:
            self.stalled = 0
        if self.stalled >= 2:
            raise RuntimeError("Return movement is not making progress; stopping with the item held.")
        self.last_position = here
        return "move", {"distance": round(travel, 3)}, "following the observed clear route; checking actual displacement after each move"
