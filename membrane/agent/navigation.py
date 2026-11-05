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

    # Reconstruct the pale top in its horizontal plane. Adding a fixed offset
    # to an oblique/occluded dark edge shifts the goal into a surrounding wall.
    # The marker is unshaded white. Reject neutral grey barrier pixels and
    # JPEG fringes which otherwise inflate its projected footprint.
    pale = ((rgb.min(axis=2)>200) & (rgb.max(axis=2)-rgb.min(axis=2)<40)).astype(np.uint8)
    pale = cv2.morphologyEx(pale,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(pale)
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area<60 or w<15 or x<2 or x+w>637 or y+h>477:
            continue
        contours,_=cv2.findContours((labels==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hull=cv2.convexHull(max(contours,key=cv2.contourArea))[:,0,:]
        points=[ground_point(float(u),float(v),.078) for u,v in hull]
        if any(p is None for p in points):continue
        polygon=np.asarray(points,dtype=np.float32)
        moments=cv2.moments(polygon)
        # A heavily occluded platform is not a valid centre measurement.
        if not 1.65 < abs(moments['m00']) < 3.0:continue
        forward,right=moments['m10']/moments['m00'],moments['m01']/moments['m00']
        heading=math.radians(pose['heading'])
        wx=pose['x']+forward*math.cos(heading)-right*math.sin(heading)
        wy=pose['y']-forward*math.sin(heading)-right*math.cos(heading)
        if max(abs(wx),abs(wy))<3.7:
            return wx,wy

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
    barrier_points: list = field(default_factory=list)
    planned_path: list | None = None
    approach: tuple | None = None
    last_goal: tuple | None = None
    exploration_target: tuple | None = None
    explorations: int = 0
    exploration_route: list = field(default_factory=list)
    survey_centre: tuple | None = None
    survey_normal: tuple | None = None
    exploration_lookpoint: tuple | None = None

    def observe_barriers(self, jpg, pose):
        from .barriers import observe_barriers
        fresh = observe_barriers(jpg, pose)
        for point in fresh:
            if all(math.dist(point, q) > .13 for q in self.barrier_points):
                self.barrier_points.append(point)
        self.barrier_points = self.barrier_points[-800:]

    def barrier_obstacles(self):
        return [dict(label=f"visual_barrier_{i}",pos=p,kind="barrier",_radius=.08)
                for i,p in enumerate(self.barrier_points)]


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
        self.observe_barriers(jpg, pose)
        if self.prepared:
            return None
        if self.exploration_target is not None:
            here=(pose['x'],pose['y'])
            distance=math.dist(here,self.exploration_target)
            if distance>.08:
                angle=self._angle(self.exploration_target,pose)
                if abs(angle)>3:
                    return 'turn',{'degrees':round(angle,2)},'checking a clear side to uncover the hidden drop zone'
                return 'move',{'distance':round(min(1.6,distance),3)},'moving to inspect the drop zone from another side'
            self.exploration_target=None
            if self.exploration_route:
                self.exploration_target=self.exploration_route.pop(0)
                return self.prepare(jpg,pose)
            self.survey_turns=0
            if self.exploration_lookpoint is not None:
                angle=self._angle(self.exploration_lookpoint,pose)
                if abs(angle)>3:
                    return 'turn',{'degrees':round(angle,2)},'inspecting the delivery area from this side'
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
        if self.survey_turns >= (2 if self.explorations else (4 if len(self.barrier_points)>3 else MAX_SURVEY_TURNS)) and self.goal is None and self.explorations<2:
            from scripts import planner
            here=(pose['x'],pose['y'])
            origin=self.route[0] if self.route else here
            if self.barrier_points:
                # Look beyond each end of the observed barrier. The destination
                # is inferred from pixels, and every connecting leg is planned.
                if self.survey_centre is None:
                    points=np.asarray(self.barrier_points)
                    centre=np.median(points,axis=0)
                    normal=centre-np.asarray(origin)
                    normal=normal/max(.01,float(np.linalg.norm(normal)))
                    self.survey_centre=tuple(centre)
                    self.survey_normal=tuple(normal)
                centre=np.asarray(self.survey_centre)
                normal=np.asarray(self.survey_normal)
                self.exploration_lookpoint=tuple(centre+normal*.95)
                tangent=np.asarray([-normal[1],normal[0]])
                side=1 if self.explorations==0 else -1
                # Stand far enough back to see the enlarged marker's entire
                # footprint through the open side (including the lens offset).
                dest=tuple(centre+normal*.95+tangent*side*2.1)
            else:
                dest=(origin[0]+(1.6 if self.explorations==0 else -1.6),origin[1])
            if self.explorations:
                back=planner.plan(here,origin,self.barrier_obstacles())
                onward=planner.plan(origin,dest,self.barrier_obstacles())
                path=back+onward if back and onward else []
            else:
                path=planner.plan(here,dest,self.barrier_obstacles())
            if path:
                self.exploration_route=path
                self.exploration_target=self.exploration_route.pop(0)
                self.explorations+=1
                return 'turn',{'degrees':round(self._angle(self.exploration_target,pose),2)},'marker occluded; inspecting an alternate side before pickup'
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
        self.observe_barriers(jpg, pose)
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

        from scripts import planner
        here = (pose["x"], pose["y"])
        obstacles = self.barrier_obstacles()
        if (self.planned_path is None or self.last_goal is None
                or math.dist(self.goal,self.last_goal)>.2
                or (self.planned_path and planner.blockers(here,self.planned_path[0],obstacles))):
            candidate = planner.delivery_route(here,self.goal,obstacles)
            if candidate is None:
                self.phase = "No measured route; checking alternate sides"
                return "turn", {"degrees":45}, "checking another view; no verified clear approach yet"
            self.planned_path, self.approach = candidate
            self.last_goal = self.goal
        while self.planned_path and math.dist(here,self.planned_path[0])<.08:
            self.planned_path.pop(0)
        if not self.planned_path:
            angle=self._angle(self.goal,pose)
            if abs(angle)>2:
                return "turn", {"degrees":round(angle,2)}, "aligning release toward the measured drop zone"
            self.phase="Placing object on platform"
            return "drop", {}, "at a clear camera-measured release pose"
        aim=self.planned_path[0]
        angle=self._angle(aim,pose)
        if abs(angle)>2:
            return "turn", {"degrees":round(angle,2)}, "turning onto the shortest validated approach among available sides"
        travel=min(1.6,math.dist(here,aim))
        self.phase="Carrying around observed barriers"
        return "move", {"distance":round(travel,3)}, "following a validated camera-derived route to a clear side"
