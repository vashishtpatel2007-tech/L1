"""The simulator. Owns scene, physics, camera and the state machine.

Conventions from the brief, implemented exactly:
  heading 0 = +x axis, positive CLOCKWISE viewed from above
  /move   speed 1 m/s, negative = reverse, stops early on collision
  /turn   speed 90 deg/s, positive = right
  /pickup reach 0.5 m, within +/-30 deg of heading
  collision: stop, back off 0.1 m, collided=true, enter Blocked

The engine knows object colours, shapes and positions. NONE of that is exposed
by any method whose result reaches the agent -- see `state()`.
"""
from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np
from PIL import Image

from .world import (
    CAM_FWD_OFFSET,
    CAM_HEIGHT,
    CAM_TILT_DEG,
    DROP_ZONE_HALF,
    IMG_H,
    IMG_W,
    PICKUP_CONE_DEG,
    PICKUP_REACH,
    ROOM_HALF,
    Scene,
    build_xml,
    default_scene,
)

State = Literal["Idle", "Locate", "GoToObject", "PickUp", "Carry", "Drop", "Done", "Blocked"]

MOVE_SPEED = 1.0          # m/s
TURN_SPEED = 90.0         # deg/s
BACKOFF = 0.1             # m
MAX_RETRIES = 3
PROGRESS_RESET = 0.25     # m of real travel that counts as escaping a blockage
BLOCK_SECONDS = 2.0
# A carried object must clear the camera's near plane. The camera sits
# CAM_FWD_OFFSET ahead of the robot centre at CAM_HEIGHT; an object whose radius
# reaches back to within ~0.16 m of the lens engulfs it and the frame renders as
# an empty room. 0.70 m keeps the largest object (r=0.25) well clear while still
# reading unmistakably as "the robot is holding something".
CARRY_FWD = 0.60
CARRY_CLEAR = 0.10        # keep the carried object's base this far above the floor
VIDEO_FPS = 30
VIDEO_EVERY = 8  # legacy import compatibility; scheduling uses simulated time

# Wall-clock pacing. The engine always advances simulated time correctly -- a
# 1 m move is integrated as exactly 1 m / 1 m/s = 1 s of simulated time -- but
# by default it runs those steps as fast as the CPU manages, so on screen the
# robot teleports. `Sim.realtime` ties simulated time to real time: 1.0 plays
# the motion at the speed the brief specifies, 2.0 plays it at double speed,
# and 0.0 is free-running, which is what the validation sweeps and the test
# suite want (pacing 450 fetches in real time would take hours).
#
# Pacing is applied once per rendered frame rather than once per physics step:
# a 2 ms step cannot be slept on accurately, and the frame boundary is exactly
# where the visible motion changes anyway.
REALTIME_DEFAULT = 0.0


@dataclass
class RunOutcome:
    result: str = ""                 # pass | fail_wrong_object | fail_not_reached | fail_collision
    dropped_label: str | None = None  # engine-side only, for our own testing
    steps: int = 0


@dataclass
class Sim:
    scene: Scene = field(default_factory=default_scene)
    record_video: bool = True
    realtime: float = REALTIME_DEFAULT

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.scene))
        self.data = mujoco.MjData(self.model)
        # One renderer per Sim. Building a second MuJoCo Renderer on the same
        # thread without closing the first corrupts the shared GL context and
        # every subsequent frame renders as garbage stripes -- which silently
        # destroys every run after the first. Sim.close() must be called, and
        # reset_state() exists so a same-scene reset never needs a new context.
        self.renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

        self.robot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.robot_mocap = self.model.body_mocapid[self.robot_bid]
        self.robot_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("robot_body", "robot_nose")
        }
        # The drop zone is a low platform the robot drives ONTO, so contact with
        # it is expected and must never count as a collision.
        self.ignore_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("floor", "dropzone", "dropzone_rim")
        } - {-1}
        # MuJoCo filters contacts between two bodies that both have no degrees of
        # freedom. The robot is a mocap body and the walls are static world geoms,
        # so no contact is EVER generated for them and the robot drives straight
        # through. Objects have free joints and do collide normally. These static
        # geoms are therefore tested explicitly with mj_geomDistance.
        self.static_obstacles = []
        for gi in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            if nm.startswith(("wall", "obstacle")):
                self.static_obstacles.append(gi)
        # object bookkeeping (ENGINE-ONLY)
        self.obj_bids = {
            o.name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.name)
            for o in self.scene.objects
        }
        self.obj_qadr = {
            o.name: self.model.jnt_qposadr[self.model.body_jntadr[self.obj_bids[o.name]]]
            for o in self.scene.objects
        }
        self.obj_geoms = {}
        for o in self.scene.objects:
            bid = self.obj_bids[o.name]
            start = self.model.body_geomadr[bid]
            for g in range(start, start + self.model.body_geomnum[bid]):
                self.obj_geoms[g] = o.name

        self.state: State = "Idle"
        self.retries = 0
        self.blocked_since: float | None = None
        self.holding: str | None = None
        self.video_frames: list[np.ndarray] = []
        self.outcome = RunOutcome()
        self._step_counter = 0
        self.sim_clock = 0.0              # simulated seconds, advances only on steps
        #: The robot's actual route, sampled from the physics rather than from
        #: whatever the browser managed to poll. The console used to build this
        #: itself out of /state reads, which during a run arrive about once a
        #: second: the result was a handful of points joined by long straight
        #: lines that cut corners the robot never cut, and showed a path it
        #: never drove. The engine sees every step, so it records it.
        self.path_trace: list[tuple[float, float, bool]] = []
        self._pace_sim0 = 0.0
        self._pace_wall0 = time.perf_counter()
        self.last_physics: dict[str, Any] = {}

        x, y, h = self.scene.robot_start
        self.x, self.y, self.heading = x, y, h
        self._apply_pose()
        mujoco.mj_forward(self.model, self.data)
        self._settle(300)

    # ------------------------------------------------------------- helpers --
    def _forward(self, heading: float | None = None) -> np.ndarray:
        h = math.radians(self.heading if heading is None else heading)
        return np.array([math.cos(h), -math.sin(h), 0.0])   # clockwise-positive

    def _apply_pose(self) -> None:
        self.data.mocap_pos[self.robot_mocap] = [self.x, self.y, 0.2]
        th = math.radians(-self.heading)                    # mujoco yaw is CCW-positive
        self.data.mocap_quat[self.robot_mocap] = [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]

    def _carry_height(self, name: str) -> float:
        """Ride height so the object's base clears the floor, whatever its size."""
        o = next(x for x in self.scene.objects if x.name == name)
        return CARRY_CLEAR + o.height / 2.0

    def _hold_object_in_place(self) -> None:
        if not self.holding:
            return
        adr = self.obj_qadr[self.holding]
        p = np.array([self.x, self.y, 0.0]) + self._forward() * CARRY_FWD
        target=np.array([p[0],p[1],self._carry_height(self.holding)])
        if getattr(self, '_grasp_from', None) is not None:
            t=min(1.0,max(0.0,(self.sim_clock-self._grasp_started)/.30))
            t=t*t*(3-2*t)
            target=self._grasp_from*(1-t)+target*t
            if t>=1.0:self._grasp_from=None
        self.data.qpos[adr : adr + 3] = target
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        jid = self.model.body_jntadr[self.obj_bids[self.holding]]
        vadr = self.model.jnt_dofadr[jid]
        self.data.qvel[vadr : vadr + 6] = 0.0

    def _physics_step(self) -> None:
        self._apply_pose()
        self._hold_object_in_place()
        mujoco.mj_step(self.model, self.data)
        self._step_counter += 1
        previous_frame = int(self.sim_clock * VIDEO_FPS + 1e-9)
        self.sim_clock += float(self.model.opt.timestep)
        frame_due = int(self.sim_clock * VIDEO_FPS + 1e-9) > previous_frame
        if frame_due:
            last = self.path_trace[-1] if self.path_trace else None
            held = self.holding is not None
            if (last is None or held != last[2]
                    or math.hypot(self.x - last[0], self.y - last[1]) > 0.02):
                self.path_trace.append((round(self.x, 3), round(self.y, 3), held))
                # A fetch is a few hundred samples; the cap only stops an
                # unattended console growing without bound.
                if len(self.path_trace) > 4000:
                    del self.path_trace[:1000]
        if frame_due and (self.record_video or getattr(self, "live_render", False)):
            frame = self._render()
            if self.record_video:
                self.video_frames.append(frame)
        if self.realtime > 0.0 and frame_due:
            self._pace()

    def _record_physics(self, command: str, *, requested: float, achieved: float,
                        unit: str, rate: float, n_steps: int, elapsed: float,
                        collided: bool) -> None:
        """Keep the integration record for the last command.

        This is what the console's physics readout displays, so the numbers on
        screen are the ones the integrator actually used rather than a
        re-derivation that could drift from them. It carries no object
        identities, positions or labels -- nothing the brief withholds from the
        agent -- so it is safe to serve from an endpoint.
        """
        self.last_physics = {
            "command": command,
            "requested": round(requested, 4),
            "achieved": round(achieved, 4),
            "unit": unit,
            "rate": rate,
            "timestep": round(float(self.model.opt.timestep), 6),
            "steps": n_steps,
            "sim_seconds": round(elapsed, 4),
            "expected_seconds": round(abs(requested) / rate, 4),
            "collided": collided,
            "sim_clock": round(self.sim_clock, 3),
            "realtime": self.realtime,
        }

    def _pace_reset(self) -> None:
        """Anchor wall-clock pacing to now.

        Called at the start of each command so that time spent thinking between
        commands -- a model call takes seconds -- is not then "caught up" by
        replaying the next move at a sprint.
        """
        self._pace_sim0 = self.sim_clock
        self._pace_wall0 = time.perf_counter()

    def _pace(self) -> None:
        """Hold the frame until real time has caught up with simulated time."""
        target = (self.sim_clock - self._pace_sim0) / self.realtime
        behind = target - (time.perf_counter() - self._pace_wall0)
        if behind > 0:
            time.sleep(behind)

    def _settle(self, n: int) -> None:
        # Settling is not motion the operator is watching, so it is never
        # paced: sleeping through it would add a visible stall after every drop.
        was, self.record_video = self.record_video, False
        was_rt, self.realtime = self.realtime, 0.0
        for _ in range(n):
            self._physics_step()
        self.record_video, self.realtime = was, was_rt

    STATIC_MARGIN = 0.005

    def _collision(self) -> bool:
        """True if the robot shell touches anything that is not floor, drop-zone
        marking, or the object it is currently carrying.

        Two mechanisms, because one is not enough (see static_obstacles above):
        explicit distance queries for static geometry, contact list for objects.
        """
        body_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_body")
        probes = [body_geom]
        if self.holding:
            # Whatever is being carried moves with the robot, so it is part of
            # the footprint. Without this it slides through walls.
            bid = self.obj_bids[self.holding]
            start = self.model.body_geomadr[bid]
            probes += list(range(start, start + self.model.body_geomnum[bid]))
        for gi in self.static_obstacles:
            for pg in probes:
                dist = mujoco.mj_geomDistance(self.model, self.data, pg, gi, 0.25, None)
                if dist <= self.STATIC_MARGIN:
                    return True
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if not (pair & self.robot_geoms):
                continue
            other = (pair - self.robot_geoms)
            if not other:
                continue
            o = other.pop()
            if o in self.ignore_geoms:
                continue
            if self.holding and self.obj_geoms.get(o) == self.holding:
                continue
            return True
        return False

    def _render(self, camera: str = "rgb") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    # ------------------------------------------------------------ endpoints --
    def look(self) -> bytes:
        """GET /look -- JPEG 640x480 from the robot camera.
        Claude's ONLY source of object information."""
        img = Image.fromarray(self._render())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def depth(self) -> float | None:
        """GET /depth -- single ray straight ahead from the camera centre.
        Returns metres, max 10, or None. Does NOT identify what was hit."""
        origin = np.array([self.x, self.y, CAM_HEIGHT]) + self._forward() * CAM_FWD_OFFSET
        tilt = math.radians(CAM_TILT_DEG)
        f = self._forward()
        vec = np.array([f[0] * math.cos(tilt), f[1] * math.cos(tilt), -math.sin(tilt)])
        vec /= np.linalg.norm(vec)
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model, self.data, origin, vec, None, 1, self.robot_bid, geomid
        )
        if dist < 0 or dist > 10.0:
            return None
        if int(geomid[0]) in self.ignore_geoms:
            return None
        return round(float(dist), 3)

    def move(self, distance: float) -> dict[str, Any]:
        """POST /move -- forward/backward at 1 m/s. Stops early on collision."""
        # Cap at the room span, not smaller: a tighter cap silently truncates a
        # legitimate cross-room carry and the drop then misses the zone.
        distance = float(np.clip(distance, -2 * ROOM_HALF, 2 * ROOM_HALF))
        if self.state in ("Idle", "Locate"):
            self.state = "Carry" if self.holding else "GoToObject"

        n_steps = max(1, int(abs(distance) / (MOVE_SPEED * self.model.opt.timestep)))
        step_len = distance / n_steps
        self._pace_reset()
        clock_at_start = self.sim_clock
        start = np.array([self.x, self.y])
        heading_at_start = self._forward()[:2]
        collided = False

        for _ in range(n_steps):
            f = self._forward()
            # No positional clamp: the walls are real geometry and must produce a
            # real collision. Clamping here silently pinned the robot at the room
            # edge while still reporting a successful move -- the agent was being
            # lied to, and wall collisions could never happen at all.
            self.x += f[0] * step_len
            self.y += f[1] * step_len
            self._physics_step()
            if self._collision():
                collided = True
                break

        if collided:
            f = self._forward()
            sign = 1.0 if step_len >= 0 else -1.0
            self.x -= f[0] * BACKOFF * sign
            self.y -= f[1] * BACKOFF * sign
            self._physics_step()
            self.retries += 1
            self.state = "Blocked"
            if self.retries >= MAX_RETRIES:
                self.state = "Done"
                self.outcome.result = "fail_collision"
        elif abs(float(np.dot(np.array([self.x, self.y]) - start,
                              heading_at_start))) >= PROGRESS_RESET:
            # Escaped and made real progress, so the robot is no longer stuck.
            # The three retries the brief allows are for one blocked situation.
            self.retries = 0
            self.blocked_since = None

        # Measured AFTER the back-off, signed along the heading. Computing it
        # before the back-off overstates progress by exactly BACKOFF, and an
        # unsigned norm reports a back-off as forward motion -- either way the
        # agent reasons from a number that did not happen.
        delta = np.array([self.x, self.y]) - start
        travelled = float(np.dot(delta, heading_at_start))

        if not collided and abs(distance) >= 0.2 and abs(travelled) < 0.02:
            # Commanded to drive and went nowhere: the brief's second Blocked
            # trigger, "no movement for 2 seconds".
            self.retries += 1
            self.state = "Blocked"
            if self.retries >= MAX_RETRIES:
                self.state = "Done"
                self.outcome.result = "fail_collision"

        self._record_physics("move", requested=distance, achieved=travelled,
                             unit="m", rate=MOVE_SPEED, n_steps=n_steps,
                             elapsed=self.sim_clock - clock_at_start,
                             collided=collided)
        return {
            "collided": collided,
            "distance_travelled": round(travelled, 3),
            "state": self.state,
            "retry_count": self.retries,
        }

    def turn(self, degrees: float) -> dict[str, Any]:
        """POST /turn -- turn in place at 90 deg/s. Positive = right."""
        degrees = float(np.clip(degrees, -180.0, 180.0))
        if self.state == "Idle":
            self.state = "Locate"
        n_steps = max(1, int(abs(degrees) / (TURN_SPEED * self.model.opt.timestep)))
        inc = degrees / n_steps
        self._pace_reset()
        clock_at_start = self.sim_clock
        h0 = self.heading
        collided = False
        for _ in range(n_steps):
            previous_heading = self.heading
            self.heading = (self.heading + inc) % 360.0
            self._physics_step()
            if self.holding and self._collision():
                self.heading = previous_heading
                self._apply_pose()
                self._hold_object_in_place()
                mujoco.mj_forward(self.model,self.data)
                collided = True
                self.retries += 1
                self.state = 'Blocked'
                if self.retries >= MAX_RETRIES:
                    self.state = 'Done'
                    self.outcome.result = 'fail_collision'
                break
        if self.state == "Blocked" and not collided:
            self.state = "Carry" if self.holding else "GoToObject"
        self._record_physics(
            "turn", requested=degrees, achieved=((self.heading - h0 + 540) % 360) - 180,
            unit="deg", rate=TURN_SPEED, n_steps=n_steps,
            elapsed=self.sim_clock - clock_at_start, collided=collided)
        return {"heading": round(self.heading, 2), "state": self.state}

    def pickup(self) -> dict[str, Any]:
        """POST /pickup -- attach nearest object within reach and heading cone.
        Returns success/failure ONLY; never identifies the object."""
        if self.holding:
            return {"success": False, "reason": "already_holding", "state": self.state}
        best, best_d = None, 1e9
        f = self._forward()
        for o in self.scene.objects:
            p = self.data.xpos[self.obj_bids[o.name]][:2]
            v = np.array([p[0] - self.x, p[1] - self.y])
            d = float(np.linalg.norm(v))
            if d > PICKUP_REACH or d < 1e-6:
                continue
            cosang = float(np.dot(v / d, f[:2]))
            if math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) > PICKUP_CONE_DEG:
                continue
            if d < best_d:
                best, best_d = o.name, d
        if best is None:
            self.state = "GoToObject"
            return {"success": False, "reason": "nothing_in_reach", "state": self.state}
        self._grasp_from=self.data.qpos[self.obj_qadr[best]:self.obj_qadr[best]+3].copy()
        self._grasp_started=self.sim_clock
        self.holding = best
        self.state = "Carry"
        self._pace_reset()
        for _ in range(round(.30/self.model.opt.timestep)+1):
            self._physics_step()
        return {"success": True, "state": self.state}

    def drop(self) -> dict[str, Any]:
        """POST /drop -- release the held object, report whether it landed in the
        zone. Reporting zone success is required by the brief; zone COORDINATES
        are never exposed."""
        if not self.holding:
            return {"success": False, "in_zone": False, "reason": "not_holding", "state": self.state}
        name = self.holding
        adr = self.obj_qadr[name]
        p = np.array([self.x, self.y, 0.0]) + self._forward() * CARRY_FWD
        self.data.qpos[adr : adr + 3] = [p[0], p[1], max(float(self.data.qpos[adr+2]), self._carry_height(name))]
        self.holding = None
        self.state = "Drop"
        self._settle(220)

        pos = self.data.xpos[self.obj_bids[name]][:2]
        dz = self.scene.drop_zone
        in_zone = bool(abs(pos[0] - dz[0]) <= DROP_ZONE_HALF and abs(pos[1] - dz[1]) <= DROP_ZONE_HALF)
        self.state = "Done"
        label = next(o.label for o in self.scene.objects if o.name == name)
        self.outcome.dropped_label = label
        self.outcome.result = "pass" if in_zone else "fail_not_reached"
        return {"success": True, "in_zone": in_zone, "state": self.state}

    def state_payload(self) -> dict[str, Any]:
        """GET /state -- SANITIZED. Exactly the fields the brief permits:
        state name, robot x/y, heading, holding flag, retry count.
        No object information of any kind."""
        return {
            "state": self.state,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading": round(self.heading, 2),
            "holding": self.holding is not None,
            "retry_count": self.retries,
        }

    # -------------------------------------------------------------- control --
    def close(self) -> None:
        """Release the GL context. Always call before building another Sim."""
        try:
            self.renderer.close()
        except Exception:
            pass

    def reset_state(self) -> None:
        """Return to the start of a run WITHOUT touching the model or renderer.
        This is the normal path between runs: no new GL context, no corruption."""
        mujoco.mj_resetData(self.model, self.data)
        self.state = "Idle"
        self.retries = 0
        self.blocked_since = None
        self.holding = None
        self.video_frames = []
        self.outcome = RunOutcome()
        self._step_counter = 0
        self.sim_clock = 0.0              # simulated seconds, advances only on steps
        #: The robot's actual route, sampled from the physics rather than from
        #: whatever the browser managed to poll. The console used to build this
        #: itself out of /state reads, which during a run arrive about once a
        #: second: the result was a handful of points joined by long straight
        #: lines that cut corners the robot never cut, and showed a path it
        #: never drove. The engine sees every step, so it records it.
        self.path_trace: list[tuple[float, float, bool]] = []
        self._pace_sim0 = 0.0
        self._pace_wall0 = time.perf_counter()
        self.last_physics: dict[str, Any] = {}
        x, y, h = self.scene.robot_start
        self.x, self.y, self.heading = x, y, h
        self._apply_pose()
        mujoco.mj_forward(self.model, self.data)
        self._settle(300)

    def overhead_frame(self) -> np.ndarray:
        """Third-person view. For OUR debugging and the judge-facing README only
        -- never sent to the agent."""
        return self._render(camera="overhead")
