"""Real-renderer return regression. Ground truth is used ONLY for test setup/assertions.

The controller receives JPEGs and public robot pose, never the scene or truth.
Run from membrane: python -m scripts.verify_return
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from agent.navigation import ReturnNavigator, relative_angle
from engine.sim import Sim
from engine.world import default_scene


def execute(sim, action):
    tool, args, _ = action
    if tool == "turn":
        return sim.turn(args["degrees"])
    if tool == "move":
        return sim.move(args["distance"])
    if tool == "drop":
        return sim.drop()
    raise AssertionError(tool)


def prepare(sim, nav):
    for _ in range(10):
        action = nav.prepare(sim.look(), sim.state_payload())
        if action is None:
            return
        execute(sim, action)
    raise AssertionError("survey did not finish")


def drive_fixture(sim, nav, point, standoff=0):
    """Fixture only: place the robot at a pickup using a collision-free route."""
    dx, dy = point[0] - sim.x, point[1] - sim.y
    if math.hypot(dx, dy) <= standoff + 0.01:
        return
    sim.turn(relative_angle(math.degrees(math.atan2(-dy, dx)), sim.heading))
    result = sim.move(math.hypot(dx, dy) - standoff)
    assert not result["collided"], (point, result)
    nav.remember_pose(sim.state_payload())


def main():
    reports = []
    for obstacles in (False, True):
        scene = default_scene(with_obstacles=obstacles)
        sim = Sim(scene=scene, record_video=False)
        try:
            for obj in scene.objects:
                sim.reset_state()
                nav = ReturnNavigator()
                prepare(sim, nav)
                measured_error = math.dist(nav.goal, scene.drop_zone)
                assert measured_error < 0.20, ("marker calibration", nav.goal, measured_error)
                if obstacles and obj.pos[1] > 1:
                    side = -1.8 if obj.pos[0] < 0 else 1.8
                    drive_fixture(sim, nav, (side, 0))
                    drive_fixture(sim, nav, (side, 2.1))
                drive_fixture(sim, nav, obj.pos, standoff=0.42)
                assert sim.pickup()["success"], obj.label
                started = time.monotonic()
                actions = []
                for _ in range(45):
                    action = nav.decide(sim.look(), sim.state_payload())
                    result = execute(sim, action)
                    actions.append(action[0])
                    assert not result.get("collided"), (obj.label, action, result)
                    if action[0] == "drop":
                        assert result["in_zone"], (obj.label, nav.goal, sim.state_payload())
                        break
                else:
                    raise AssertionError((obj.label, "return did not terminate", actions))
                assert sim.outcome.dropped_label == obj.label
                reports.append({"target": obj.label, "obstacles": obstacles,
                                "result": sim.outcome.result, "steps": len(actions),
                                "return_seconds": round(time.monotonic() - started, 2),
                                "marker_error_m": round(measured_error, 3), "collisions": sim.retries})
                print(json.dumps(reports[-1]), flush=True)
        finally:
            sim.close()
    out = Path(__file__).resolve().parents[2] / "runs" / "return_validation.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"PASS {len(reports)}/{len(reports)} complete returns", flush=True)


if __name__ == "__main__":
    main()
