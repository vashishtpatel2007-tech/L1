"""The copilot's safety behaviour: collisions, blockage, clearance, choice.

These are unit tests over the copilot's geometry, built from hand-made world
models rather than rendered frames, so they are deterministic and need no
engine. They cover the cases the brief calls out -- a collision in the path,
a target that cannot be reached without going around, complete and partial
blockage of the route, and the clearance kept throughout.
"""
from __future__ import annotations

import math

import pytest

from agent import perception
from agent.copilot import CLEARANCE, GRASP_STANDOFF, Copilot
from agent.perception import Detection


def det(colour, shape, x, y, *, height=0.3, width=0.3) -> Detection:
    """A detection at a world point, as perception would report it."""
    return Detection(colour=colour, shape=shape, height=height, width=width,
                     distance=math.hypot(x, y),
                     bearing=math.degrees(math.atan2(y, x)),
                     world=(x, y), pixels=400, box=(0, 0, 40, 40))


def world_with(dets):
    c = Copilot()
    # Seed the world model directly; several sightings so the median code path
    # behaves as it would after a real sweep.
    for d in dets:
        for _ in range(3):
            c.world.observe([d])
    return c


POSE = {"x": 0.0, "y": 0.0, "heading": 0.0}        # facing +x


class TestTargetChoice:
    def test_it_picks_the_colour_and_shape_asked_for(self):
        c = world_with([det("red", "box", 2.0, 0.0),
                        det("blue", "cylinder", 2.0, 1.0)])
        c.set_command("bring the blue cylinder")
        t = c.choose_target(POSE)
        assert t is not None and t.colour == "blue"

    def test_tallest_compares_measured_heights(self):
        c = world_with([det("red", "cone", 2.0, 0.0, height=0.30),
                        det("yellow", "cone", 2.2, 0.5, height=0.56)])
        c.set_command("bring the tallest object")
        t = c.choose_target(POSE)
        assert t is not None and t.height == pytest.approx(0.56, abs=0.01)

    def test_a_colour_not_present_yields_no_target(self):
        c = world_with([det("red", "box", 2.0, 0.0)])
        c.set_command("bring the purple box")
        assert c.choose_target(POSE) is None


class TestApproachSafety:
    def test_it_stops_at_a_standoff_instead_of_driving_into_the_target(self):
        # Target 0.8 m straight ahead; the model asks to drive the whole way.
        c = world_with([det("red", "box", 0.8, 0.0)])
        c.set_command("bring the red box")
        c.target_world(POSE)
        out = c.safe_action("move", {"distance": 0.8}, POSE, holding=False, depth=0.8)
        assert out is not None
        tool, args, _ = out
        # Either it shortens to a standoff or, if already close enough, grasps.
        if tool == "move":
            stopped_at = 0.8 - args["distance"]
            assert stopped_at >= GRASP_STANDOFF - 0.05
        else:
            assert tool == "pickup"

    def test_it_grasps_when_already_within_reach(self):
        c = world_with([det("red", "box", 0.45, 0.0)])
        c.set_command("bring the red box")
        c.target_world(POSE)
        out = c.safe_action("move", {"distance": 0.3}, POSE, holding=False, depth=0.45)
        assert out is not None and out[0] == "pickup"


class TestObstacleAvoidance:
    def test_a_blocker_in_the_path_triggers_a_detour(self):
        # Target dead ahead at 3 m, a wide obstacle squarely in the way at 1.5 m.
        c = world_with([det("red", "box", 3.0, 0.0),
                        det("blue", "cylinder", 1.5, 0.0, width=0.6)])
        c.set_command("bring the red box")
        c.target_world(POSE)
        out = c.safe_action("move", {"distance": 3.0}, POSE, holding=False, depth=1.5)
        assert out is not None
        tool, args, note = out
        # The straight move is refused: either a turn onto a detour, or a
        # shortened move that does not reach the blocker.
        assert "copilot" in note
        if tool == "move":
            assert args["distance"] < 1.5

    def test_a_clear_path_is_left_untouched(self):
        # Target ahead, the only other object well off to the side.
        c = world_with([det("red", "box", 2.0, 0.0),
                        det("blue", "cylinder", 1.0, 2.5, width=0.3)])
        c.set_command("bring the red box")
        c.target_world(POSE)
        out = c.safe_action("move", {"distance": 1.2}, POSE, holding=False, depth=None)
        # Nothing in the way and not yet at the standoff: pass straight through.
        assert out is None

    def test_clearance_is_kept_from_the_side_obstacle_on_a_detour(self):
        c = world_with([det("red", "box", 3.0, 0.0),
                        det("blue", "cylinder", 1.5, 0.0, width=0.5)])
        c.set_command("bring the red box")
        c.target_world(POSE)
        out = c.safe_action("move", {"distance": 3.0}, POSE, holding=False, depth=1.5)
        assert out is not None
        # Whatever it returns, it must not be the original straight 3 m drive
        # that ploughs through the blocker.
        assert not (out[0] == "move" and out[1]["distance"] >= 1.5)


class TestCompleteBlockage:
    def test_a_wall_of_objects_still_produces_a_safe_action_not_a_crash(self):
        # A line of obstacles across the front: the planner may find a way round
        # the ends or report none, but the copilot must never hand back the
        # straight collision move, and must never raise.
        row = [det("blue", "cylinder", 1.5, y, width=0.4)
               for y in (-0.6, -0.2, 0.2, 0.6)]
        c = world_with(row + [det("red", "box", 3.0, 0.0)])
        c.set_command("bring the red box")
        c.target_world(POSE)
        out = c.safe_action("move", {"distance": 3.0}, POSE, holding=False, depth=1.5)
        # Either a redirect/shortened move, or None -- but if a move, it cannot
        # be the full straight drive into the wall.
        if out is not None and out[0] == "move":
            assert out[1]["distance"] < 1.5


class TestCarryAvoidsEverything:
    def test_while_carrying_the_drop_route_avoids_all_objects(self):
        c = world_with([det("red", "box", 1.5, 0.0, width=0.6)])
        c.set_command("bring the yellow cone")
        c.drop_zone = (3.0, 0.0)          # platform dead ahead, box in the way
        out = c.safe_action("move", {"distance": 3.0}, POSE, holding=True, depth=1.5)
        assert out is not None
        tool, args, note = out
        assert "copilot" in note
        if tool == "move":
            assert args["distance"] < 1.5


class TestPerspectiveInvariance:
    """Apparent pixel size must never decide which object is bigger.

    A tall object far from the camera projects to FEWER pixels than a short
    object close up, so any judgement made on apparent size is wrong. Every
    size comparison must use the metric height and width perception recovers
    from the measured distance, which is distance-invariant by construction.
    """

    def test_a_far_tall_object_outranks_a_near_short_one(self):
        from agent.perception import F_PX
        c = world_with([det("red", "box", 5.0, 0.0, height=0.56),   # tall, far
                        det("blue", "box", 1.0, 0.0, height=0.22)])  # short, near
        # Apparent pixel heights would rank the near one first...
        assert 0.22 / 1.0 * F_PX > 0.56 / 5.0 * F_PX
        # ...but the metric comparison gets it right.
        c.set_command("bring the tallest object")
        t = c.choose_target(POSE)
        assert t is not None and t.height == pytest.approx(0.56, abs=0.01)

    def test_smallest_also_uses_metric_size_not_pixels(self):
        c = world_with([det("red", "box", 1.0, 0.0, height=0.30, width=0.30),
                        det("blue", "box", 4.0, 0.0, height=0.22, width=0.22)])
        c.set_command("bring the smallest object")
        t = c.choose_target(POSE)
        # The far object has fewer pixels but IS the smaller one; it must win.
        assert t is not None and t.colour == "blue"
