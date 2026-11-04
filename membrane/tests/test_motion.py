"""Motion reporting and collision behaviour."""
from __future__ import annotations

import pytest

from engine.sim import MAX_RETRIES, Sim


def test_state_payload_exposes_exactly_the_permitted_fields(sim: Sim):
    assert set(sim.state_payload()) == {"state", "x", "y", "heading", "holding", "retry_count"}


def test_holding_is_a_bare_boolean_not_an_object_identifier(sim: Sim):
    assert isinstance(sim.state_payload()["holding"], bool)


def test_look_returns_a_640x480_jpeg(sim: Sim):
    import io

    from PIL import Image

    jpg = sim.look()
    assert jpg[:2] == b"\xff\xd8", "not a JPEG"
    assert Image.open(io.BytesIO(jpg)).size == (640, 480)


def test_driving_into_a_wall_collides_and_blocks(sim: Sim):
    """Regression: the robot is a mocap body and the walls are static world
    geometry. MuJoCo filters contacts between two bodies that both lack degrees
    of freedom, so wall collisions never appear in the contact list and the robot
    drove straight through. Static geometry is tested with mj_geomDistance."""
    collided = False
    for _ in range(8):
        r = sim.move(1.5)
        if r["collided"]:
            collided = True
            break
    assert collided, "robot passed through a wall"
    assert sim.state == "Blocked"
    assert sim.retries == 1


def test_three_collisions_end_the_run_as_fail_collision(sim: Sim):
    for _ in range(12):
        sim.move(1.5)
        if sim.state == "Done":
            break
    assert sim.retries == MAX_RETRIES
    assert sim.outcome.result == "fail_collision"


def test_distance_travelled_reports_actual_not_requested_displacement(sim: Sim):
    """Regression: move() used to add up the requested step length, so a robot
    pinned against a wall still reported a full successful move. The agent then
    reasons from that lie."""
    for _ in range(8):
        r = sim.move(1.5)
        if r["collided"]:
            assert abs(r["distance_travelled"]) < 1.5, (
                "a blocked move reported the full requested distance"
            )
            return
    pytest.fail("never collided, so the claim was not exercised")


def test_a_blocked_move_is_not_reported_as_forward_progress(sim: Sim):
    """After backing off, net displacement along the heading is <= 0."""
    for _ in range(8):
        r = sim.move(1.5)
        if r["collided"] and abs(r["distance_travelled"]) < 0.2:
            assert r["distance_travelled"] <= 0.01
            return
    pytest.skip("no immediate re-collision occurred in this layout")


def test_obstacle_blocks_the_straight_path(obstacle_sim: Sim):
    """Stretch goal: an added barrier must actually stop the robot."""
    obstacle_sim.reset_state()
    obstacle_sim.turn(-90.0)                 # face +y, toward the barrier
    hit = any(obstacle_sim.move(1.0)["collided"] for _ in range(4))
    assert hit, "the obstacle did not block the path"


class TestBlockedRecovery:
    """The brief: 'Blocked: Collision or no movement for 2 seconds; back off/
    turn/retry. Fail after 3 retries.' Three attempts to escape ONE blockage --
    not three bumps across a whole run."""

    def test_retries_reset_after_the_robot_escapes_and_makes_progress(self, sim: Sim):
        """Regression: carrying the cone around the room used to die on a clean
        circuit, because every brush against a wall counted forever."""
        for _ in range(8):                      # bump the wall once
            if sim.move(1.5)["collided"]:
                break
        assert sim.retries == 1
        sim.turn(180)                           # turn away and drive off
        sim.move(1.5)
        assert sim.retries == 0, "a successful escape must clear the retry count"
        assert sim.state != "Done"

    def test_three_failures_without_escaping_still_end_the_run(self, sim: Sim):
        """The limit must still bite when the robot is genuinely stuck."""
        for _ in range(12):
            sim.move(1.5)
            if sim.state == "Done":
                break
        assert sim.retries == MAX_RETRIES
        assert sim.outcome.result == "fail_collision"

    def test_a_commanded_move_that_goes_nowhere_counts_as_blocked(self, sim: Sim):
        """The brief's second Blocked trigger, 'no movement for 2 seconds',
        which was never implemented."""
        for _ in range(8):
            if sim.move(1.5)["collided"]:
                break
        before = sim.retries
        sim.move(0.5)                            # nose against the wall: no travel
        assert sim.retries > before


class TestCarriedObjectIsPartOfTheFootprint:
    def test_a_carried_object_cannot_pass_through_a_wall(self, sim: Sim):
        """Regression: a cone held 0.6 m ahead reached y=3.95 with the wall face
        at 3.90. An object sliding through a wall is the most obviously wrong
        thing a judge can see in a clip."""
        import math

        o = next(x for x in sim.scene.objects if x.label == "yellow cone")
        b = math.degrees(math.atan2(-(o.pos[1] - sim.y), o.pos[0] - sim.x))
        sim.turn(((b - sim.heading + 180) % 360) - 180)
        sim.move(math.hypot(o.pos[0] - sim.x, o.pos[1] - sim.y) - 0.42)
        assert sim.pickup()["success"] is True

        for _ in range(14):                      # lap the room while carrying
            sim.turn(30)
            sim.move(1.0)
            pos = sim.data.xpos[sim.obj_bids[sim.holding]][:2] if sim.holding else None
            if pos is not None:
                assert abs(pos[0]) < 3.95 and abs(pos[1]) < 3.95, (
                    f"carried object left the room at {pos}"
                )
