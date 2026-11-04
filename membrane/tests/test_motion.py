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
