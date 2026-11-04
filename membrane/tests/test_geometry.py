"""Heading and ranging conventions.

The brief fixes these exactly: heading 0 = +x, positive CLOCKWISE from above,
move at 1 m/s, turn at 90 deg/s. A sign error here is invisible in a screenshot
and ruins every run, so the arithmetic is pinned down.
"""
from __future__ import annotations

import math

import pytest

from engine.sim import Sim


@pytest.mark.parametrize(
    "heading,expected",
    [
        (0.0, (1.0, 0.0)),      # 0 points along +x
        (90.0, (0.0, -1.0)),    # clockwise from above: +x -> -y
        (180.0, (-1.0, 0.0)),
        (270.0, (0.0, 1.0)),
    ],
    ids=["east", "clockwise-quarter", "west", "clockwise-three-quarter"],
)
def test_heading_zero_is_plus_x_and_positive_is_clockwise(sim: Sim, heading, expected):
    sim.heading = heading
    f = sim._forward()
    assert f[0] == pytest.approx(expected[0], abs=1e-9)
    assert f[1] == pytest.approx(expected[1], abs=1e-9)


@pytest.mark.parametrize("distance,heading", [(1.0, 0.0), (2.0, 326.82), (1.5, 45.0), (-1.0, 90.0)])
def test_move_lands_where_trigonometry_says(sim: Sim, distance, heading):
    sim.heading = heading
    sim._apply_pose()
    x0, y0 = sim.x, sim.y
    sim.move(distance)
    h = math.radians(heading)
    assert sim.x == pytest.approx(x0 + distance * math.cos(h), abs=0.02)
    assert sim.y == pytest.approx(y0 - distance * math.sin(h), abs=0.02)


def test_positive_turn_goes_right(sim: Sim):
    sim.turn(90.0)
    assert sim.heading == pytest.approx(90.0, abs=0.5)


def test_heading_wraps_into_zero_to_360(sim: Sim):
    sim.turn(-45.0)
    assert 0.0 <= sim.heading < 360.0
    assert sim.heading == pytest.approx(315.0, abs=0.5)


def test_depth_is_none_or_a_bounded_distance(sim: Sim):
    d = sim.depth()
    assert d is None or (0.0 < d <= 10.0)


def test_depth_ray_overshoots_short_objects_up_close(sim: Sim):
    """Documents a real property of the specified optics: the ray leaves the lens
    10 degrees below horizontal, so at close range it passes OVER a low object
    and reports whatever is behind it. The agent prompt depends on this being
    true, so it is asserted rather than assumed."""
    target = next(o for o in sim.scene.objects if o.label == "red sphere")
    bearing = math.degrees(math.atan2(-(target.pos[1] - sim.y), target.pos[0] - sim.x))
    sim.turn(((bearing - sim.heading + 180) % 360) - 180)
    gap = math.hypot(target.pos[0] - sim.x, target.pos[1] - sim.y)
    sim.move(gap - 0.45)                      # now ~0.45 m away, inside pickup reach
    d = sim.depth()
    assert d is None or d > 0.8, (
        f"expected the tilted ray to miss a {target.height} m object at 0.45 m, got {d}"
    )
