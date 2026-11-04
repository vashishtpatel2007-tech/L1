"""Path planning and approach geometry.

These guard measured behaviour, not opinions: each test names the number it
protects and what regressed when it was absent.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import planner  # noqa: E402


def obj(label, x, y, kind="sphere", height=0.32):
    return {"label": label, "pos": [x, y], "kind": kind, "height": height}


class TestClearPaths:
    def test_an_unobstructed_route_is_a_single_hop(self):
        assert planner.plan((0, 0), (2, 0), [obj("a", 0, 3)]) == [(2, 0)]

    def test_an_obstacle_on_the_line_produces_a_detour(self):
        route = planner.plan((-2, 0), (2, 0), [obj("mid", 0, 0)])
        assert len(route) > 1, "drove straight through the obstacle"
        assert route[-1] == (2, 0)

    def test_every_leg_of_a_detour_is_actually_clear(self):
        obstacles = [obj("a", 0, 0), obj("b", 0.8, 0.4), obj("c", -0.6, -0.5)]
        route = planner.plan((-3, -2), (3, 2), obstacles)
        pts = [(-3, -2)] + route
        for a, b in zip(pts, pts[1:]):
            assert not planner.blockers(a, b, obstacles), f"leg {a}->{b} is blocked"


class TestBarriers:
    def test_a_box_barrier_becomes_overlapping_circles(self):
        circles = planner.barrier_circles([(0.0, 1.1, 1.3, 0.12)])
        assert len(circles) >= 10
        xs = sorted(c["pos"][0] for c in circles)
        assert xs[0] <= -1.2 and xs[-1] >= 1.2, "barrier not covered end to end"
        for a, b in zip(xs, xs[1:]):
            assert b - a <= 2 * 0.12, "gap between circles the robot could slip through"

    def test_a_barrier_carries_its_own_radius(self):
        c = planner.barrier_circles([(0.0, 1.1, 1.3, 0.12)])[0]
        assert planner._obj_radius(c) == pytest.approx(0.12)

    def test_a_route_goes_around_a_wall_rather_than_through_it(self):
        """A 2.6 m wall needs a 1.3 m lateral detour. A single perpendicular
        hop gave 0.38 m, which left delivery at 69% and 2.67 collisions per
        attempt; grid A* raised it to 93%."""
        wall = planner.barrier_circles([(0.0, 0.0, 1.3, 0.12)])
        route = planner.plan((0.0, -2.5), (0.0, 2.5), wall)
        pts = [(0.0, -2.5)] + route
        for a, b in zip(pts, pts[1:]):
            assert not planner.blockers(a, b, wall), f"leg {a}->{b} crosses the wall"
        assert max(abs(p[0]) for p in route) > 0.8, "did not detour around the end"


class TestApproachPose:
    def test_the_target_is_the_nearest_object_from_the_chosen_pose(self):
        """pickup takes whatever is closest, so the wrong side of a crowded
        pair silently fetches the wrong item."""
        target, rival = obj("target", 0, 0), obj("rival", 0.52, 0)
        pose = planner.approach_pose(target, [target, rival], 0.40, prefer_from=(-2, 0))
        assert math.dist(pose, (0, 0)) < math.dist(pose, (0.52, 0))

    def test_the_drive_to_the_pose_does_not_cross_the_target(self):
        """Every collision measured in a 150-attempt sweep was an approach
        hitting an object: 222 of them, none against walls."""
        target = obj("target", 0, 0, height=0.42)
        start = (-2.5, 0.0)
        pose = planner.approach_pose(target, [target], 0.40, prefer_from=start)
        gap = planner._dist_point_segment((0, 0), start, pose)
        assert gap >= planner.ROBOT_RADIUS + planner._obj_radius(target) - 0.01

    def test_the_pose_sits_at_the_requested_standoff(self):
        target = obj("t", 1.0, -1.0)
        pose = planner.approach_pose(target, [target], 0.45, prefer_from=(0, 0))
        assert math.dist(pose, (1.0, -1.0)) == pytest.approx(0.45, abs=0.02)

    def test_it_still_returns_a_pose_when_the_target_is_boxed_in(self):
        target = obj("t", 0, 0)
        ring = [obj(f"r{i}", 0.75 * math.cos(a), 0.75 * math.sin(a))
                for i, a in enumerate([j * math.pi / 4 for j in range(8)])]
        pose = planner.approach_pose(target, [target] + ring, 0.40, prefer_from=(2, 2))
        assert pose is not None and len(pose) == 2


class TestGrid:
    def test_the_room_boundary_is_occupied(self):
        grid = planner.build_grid([])
        i, j = planner._to_cell(3.98, 0.0)
        assert grid[i][j], "the robot could plan into a wall"

    def test_a_start_inside_inflation_still_plans(self):
        """A standoff sits close enough to its object to be inside that
        object's inflated radius; refusing to plan there rejects most
        legitimate approaches."""
        obstacles = [obj("a", 0, 0, height=0.42)]
        grid = planner.build_grid(obstacles)
        assert planner.astar(grid, (0.05, 0.05), (2.0, 2.0)) is not None

    def test_smoothing_leaves_far_fewer_waypoints_than_grid_cells(self):
        grid = planner.build_grid([])
        raw = planner.astar(grid, (-3, -3), (3, 3))
        assert raw and len(raw) > 20
        assert len(planner._smooth(grid, raw, (3, 3))) <= 4
