"""The scene requirements the brief states outright.

These are not implementation details: a scene that violates them is a scene the
judges' commands cannot be fairly answered on, so they are asserted explicitly.
"""
from __future__ import annotations

from engine.world import FOV_H_DEG, FOV_V_DEG, IMG_H, IMG_W, default_scene


def test_scene_has_four_to_six_primitives():
    assert 4 <= len(default_scene().objects) <= 6


def test_at_least_two_objects_share_a_colour():
    objs = default_scene().objects
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.colour] = counts.get(o.colour, 0) + 1
    assert max(counts.values()) >= 2, f"no shared colour: {counts}"


def test_at_least_two_objects_share_a_shape():
    objs = default_scene().objects
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.kind] = counts.get(o.kind, 0) + 1
    assert max(counts.values()) >= 2, f"no shared shape: {counts}"


def test_all_heights_are_distinct_so_superlatives_resolve():
    """With tied heights, "bring the tallest/shortest" becomes a coin flip
    against whatever the judge meant."""
    heights = [o.height for o in default_scene().objects]
    assert len(set(heights)) == len(heights), f"tied heights: {sorted(heights)}"


def test_no_object_is_the_drop_zone_colour():
    """The drop zone is green; a green object would make "the green square"
    ambiguous for the multi-view stretch goal."""
    assert all(o.colour != "green" for o in default_scene().objects)


def test_camera_matches_the_specified_optics():
    assert (IMG_W, IMG_H) == (640, 480)
    assert FOV_H_DEG == 90.0
    # vertical FOV must be DERIVED from the horizontal spec at 4:3, not guessed
    assert round(FOV_V_DEG, 2) == 73.74
