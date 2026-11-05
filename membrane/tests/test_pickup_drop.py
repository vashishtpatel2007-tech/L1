"""Grasping and delivery: the reach envelope and zone adjudication."""
from __future__ import annotations

import math

import pytest

from engine.sim import Sim
from engine.world import PICKUP_CONE_DEG, PICKUP_REACH


def _face(sim: Sim, tx: float, ty: float) -> None:
    b = math.degrees(math.atan2(-(ty - sim.y), tx - sim.x))
    sim.turn(((b - sim.heading + 180) % 360) - 180)


def _approach(sim: Sim, label: str, standoff: float) -> None:
    o = next(x for x in sim.scene.objects if x.label == label)
    _face(sim, *o.pos)
    sim.move(math.hypot(o.pos[0] - sim.x, o.pos[1] - sim.y) - standoff)


def test_pickup_fails_beyond_reach(sim: Sim):
    _approach(sim, "red sphere", PICKUP_REACH + 0.6)
    assert sim.pickup() == {"success": False, "reason": "nothing_in_reach", "state": "GoToObject"}


def test_pickup_succeeds_inside_reach(sim: Sim):
    _approach(sim, "red sphere", 0.42)
    r = sim.pickup()
    assert r["success"] is True
    assert sim.state == "Carry"


def test_pickup_fails_outside_the_heading_cone(sim: Sim):
    """Within reach but facing away: the brief allows +/-30 degrees only."""
    _approach(sim, "red sphere", 0.42)
    sim.turn(PICKUP_CONE_DEG + 35)
    assert sim.pickup()["success"] is False


def test_pickup_result_never_identifies_the_object(sim: Sim):
    _approach(sim, "red sphere", 0.42)
    r = sim.pickup()
    assert set(r) <= {"success", "reason", "state"}
    assert not any(isinstance(v, str) and ("red" in v or "sphere" in v) for v in r.values())


def test_cannot_pick_up_twice(sim: Sim):
    _approach(sim, "red sphere", 0.42)
    sim.pickup()
    assert sim.pickup() == {"success": False, "reason": "already_holding", "state": "Carry"}


def test_drop_without_holding_is_rejected(sim: Sim):
    r = sim.drop()
    assert r["success"] is False and r["in_zone"] is False


@pytest.mark.parametrize(
    "label", ["red sphere", "red cube", "blue cube", "blue sphere", "yellow cone"]
)
def test_every_object_can_be_fetched_into_the_zone(sim: Sim, label: str):
    """The full happy path for each object, which is what the judges demo."""
    sim.reset_state()
    _approach(sim, label, 0.42)
    assert sim.pickup()["success"] is True
    dz = sim.scene.drop_zone
    _face(sim, *dz)
    sim.move(math.hypot(dz[0] - sim.x, dz[1] - sim.y) - 0.3)
    r = sim.drop()
    assert r["in_zone"] is True, f"{label} did not land in the zone"
    assert sim.outcome.result == "pass"
    assert sim.outcome.dropped_label == label


def test_a_carried_object_does_not_blind_the_camera(sim: Sim):
    """Regression: the carried object used to sit so close that it engulfed the
    lens and the frame rendered as an empty room. The horizon must stay clear."""
    import io

    import numpy as np
    from PIL import Image

    _approach(sim, "yellow cone", 0.42)       # the largest object, worst case
    assert sim.pickup()["success"] is True
    img = np.asarray(Image.open(io.BytesIO(sim.look())).convert("RGB")).astype("int16")

    # Saturation, not hardcoded RGB bands: a channel-threshold test silently
    # breaks the moment the lighting is retuned, which says nothing about the
    # property being asserted.
    sat = img.max(axis=2) - img.min(axis=2)
    held = sat > 90
    assert 0.02 < held.mean() < 0.30, f"held object covers {held.mean():.1%} of the frame"
    assert held[:192].mean() < 0.25, "tall held cone leaves less than 75% of upper view clear"
