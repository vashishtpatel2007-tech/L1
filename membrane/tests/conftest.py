"""Shared fixtures.

The MuJoCo renderer owns an OpenGL context bound to the thread that created it,
and building a second Renderer in the same process corrupts that context so every
later frame renders as noise. The Sim is therefore SESSION-scoped and each test
calls reset_state(), which restores the run without touching the model or the
context. A function-scoped Sim fixture would poison the whole suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.sim import Sim          # noqa: E402
from engine.world import default_scene  # noqa: E402


@pytest.fixture(scope="session")
def sim() -> Sim:
    s = Sim(scene=default_scene(), record_video=False)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _fresh(sim: Sim):
    """Every test starts from the run's initial state, with no shared drift."""
    sim.reset_state()
    yield


@pytest.fixture(scope="session")
def obstacle_sim() -> Sim:
    """Separate model for the obstacle layout. Built once, closed explicitly."""
    s = Sim(scene=default_scene(with_obstacles=True), record_video=False)
    yield s
    s.close()
