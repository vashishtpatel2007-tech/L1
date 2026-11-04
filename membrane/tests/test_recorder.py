"""Phase 2 clip assembly.

A clip outside 10-20 seconds, or below 720p, scores 0 for Phase 2 -- 20 of 100
points -- and the failure is purely arithmetic. The duration solver is therefore
tested across the whole plausible range of run lengths.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.recorder import SAFE_MAX, SAFE_MIN, TARGET_MAX, TARGET_MIN, _fit_frames, write_clip


def _frames(n: int, w: int = 640, h: int = 480) -> list[np.ndarray]:
    return [np.full((h, w, 3), (i * 7) % 255, np.uint8) for i in range(n)]


@pytest.mark.parametrize("n", [60, 120, 180, 289, 400, 575, 800, 1200, 2000])
def test_any_run_length_lands_inside_the_required_window(n: int):
    sel, fps = _fit_frames(_frames(n), 30)
    duration = len(sel) / fps
    assert TARGET_MIN <= duration <= TARGET_MAX, (
        f"{n} frames -> {duration:.2f}s at {fps} fps, outside 10-20s"
    )


@pytest.mark.parametrize("n", [120, 289, 575, 1200])
def test_chosen_duration_keeps_a_safety_margin(n: int):
    sel, fps = _fit_frames(_frames(n), 30)
    duration = len(sel) / fps
    assert SAFE_MIN - 0.5 <= duration <= SAFE_MAX + 0.5


def test_an_empty_run_is_refused_rather_than_writing_a_bad_clip():
    with pytest.raises(ValueError, match="no frames"):
        _fit_frames([], 30)


def test_written_clip_is_720p_and_within_the_window(tmp_path):
    out = tmp_path / "clip.mp4"
    report = write_clip(_frames(289), "Bring the tallest object", out)
    assert report["resolution"] == "1280x720"
    assert report["within_10_20s"] is True
    assert TARGET_MIN <= report["duration_s"] <= TARGET_MAX
    assert report["size_bytes"] > 0
    assert out.exists()


def test_written_clip_is_decodable_at_720p(tmp_path):
    import cv2

    out = tmp_path / "clip.mp4"
    write_clip(_frames(200), "test command", out)
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) >= 720
        ok, frame = cap.read()
        assert ok and frame is not None
    finally:
        cap.release()


def test_a_long_command_is_truncated_not_overflowed(tmp_path):
    """The overlay must stay on screen rather than running off the frame."""
    out = tmp_path / "clip.mp4"
    long_cmd = "Bring the object that is farthest from the marked area " * 4
    report = write_clip(_frames(150), long_cmd, out)
    assert report["within_10_20s"] is True
