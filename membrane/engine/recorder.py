"""POV clip assembly for Phase 2.

Phase 2 zero-score conditions are purely procedural, so they are enforced in
code rather than trusted to a tired human at hour 19:

  * MP4, 720p or higher
  * 10-20 seconds  (we target 12-18 for margin)
  * recording starts when state leaves Idle, stops at Done
  * the typed command shown as text for the first 2 seconds
  * NO team names, logos or code on screen

The frames come from the robot camera, never from a screen recording -- a screen
recording would show the IDE and score 0.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TARGET_MIN, TARGET_MAX = 10.0, 20.0
SAFE_MIN, SAFE_MAX = 12.0, 18.0
OUT_W, OUT_H = 1280, 720          # 720p as required
OVERLAY_SECONDS = 2.0


def _fit_frames(frames: list[np.ndarray], fps: int) -> tuple[list[np.ndarray], int]:
    """Choose a frame set and fps whose product lands inside the safe window."""
    n = len(frames)
    if n == 0:
        raise ValueError("no frames captured -- was record_video enabled?")

    # Prefer changing fps; only resample if fps would leave a sane range.
    lo_fps, hi_fps = 12, 40
    for d in (15.0, 14.0, 16.0, 13.0, 17.0, 12.0, 18.0):
        f = round(n / d)
        if lo_fps <= f <= hi_fps:
            return frames, int(f)

    # Too many frames for the window: subsample evenly.
    if n / hi_fps > SAFE_MAX:
        keep = int(SAFE_MAX * hi_fps)
        idx = np.linspace(0, n - 1, keep).astype(int)
        return [frames[i] for i in idx], hi_fps
    # Too few: hold each frame longer by duplicating.
    keep = int(SAFE_MIN * lo_fps)
    idx = np.linspace(0, n - 1, keep).astype(int)
    return [frames[i] for i in idx], lo_fps


def write_clip(frames: list[np.ndarray], command: str, out_path: str | Path) -> dict:
    """Render frames to a compliant MP4 and return a compliance report."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sel, fps = _fit_frames(frames, 30)
    duration = len(sel) / fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))
    if not vw.isOpened():
        raise RuntimeError("cv2 could not open the MP4 writer")

    overlay_frames = int(OVERLAY_SECONDS * fps)
    for i, f in enumerate(sel):
        img = cv2.resize(f, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if i < overlay_frames:
            # Command text only. No team name, no logo, no code.
            txt = command if len(command) <= 70 else command[:67] + "..."
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x, y = max(20, (OUT_W - tw) // 2), 70
            cv2.rectangle(img, (x - 18, y - th - 18), (x + tw + 18, y + 18), (0, 0, 0), -1)
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    return {
        "path": str(out_path),
        "frames": len(sel),
        "fps": fps,
        "duration_s": round(duration, 2),
        "resolution": f"{OUT_W}x{OUT_H}",
        "within_10_20s": TARGET_MIN <= duration <= TARGET_MAX,
        "overlay_seconds": round(min(OVERLAY_SECONDS, duration), 2),
        "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
    }
