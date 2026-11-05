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

    # Fixed output cadence. Preserve timing inside the accepted duration window;
    # outside it, resample explicitly (time compression/extension, not FPS drift).
    fps = 30
    duration = min(SAFE_MAX, max(SAFE_MIN, n / fps))
    keep = round(duration * fps)
    idx = np.linspace(0, n - 1, keep).round().astype(int)
    return [frames[i] for i in idx], fps


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
        # Preserve calibrated camera geometry: letterbox 4:3 into 720p.
        scale=min(OUT_W/f.shape[1],OUT_H/f.shape[0])
        width,height=round(f.shape[1]*scale),round(f.shape[0]*scale)
        resized=cv2.resize(f,(width,height),interpolation=cv2.INTER_LINEAR)
        img=np.zeros((OUT_H,OUT_W,3),dtype=np.uint8)
        left,top=(OUT_W-width)//2,(OUT_H-height)//2
        img[top:top+height,left:left+width]=resized
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
