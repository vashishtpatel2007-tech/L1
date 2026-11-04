#!/usr/bin/env python3
"""Phase 2 clip compliance gate.

A clip over 20 seconds, under 720p, or showing team names, logos or code scores
0 for Phase 2 -- 20 of 100 points -- and the failure is purely procedural. This
checks the mechanical requirements before upload.

    python -m audit.check_clip ../runs/oracle_sphere.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2


def check(path: Path) -> tuple[bool, list[str], dict]:
    if not path.exists():
        return False, [f"missing file: {path}"], {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, [f"not a readable video: {path}"], {}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok_read, _ = cap.read()
    cap.release()

    dur = n / fps if fps else 0
    facts = {"frames": n, "fps": round(fps, 1), "resolution": f"{w}x{h}",
             "duration_s": round(dur, 2), "size_mb": round(path.stat().st_size / 1e6, 2)}

    fails = []
    if not ok_read:
        fails.append("first frame is not decodable")
    if h < 720:
        fails.append(f"resolution {w}x{h} is below 720p")
    if not (10.0 <= dur <= 20.0):
        fails.append(f"duration {dur:.2f}s is outside the required 10-20s window")
    if path.suffix.lower() != ".mp4":
        fails.append(f"extension {path.suffix} is not .mp4")
    return (not fails), fails, facts


def main() -> int:
    args = sys.argv[1:] or ["../runs"]
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        paths.extend(sorted(p.glob("*.mp4")) if p.is_dir() else [p])
    if not paths:
        print("no .mp4 files found")
        return 2

    bad = 0
    for p in paths:
        ok, fails, facts = check(p)
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {p.name}  {facts}")
        for f in fails:
            print(f"          - {f}")
        bad += 0 if ok else 1
    print()
    if bad:
        print(f"{bad} clip(s) would score 0 for Phase 2. Fix before uploading.")
        return 1
    print("All clips meet the mechanical requirements.")
    print("STILL CHECK BY EYE: no team name, no logo, no code or IDE visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
