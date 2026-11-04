#!/usr/bin/env python3
"""Turn a run into something a judge can read in thirty seconds.

25 of 100 points come from judges reading the tool-call and image logs. A
directory of PNGs and a JSONL file is technically compliant and practically
unreadable, so this renders:

  contact_sheet.png  every decision frame, captioned with the step, the tool
                     call, the result, and Claude's stated reason
  trajectory.png     the path over the overhead view, which is the directness
                     evidence Phase 1 asks for

    python scripts/visualize.py ../runs/run_reg01
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ENGINE = "http://127.0.0.1:8080"
ROOM_HALF = 4.0


def load(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l.strip()]


def _fit(text: str, max_px: int, scale: float = 0.40) -> str:
    """Truncate to fit a cell. Measured, not guessed -- a character count
    overflows into the neighbouring column at any proportional font."""
    if not text:
        return ""
    while text:
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if w <= max_px:
            return text
        text = text[:-2]
    return ""


def contact_sheet(run_dir: Path, rows: list[dict], cols: int = 4) -> Path:
    steps = [d for d in rows if "step" in d and d.get("frame_path")]
    if not steps:
        raise SystemExit("no frames to render")
    cw, ch, cap = 320, 240, 86
    n = len(steps)
    grid_r = math.ceil(n / cols)
    sheet = np.full((grid_r * (ch + cap), cols * cw, 3), 22, np.uint8)

    for i, d in enumerate(steps):
        img = cv2.imread(str(run_dir / d["frame_path"]))
        if img is None:
            continue
        img = cv2.resize(img, (cw, ch))
        r, c = divmod(i, cols)
        y0, x0 = r * (ch + cap), c * cw
        sheet[y0:y0 + ch, x0:x0 + cw] = img

        args = json.dumps(d.get("args", {}))
        res = d.get("result", {})
        res_txt = ", ".join(f"{k}={v}" for k, v in res.items()) if isinstance(res, dict) else str(res)
        reason = (d.get("claude_reasoning") or "").replace("\n", " ")
        half = max(1, len(reason) // 2)
        cut = reason.rfind(" ", 0, half + 12) if len(reason) > 40 else len(reason)
        lines = [
            f"step {d['step']}  {d['tool']}{args}",
            res_txt,
            reason[:cut],
            reason[cut:].strip(),
        ]
        for j, text in enumerate(lines):
            text = _fit(text, cw - 14)
            if not text:
                continue
            colour = (235, 235, 235) if j == 0 else (150, 190, 255) if j == 1 else (165, 165, 165)
            cv2.putText(sheet, text, (x0 + 6, y0 + ch + 18 + j * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)

    out = run_dir / "contact_sheet.png"
    cv2.imwrite(str(out), sheet)
    return out


def trajectory(run_dir: Path, rows: list[dict]) -> Path | None:
    poses = [d["robot_pose"] for d in rows if d.get("robot_pose")]
    if not poses:
        return None
    try:
        with urllib.request.urlopen(f"{ENGINE}/admin/overhead", timeout=30) as r:
            buf = np.frombuffer(r.read(), np.uint8)
        bg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        bg = np.full((720, 720, 3), 30, np.uint8)
    H, W = bg.shape[:2]

    def to_px(x: float, y: float) -> tuple[int, int]:
        u = int((x + ROOM_HALF) / (2 * ROOM_HALF) * W)
        v = int((ROOM_HALF - y) / (2 * ROOM_HALF) * H)
        return np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)

    pts = [to_px(p["x"], p["y"]) for p in poses]
    for i in range(1, len(pts)):
        cv2.line(bg, pts[i - 1], pts[i], (70, 200, 255), 3, cv2.LINE_AA)
    for i, (p, pose) in enumerate(zip(pts, poses), 1):
        held = pose.get("holding")
        cv2.circle(bg, p, 7, (60, 230, 130) if held else (70, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(bg, str(i), (p[0] + 9, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    path_len = sum(math.dist((poses[i - 1]["x"], poses[i - 1]["y"]),
                             (poses[i]["x"], poses[i]["y"])) for i in range(1, len(poses)))
    straight = math.dist((poses[0]["x"], poses[0]["y"]), (poses[-1]["x"], poses[-1]["y"]))
    cv2.rectangle(bg, (0, 0), (W, 44), (18, 18, 18), -1)
    cv2.putText(bg, f"steps {len(poses)}   path {path_len:.2f} m   net {straight:.2f} m   "
                    f"wander ratio {path_len / max(straight, 1e-6):.2f}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)

    out = run_dir / "trajectory.png"
    cv2.imwrite(str(out), bg)
    return out


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../runs")
    if not (run_dir / "log.jsonl").exists():
        cands = sorted(run_dir.glob("run_*"))
        if not cands:
            raise SystemExit(f"no run found under {run_dir}")
        run_dir = cands[-1]
    rows = load(run_dir)
    cs = contact_sheet(run_dir, rows)
    tj = trajectory(run_dir, rows)
    print(f"  contact sheet -> {cs}")
    print(f"  trajectory    -> {tj}" if tj else "  trajectory    -> (no pose data in this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
