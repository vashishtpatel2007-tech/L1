"""JSONL run logging. The brief requires one line per step with a frame path,
and frames saved as PNG beside the log. Judges read these logs for 25 of the
100 points, so they are written to be read by a human.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .firewall import assert_frame_name_is_neutral


class RunLog:
    def __init__(self, run_dir: str | Path, command: str) -> None:
        self.dir = Path(run_dir)
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.jsonl"
        self.command = command
        self.step = 0
        self.path.write_text("")
        self._write({"event": "run_start", "command": command, "ts": time.time()})

    def _write(self, obj: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(obj) + "\n")

    def progress(self, phase: str, step: int) -> None:
        path = self.dir / "progress.json"
        temp = self.dir / "progress.tmp"
        temp.write_text(json.dumps({"phase": phase, "step": step, "ts": time.time()}), encoding="utf-8")
        os.replace(temp, path)

    def save_frame(self, jpg_bytes: bytes) -> str:
        """Frames are named by step index only -- never by what is in them."""
        name = f"step_{self.step:03d}.png"
        assert_frame_name_is_neutral(name)
        out = self.frames_dir / name
        import io

        Image.open(io.BytesIO(jpg_bytes)).save(out, format="PNG")
        return str(out.relative_to(self.dir))

    def log_step(self, tool: str, args: dict[str, Any], result: Any,
                 frame_path: str | None, thinking: str = "",
                 pose: dict[str, Any] | None = None) -> None:
        """`pose` is the robot's own x/y/heading, which /state is explicitly
        allowed to report. It is recorded so the trajectory can be plotted as
        directness evidence; it says nothing about any object."""
        self.step += 1
        self._write({
            "step": self.step,
            "timestamp": round(time.time(), 3),
            "tool": tool,
            "args": args,
            "result": result,
            "frame_path": frame_path,
            "robot_pose": pose,
            "claude_reasoning": thinking[:600],
        })

    def finish(self, outcome: str, extra: dict[str, Any] | None = None) -> None:
        self._write({"event": "run_end", "outcome": outcome, "steps": self.step,
                     "ts": time.time(), **(extra or {})})
