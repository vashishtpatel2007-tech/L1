"""The agent: Claude tool-use loop over the engine's HTTP API.

One Claude call per step. Each step Claude receives the latest camera frame as
an image block, a short status line, and a compacted text history -- then
returns exactly one tool call. Nothing about the scene's contents reaches it
except those pixels (plus the optional depth ray and the collision flag).

Run:
    python -m agent.loop --command "Bring the round red thing"          # Sonnet
    python -m agent.loop --command "..." --model claude-haiku-5-5       # cheap
    python -m agent.loop --command "..." --mock                         # free
"""
from __future__ import annotations

import argparse
import io
import math
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .firewall import (
    LeakageError,
    assert_no_engine_identifiers,
    assert_prompt_is_clean,
    sanitize,
)
from .llm import Claude, image_block
from .runlog import RunLog

ENGINE = "http://127.0.0.1:8080"
RUNS = Path(__file__).resolve().parents[2] / "runs"

# --------------------------------------------------------------- prompt ----
# Deliberately free of every object-describing word. assert_prompt_is_clean()
# enforces this at startup, so a careless edit fails loudly instead of silently
# costing 65 points.
SYSTEM = """You drive a wheeled robot inside a closed rectangular room.

A single plain-language command names one item in the room. Your job: find that
item using the camera alone, drive to it, pick it up, carry it to the marked
delivery patch on the floor, and release it there.

EACH STEP you receive one camera frame and a short status line. Reply with
exactly one tool call.

CAMERA GEOMETRY
- The frame is 640 wide and 480 tall, with a 90 degree horizontal field of view.
- To centre something seen at horizontal pixel u, turn by (u - 320) / 640 * 90
  degrees. Positive turns right.
- The camera sits 0.5 m up and is tilted 10 degrees down, so the floor fills the
  lower frame and items further away appear higher and smaller.

JUDGING DISTANCE FROM THE IMAGE  (your most reliable range cue)
Anything resting on the floor meets the floor at some image row v, counting 0 at
the top and 479 at the bottom. Because the camera is fixed at 0.5 m and tilted
down, that row gives the ground distance directly:

    v = 479  ->  0.47 m        v = 320  ->  1.12 m
    v = 440  ->  0.56 m        v = 290  ->  1.46 m
    v = 400  ->  0.67 m        v = 265  ->  1.94 m
    v = 360  ->  0.85 m        v = 250  ->  2.40 m

So: read off where the BASE of the target sits, convert to metres, and move that
distance minus about 0.4 m. Nearer than row ~430 you are inside grasping range.

Important: if the base is cut off by the bottom edge of the frame, the estimate
saturates and understates the true distance. In that case move forward in small
steps of about 0.25 m rather than grasping. Likewise, if `pickup` reports failure,
you are further away than you think -- advance 0.25 m and try once more rather
than repeating the same call.

SENSORS
- `depth_ahead` is a single ray straight out from the frame centre, in metres, or
  null when nothing is within 10 m. It does NOT tell you what it hit. Because the
  ray leaves the lens 10 degrees below horizontal, it passes OVER short items at
  close range and reports the far wall instead. Trust the base-row estimate above
  over `depth_ahead` whenever the two disagree.
- `collided` true means you touched something and were pushed back 0.1 m.

HOW TO ACT WELL
- You are scored on reaching the target directly. Avoid wandering: do not spin
  repeatedly or take many tiny steps.
- If nothing matching the command is visible, turn in steps of 45 to 60 degrees
  to sweep the room rather than guessing a direction.
- Once the target is identified, centre it, then close distance in decisive
  moves using the base-row estimate. Leave roughly 0.4 m before picking up.
- `pickup` succeeds only within 0.5 m and within 30 degrees of straight ahead.
  If it fails, adjust and approach again.
- After a collision, turn away before moving again. Three collisions end the run.
- While carrying, the held item occupies the lower middle of the frame. The
  delivery patch is a flat marked area on the floor; drive onto it, then release.
- Call `finish` only when you have released the item on the patch, or when you
  are certain you cannot continue.

Think briefly before each tool call: say what you can see and why you chose the
action. That reasoning is recorded."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "turn",
        "description": "Rotate in place. Positive degrees turn right, negative turn left.",
        "input_schema": {
            "type": "object",
            "properties": {"degrees": {"type": "number",
                                       "description": "Degrees to rotate, -180 to 180."}},
            "required": ["degrees"],
        },
    },
    {
        "name": "move",
        "description": "Drive straight. Positive metres go forward, negative reverse. "
                       "Stops early if something is struck.",
        "input_schema": {
            "type": "object",
            "properties": {"distance": {"type": "number",
                                        "description": "Metres to travel, -3 to 5."}},
            "required": ["distance"],
        },
    },
    {
        "name": "pickup",
        "description": "Attempt to grasp whatever is directly ahead and within reach. "
                       "Reports only whether it succeeded.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "drop",
        "description": "Release what is currently held and report whether it came to rest "
                       "on the delivery patch.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finish",
        "description": "End the run and state the outcome.",
        "input_schema": {
            "type": "object",
            "properties": {"outcome": {"type": "string",
                                       "description": "What happened, in one sentence."}},
            "required": ["outcome"],
        },
    },
]


# ---------------------------------------------------------------- engine ----
def _get(path: str) -> Any:
    with urllib.request.urlopen(f"{ENGINE}{path}", timeout=30) as r:
        ct = r.headers.get("content-type", "")
        data = r.read()
        return data if ct.startswith("image/") else json.loads(data.decode())


def _post(path: str, body: dict[str, Any] | None = None) -> Any:
    req = urllib.request.Request(
        f"{ENGINE}{path}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------ mock policy ---
def _blobs(jpg: bytes) -> dict[str, Any]:
    """Connected-component colour analysis over the same pixels Claude sees.

    Used only by --mock so the whole loop can be exercised for $0. A single
    global mask is not enough: it merges every coloured item in view into one
    meaningless centroid, so components are labelled individually and the
    largest is chosen.
    """
    import cv2

    img = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    a = img.astype("int16")
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = mx - mn
    # Scene objects render at saturation 186-216; sky gradient and JPEG ringing
    # sit far below. A low threshold made this chase the sky.
    colourful = (sat > 130) & (mx > 90)
    greenish = colourful & (g > r + 40) & (g > b + 40)

    out: dict[str, Any] = {}
    for key, mask in (("target", colourful & ~greenish), ("zone", greenish)):
        m = (mask.astype("uint8")) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), "uint8"))
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        best, best_area = None, 0
        for i in range(1, n):                      # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 60 or area <= best_area:
                continue
            best_area = area
            best = {
                "u": int(cents[i][0]),
                "v": int(cents[i][1]),
                "px": area,
                "bottom": int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1),
            }
        out[key] = best
    return out


def _range_from_row(v: int) -> float:
    """Ground distance to a floor-resting item whose base is at image row v.
    Pure camera geometry: 0.5 m camera height, 10 deg down-tilt, 73.74 deg vFOV."""
    th = math.degrees(math.atan((v - 240) / 240 * math.tan(math.radians(73.74 / 2))))
    elev = 10.0 + th
    return 0.5 / math.tan(math.radians(elev)) if elev > 0.5 else 99.0


def _mock_decide(jpg: bytes, st: dict[str, Any], depth: float | None,
                 memory: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    bl = _blobs(jpg)
    holding = st["holding"]
    if holding:
        memory["grasp_fails"] = 0
    want = bl["zone"] if holding else bl["target"]
    label = "delivery patch" if holding else "candidate item"
    last = memory.get("last_tool")

    if want is None:
        memory["sweeps"] = memory.get("sweeps", 0) + 1
        if memory["sweeps"] > 10:
            return "finish", {"outcome": "could not locate the target"}, "swept the room without a match"
        return "turn", {"degrees": 50}, f"no {label} in view; sweeping right"

    memory["sweeps"] = 0
    off = (want["u"] - 320) / 640 * 90
    # Anti-oscillation: never turn twice in a row while a target is visible.
    # A greedy re-centre every frame makes the robot rock between two items and
    # never close distance -- and wandering costs directness points.
    if abs(off) > 12 and last != "turn":
        return "turn", {"degrees": round(off, 1)}, f"{label} at u={want['u']}; centring"

    if holding:
        if want["bottom"] > 455 and want["px"] > 12000:
            return "drop", {}, "patch fills the lower frame; releasing"
        return "move", {"distance": 0.6}, "patch centred; advancing"

    # Range from the base row -- the depth ray overshoots short items up close.
    # A base at the very bottom edge is CLIPPED, so the estimate saturates at
    # 0.47 m and cannot be trusted; nudge forward instead of grasping thin air.
    clipped = want["bottom"] >= 478
    rng = _range_from_row(want["bottom"])
    if memory.get("grasp_fails", 0) >= 4:
        return "finish", {"outcome": "could not close the final distance"}, "repeated grasp failures"
    if clipped:
        # Alternate nudge and grasp: the range estimate is unusable here, so let
        # the engine adjudicate rather than guessing.
        memory["grasp_fails"] = memory.get("grasp_fails", 0) + 1
        if last == "move":
            return "pickup", {}, "base clipped; range unusable, letting the engine adjudicate"
        return "move", {"distance": 0.22}, "base row clipped at frame edge; nudging closer"
    if rng <= 0.52:
        memory["grasp_fails"] = memory.get("grasp_fails", 0) + 1
        return "pickup", {}, f"base row {want['bottom']} ~= {rng:.2f} m; grasping"
    memory["grasp_fails"] = 0
    step = max(0.3, min(1.2, rng - 0.42))
    return "move", {"distance": round(step, 2)}, f"{label} at ~{rng:.2f} m; closing"


# ------------------------------------------------------------------ loop ----
def run(command: str, model: str, mock: bool, max_steps: int,
        obstacles: bool, run_id: str | None = None) -> dict[str, Any]:
    assert_prompt_is_clean(SYSTEM, TOOLS)          # startup leakage gate

    run_id = run_id or time.strftime("%H%M%S")
    run_dir = RUNS / f"run_{run_id}"
    log = RunLog(run_dir, command)

    _post("/admin/reset", {"command": command, "obstacles": obstacles, "record": True})
    claude = Claude(model=model, mock=mock)
    memory: dict[str, Any] = {}
    history: list[str] = []
    finished_reason = "max_steps"

    for step in range(1, max_steps + 1):
        jpg: bytes = _get("/look")
        frame_rel = log.save_frame(jpg)
        st = sanitize("state", _get("/state"))
        depth = sanitize("depth", _get("/depth")).get("distance")

        status = (
            f"step {step} of {max_steps} | heading {st['heading']} deg | "
            f"holding {str(st['holding']).lower()} | depth_ahead {depth} | "
            f"collisions {st['retry_count']}"
        )

        if mock:
            tool, args, thinking = _mock_decide(jpg, st, depth, memory)
            memory["last_tool"] = tool
        else:
            # Only the newest frame is an image; everything older is one text line.
            user_content: list[dict[str, Any]] = [image_block(jpg)]
            hist = "\n".join(history[-14:])
            user_content.append({"type": "text", "text":
                f"COMMAND: {command}\n{status}\n\nPREVIOUS STEPS:\n{hist or '(none)'}"})
            messages = [{"role": "user", "content": user_content}]
            assert_no_engine_identifiers(
                [c for c in user_content if c.get("type") == "text"], f"step {step} user text")
            out = claude.call(SYSTEM, TOOLS, messages)
            if not out["tool_name"]:
                history.append(f"{step}: model returned no tool call")
                continue
            tool, args, thinking = out["tool_name"], out["tool_input"] or {}, out["text"]

        # ---- execute ----
        if tool == "finish":
            log.log_step(tool, args, {"accepted": True}, frame_rel, thinking)
            finished_reason = "model_finished"
            break
        if tool == "turn":
            result = sanitize("turn", _post("/turn", {"degrees": float(args.get("degrees", 0))}))
        elif tool == "move":
            result = sanitize("move", _post("/move", {"distance": float(args.get("distance", 0))}))
        elif tool == "pickup":
            result = sanitize("pickup", _post("/pickup"))
        elif tool == "drop":
            result = sanitize("drop", _post("/drop"))
        else:
            result = {"error": "unknown tool"}

        log.log_step(tool, args, result, frame_rel, thinking)
        history.append(f"{step}: {tool}({json.dumps(args)}) -> {json.dumps(result)}")

        if tool == "drop" and result.get("success"):
            finished_reason = "dropped"
            break
        if result.get("state") == "Done":
            finished_reason = "engine_done"
            break

    outcome = _get("/admin/outcome")
    spend = claude.commit_spend() if not mock else {"total_usd": Claude.total_spend()}
    log.finish(finished_reason, {
        "engine_outcome": outcome,
        "api_calls": claude.usage.calls,
        "input_tokens": claude.usage.input_tokens,
        "output_tokens": claude.usage.output_tokens,
        "cache_read_tokens": claude.usage.cache_read,
        "run_cost_usd": round(claude.usage.cost_usd, 4),
        "cumulative_usd": spend.get("total_usd"),
    })
    return {"run_dir": str(run_dir), "steps": log.step, "reason": finished_reason,
            "engine_outcome": outcome, "cost_usd": round(claude.usage.cost_usd, 4),
            "cumulative_usd": spend.get("total_usd")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", required=True)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--mock", action="store_true", help="free CV policy; no API calls")
    ap.add_argument("--max-steps", type=int, default=35, help="brief: 25 soft, 60 hard")
    ap.add_argument("--obstacles", action="store_true", help="stretch goal: blocked path")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    try:
        res = run(a.command, a.model, a.mock, a.max_steps, a.obstacles, a.run_id)
    except LeakageError as e:
        raise SystemExit(f"LEAKAGE GATE FAILED -- run aborted: {e}")

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
