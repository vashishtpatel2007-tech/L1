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
import urllib.error
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
from .env import load as load_env
from .vlm import available as vlm_available
from .vlm import build as build_vlm
from .vlm import cost_usd
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
- NEVER TURN TWICE IN A ROW once your target is visible. If your previous action
  was a turn and you can see the target at all, your next action is a move.
  Turning repeatedly to perfect the alignment is the single most wasteful thing
  you can do, and it is scored against you.
- "Centred" means within about 100 pixels of column 320. That is close enough to
  drive. You re-centre naturally as you get nearer, so do not chase the exact
  middle from far away.
- SWEEPING IS ONLY FOR WHEN NOTHING IS VISIBLE. Turn about 50 degrees at a time
  while the thing you want is absent from the frame. The INSTANT it appears, stop
  sweeping: turn by the angle you compute from its pixel column, or if it is
  already roughly centred, move. Do not continue a fixed sweep past something you
  have just reported seeing.
- If your own reason says you can see it, your action must act on it. Reporting
  the target and then sweeping past it is a contradiction.
- The room deliberately contains items that share an appearance, so a loose
  command can fit more than one of them. Read the command for every attribute it
  gives and use all of them together. If two items still fit equally well, pick
  the one you can reach most directly and COMMIT to it. Switching targets
  part-way is the worst outcome: it wastes steps and costs directness marks.
- After a move the target will have shifted. Re-centre only when it has drifted
  beyond roughly 100 pixels from column 320; otherwise keep closing. Driving on a
  badly stale heading sails past the target, but correcting every few pixels
  never gets there at all.
- Close distance decisively using the base-row estimate: move the estimated range
  minus 0.4 m in one go. Repeated 0.1 m nudges waste the step budget and cost
  directness marks. Only use small steps once you are inside about 0.7 m.
- `pickup` succeeds only within 0.5 m and within 30 degrees of straight ahead.
  If it fails, adjust and approach again.
- After a collision, turn away before moving again. Three collisions end the run.
- While carrying, the held item occupies the lower middle of the frame. The
  delivery patch is a low raised platform on the floor, brightly coloured with a
  pale rim, and it sits somewhere you have probably not looked yet.
- THE CARRY LEG IS WHERE RUNS ARE LOST, in one of two ways. Driving forward
  hoping the platform is that way, or sweeping straight past it after spotting
  it. If it is not in frame, sweep. If it IS in frame, immediately turn by the
  computed angle to centre it and then drive -- never another blind sweep step.
- Never claim in your reason that you can see something you cannot. An honest
  "the platform is not in view, sweeping" is worth far more than a guess, because
  the next decision is made from what you wrote.
- Once the platform fills the lower frame and its near edge is close to the
  bottom of the image, you are on it; release there.
- Call `finish` only when you have released the item on the patch, or when you
  are certain you cannot continue.

Every tool call takes a `reason`. Fill it with what you actually see in THIS
frame -- what is present, roughly where in the image it sits, your range estimate
from its base row -- and why the action follows. Write it before deciding, not
after; it is the record judges read."""

#: Every tool requires a `reason`. Forcing a tool call makes a model emit the
#: call and nothing else -- no reasoning text at all -- which both costs the
#: "image-driven agent" marks (judges read the logs for it) and measurably
#: degrades the decisions, because nothing makes the model look before it acts.
#: Making the observation part of the call's signature fixes both at once.
REASON_FIELD = {
    "reason": {
        "type": "string",
        "description": "One sentence: what you can see in this frame, roughly where it sits "
                       "in the image, and why this action follows from it.",
    }
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "turn",
        "description": (
            "Rotate in place. POSITIVE turns RIGHT, NEGATIVE turns LEFT.\n"
            "If what you want is VISIBLE in the frame, you must not guess a direction: "
            "set degrees = (u - 320) / 640 * 90, where u is its horizontal pixel column. "
            "Column 150 is left of centre, so that gives about -40 (turn LEFT). "
            "Column 500 is right of centre, so that gives about +25 (turn RIGHT). "
            "Only use a fixed sweep of about 50 degrees when the frame contains "
            "nothing you are looking for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "degrees": {"type": "number", "description": "Degrees to rotate, -180 to 180."},
                **REASON_FIELD,
            },
            "required": ["degrees", "reason"],
        },
    },
    {
        "name": "move",
        "description": "Drive straight. Positive metres go forward, negative reverse. "
                       "Stops early if something is struck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "distance": {"type": "number", "description": "Metres to travel, -3 to 5."},
                **REASON_FIELD,
            },
            "required": ["distance", "reason"],
        },
    },
    {
        "name": "pickup",
        "description": "Attempt to grasp whatever is directly ahead and within reach. "
                       "Reports only whether it succeeded.",
        "input_schema": {"type": "object", "properties": {**REASON_FIELD}, "required": ["reason"]},
    },
    {
        "name": "drop",
        "description": "Release what is currently held and report whether it came to rest "
                       "on the delivery patch.",
        "input_schema": {"type": "object", "properties": {**REASON_FIELD}, "required": ["reason"]},
    },
    {
        "name": "finish",
        "description": "End the run and state the outcome.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "description": "What happened, in one sentence."},
                **REASON_FIELD,
            },
            "required": ["outcome", "reason"],
        },
    },
]


# ---------------------------------------------------------------- engine ----
#: Set while a run holds exclusive control of the engine, so an operator console
#: left open in a browser cannot drive the robot mid-run.
_TOKEN: str | None = None


def _get(path: str) -> Any:
    with urllib.request.urlopen(f"{ENGINE}{path}", timeout=30) as r:
        ct = r.headers.get("content-type", "")
        data = r.read()
        return data if ct.startswith("image/") else json.loads(data.decode())


def _post(path: str, body: dict[str, Any] | None = None) -> Any:
    headers = {"Content-Type": "application/json"}
    if _TOKEN:
        headers["X-Run-Token"] = _TOKEN
    req = urllib.request.Request(
        f"{ENGINE}{path}", data=json.dumps(body or {}).encode(),
        headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _claim(owner: str, ttl_s: float) -> None:
    global _TOKEN
    _TOKEN = None
    try:
        _TOKEN = _post(f"/admin/claim?owner={owner}&ttl_s={ttl_s}").get("token")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            "engine is already claimed -- another run or the operator console is "
            "driving it. Stop that first, or POST /admin/release."
        ) from e


def _release() -> None:
    global _TOKEN
    if _TOKEN:
        try:
            _post("/admin/release")
        except Exception:
            pass
        _TOKEN = None


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
def run(command: str, model: str | None, mock: bool, max_steps: int,
        obstacles: bool, run_id: str | None = None,
        provider: str = "anthropic", rpm: int | None = None,
        budget: float | None = None) -> dict[str, Any]:
    load_env()
    assert_prompt_is_clean(SYSTEM, TOOLS)          # startup leakage gate

    run_id = run_id or time.strftime("%H%M%S")
    run_dir = RUNS / f"run_{run_id}"
    log = RunLog(run_dir, command)

    # Exclusive control for the whole run. Without this an open console shares
    # the robot and the trace becomes fiction.
    _claim(f"agent:{run_id}", ttl_s=max(300.0, max_steps * 30.0))
    _post("/admin/reset", {"command": command, "obstacles": obstacles, "record": True})
    vlm = None if mock else build_vlm(provider, model, rpm=rpm)
    spend = {"input": 0, "output": 0, "cache_read": 0, "calls": 0, "usd": 0.0}

    # Hard budget guard. The brief sets a $40 target with a mentor alert at $30
    # and no hard block, so nothing outside this process will stop a runaway
    # loop. Checked BEFORE each call, against cumulative spend across all runs,
    # because the money is gone once the request is sent.
    started_at = _read_spend()
    budget_stop = False
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

        if not mock and budget is not None:
            projected = started_at + spend["usd"]
            if projected >= budget:
                budget_stop = True
                log.finish("budget_exceeded", {
                    "budget_usd": budget, "cumulative_usd": round(projected, 4),
                    "note": "run halted before the next API call",
                })
                break

        if mock:
            tool, args, thinking = _mock_decide(jpg, st, depth, memory)
            memory["last_tool"] = tool
        else:
            # Only the newest frame is an image; everything older is one text line.
            hist = "\n".join(history[-14:])
            user_text = f"COMMAND: {command}\n{status}\n\nPREVIOUS STEPS:\n{hist or '(none)'}"
            assert_no_engine_identifiers(user_text, f"step {step} user text")
            d = vlm.decide(SYSTEM, TOOLS, jpg, user_text)
            spend["calls"] += 1
            for k in ("input", "output", "cache_read"):
                spend[k] += d.usage.get(k, 0)
            spend["usd"] += cost_usd(vlm, d.usage)
            if d.error or not d.tool:
                history.append(f"{step}: model returned no tool call ({d.error or 'empty'})")
                continue
            tool, args = d.tool, d.args or {}
            # The reason lives in the call itself; free-form text is a bonus.
            thinking = (args.pop("reason", "") or d.text or "").strip()

        # ---- execute ----
        if tool == "finish":
            log.log_step(tool, args, {"accepted": True}, frame_rel, thinking, pose=st)
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

        log.log_step(tool, args, result, frame_rel, thinking, pose=st)
        history.append(f"{step}: {tool}({json.dumps(args)}) -> {json.dumps(result)}")

        if tool == "drop" and result.get("success"):
            finished_reason = "dropped"
            break
        if result.get("state") == "Done":
            finished_reason = "engine_done"
            break

    outcome = _get("/admin/outcome")
    _release()
    total = _commit_spend(spend["usd"]) if not mock else _read_spend()
    log.finish(finished_reason, {
        "engine_outcome": outcome,
        "provider": "mock" if mock else vlm.name,
        "model": "mock" if mock else vlm.model,
        "api_calls": spend["calls"],
        "input_tokens": spend["input"],
        "output_tokens": spend["output"],
        "cache_read_tokens": spend["cache_read"],
        "run_cost_usd": round(spend["usd"], 4),
        "cumulative_usd": total,
    })
    res = {"run_dir": str(run_dir), "steps": log.step,
           "reason": "budget_exceeded" if budget_stop else finished_reason,
           "engine_outcome": outcome, "cost_usd": round(spend["usd"], 4),
           "cumulative_usd": total,
           "provider": "mock" if mock else vlm.name,
           "model": "mock" if mock else vlm.model}
    if budget is not None:
        res["budget_usd"] = budget
        res["budget_remaining_usd"] = round(budget - total, 4)
    return res


SPEND_FILE = RUNS / "spend.json"


def _read_spend() -> float:
    try:
        return float(json.loads(SPEND_FILE.read_text()).get("total_usd", 0.0))
    except Exception:
        return 0.0


def _commit_spend(delta: float) -> float:
    total = round(_read_spend() + delta, 4)
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    runs = 0
    try:
        runs = int(json.loads(SPEND_FILE.read_text()).get("runs", 0))
    except Exception:
        pass
    SPEND_FILE.write_text(json.dumps(
        {"total_usd": total, "runs": runs + 1,
         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", required=True)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "gemini", "groq"],
                    help="anthropic ships; gemini/groq are for free validation")
    ap.add_argument("--model", default=None, help="provider default if omitted")
    ap.add_argument("--rpm", type=int, default=None,
                    help="requests per minute ceiling; 0 disables pacing")
    ap.add_argument("--budget", type=float, default=35.0,
                    help="halt before any call that would push CUMULATIVE spend "
                         "past this. Brief targets $40 with an alert at $30.")
    ap.add_argument("--mock", action="store_true", help="free CV policy; no API calls")
    ap.add_argument("--max-steps", type=int, default=35, help="brief: 25 soft, 60 hard")
    ap.add_argument("--obstacles", action="store_true", help="stretch goal: blocked path")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    try:
        res = run(a.command, a.model, a.mock, a.max_steps, a.obstacles, a.run_id,
                  a.provider, a.rpm, a.budget)
    except LeakageError as e:
        raise SystemExit(f"LEAKAGE GATE FAILED -- run aborted: {e}")
    except RuntimeError as e:
        raise SystemExit(f"{e}\n\nproviders available: {vlm_available()}")

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
