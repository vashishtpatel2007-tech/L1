# Membrane Fetch — engine + Claude agent

A typed command goes in; the robot locates the item **from camera pixels alone**,
drives to it, picks it up, carries it to the marked drop zone and releases it.

```bash
# 1. engine (owns scene, physics, camera) on localhost:8080
cd membrane && uvicorn engine.server:app --port 8080

# 2. agent
python -m agent.loop --command "Bring the round red thing"              # Sonnet
python -m agent.loop --command "Bring the tallest object" --mock        # free, no API
python scripts/oracle_run.py --target "red sphere"                      # optimal baseline

# 3. gates
bash scripts/preflight.sh
python -m audit.check_leakage ../runs
python -m audit.check_clip ../runs
```

---

## How Claude finds the object from the image

*(Submission requirement — one paragraph.)*

> Claude never receives any description of the scene. Each step it is given one
> `GET /look` frame as an image content block — JPEG 640×480, 90° horizontal
> field of view, from a camera fixed 0.5 m above the floor and tilted 10° down —
> together with the typed command, the robot's own heading and holding flag, an
> optional single-ray depth value, and one-line text summaries of previous steps.
> From those pixels it decides which item in view matches the command's
> description, converts the item's horizontal pixel position into a turn with
> `degrees = (u − 320) / 640 × 90`, and estimates range from the image row where
> the item meets the floor, since the fixed camera height and tilt make that row
> a direct function of ground distance. It then issues exactly one `turn`,
> `move`, `pickup` or `drop` call, observes the next frame, and repeats. The
> engine supplies no object names, identifiers, colours, shapes, positions,
> counts or drop-zone coordinates through any channel, so every judgement about
> *what* is in the room and *which* item to fetch is made from the image.

---

## Architecture

```
engine/            the game engine -- owns scene, physics, camera, state machine
  world.py         scene + MuJoCo XML (room, 5 primitives, drop zone, robot)
  sim.py           motion, collision, pickup/drop, camera, depth ray
  server.py        the 7 HTTP endpoints at :8080
  recorder.py      POV clip -> compliant 720p MP4
agent/             the Python agent -- holds the command, runs the tool-use loop
  firewall.py      the object-data boundary  <-- read this first
  llm.py           Claude client: caching, retries, spend tracking, Haiku/mock
  loop.py          one Claude call per step; tools: turn/move/pickup/drop/finish
  runlog.py        JSONL + PNG frames, named by step index only
audit/             check_leakage.py, check_clip.py
scripts/           oracle_run.py, preflight.sh
```

Three processes as the brief requires: **engine** (this server), **agent**
(`agent/loop.py`), **Claude API**. The engine advances its clock only inside
`/move` and `/turn`, so the scene is frozen while Claude thinks.

## The object-data boundary

Leakage scores **0 on both Phase 1 components — 65 of 100 points**, so it is
enforced in three independent layers rather than by discipline:

1. **Structural** — every tool result is rebuilt from a per-tool key allowlist
   (`firewall.TOOL_RESULT_ALLOWLIST`). An engine field cannot reach Claude just
   because someone added it to a response.
2. **Static** — the system prompt and tool schemas are scanned at startup for
   object-describing vocabulary. A prompt mentioning a colour or shape aborts
   the run before any API call is made.
3. **Post-hoc** — `audit/check_leakage.py` re-reads the finished logs the way a
   judge will, and fails on engine identifiers, non-allowlisted result keys,
   descriptive frame filenames, or any `/admin/` call during a run.

Colour and shape words are deliberately *not* banned in the command or in
Claude's own reasoning: it inferred those from pixels, which is the whole point.

`GET /admin/truth` exposes ground truth for `scripts/oracle_run.py` only. The
agent never calls it, and the audit fails any run whose log mentions `/admin/`.

## Engine conventions (as specified)

| | |
|---|---|
| Heading | 0 = +x axis, positive clockwise from above |
| `/look` | JPEG 640×480, 90° hFOV, camera 0.5 m up, tilted 10° down |
| `/depth` | single ray from frame centre, metres, max 10, else null |
| `/move` | 1 m/s, negative reverses, stops on contact |
| `/turn` | 90°/s, positive turns right |
| `/pickup` | reach 0.5 m, within ±30° of heading |
| Collision | stop, back off 0.1 m, `collided=true`, enter Blocked; fail after 3 |
| Log | one JSONL line per step with a frame path; frames saved as PNG |

## Non-obvious engineering notes

These each cost real debugging time and are easy to hit again:

- **MuJoCo's renderer is thread-affine.** FastAPI runs sync endpoints on a
  worker threadpool, so a naive `/look` deadlocks forever on the first call.
  Everything runs on one dedicated simulation thread (`engine/server.SimWorker`).
- **Building a second Renderer without closing the first corrupts the GL
  context** and every later frame renders as noise. Resets reuse the model via
  `Sim.reset_state()`; a rebuild calls `Sim.close()` first.
- **MuJoCo filters contacts between two DOF-less bodies.** The robot is a mocap
  body and walls are static, so wall collisions never fire through the contact
  list — static geometry is tested explicitly with `mj_geomDistance`.
- **The depth ray flies over short items at close range** because it leaves the
  lens 10° below horizontal, reporting the far wall instead. The base-row range
  estimate is the reliable cue; the prompt says so explicitly.
- **A carried object engulfs the camera if held too close.** `CARRY_FWD` is
  tuned so the held item covers ~11% of the frame with the horizon clear.
- **`distance_travelled` reports actual signed displacement**, not the requested
  distance — reporting the request lets the agent reason from a lie.

## Budget

Target $40, mentor alert at $30. `runs/spend.json` accumulates real spend from
the API's own usage numbers. Prompt caching is on, only the latest frame is sent
as an image, and earlier steps collapse to one text line each. Develop with
`--mock` (free) and `--model claude-haiku-5-5` (cheap); switch to Sonnet only to
rehearse and to run the hidden commands.

## Stretch goals

- **Claude plans the whole path with `move`/`turn` only.** There is no `go_to`
  endpoint and never was; all navigation is composed from primitives.
- **Obstacles** — `--obstacles` adds a barrier blocking the straight path, with
  collisions and recovery visible in the clip.
