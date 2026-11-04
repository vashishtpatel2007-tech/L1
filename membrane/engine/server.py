"""Engine HTTP server on localhost:8080.

Exposes exactly the seven endpoints in the brief. The engine knows the whole
scene; these endpoints are the boundary at which object data stops.

  GET  /state    state, x, y, heading, holding, retry_count   -- no object data
  GET  /look     JPEG 640x480 from the robot camera
  GET  /depth    single-ray distance ahead, or null            -- unidentified
  POST /move     {distance}
  POST /turn     {degrees}
  POST /pickup   success/failure only                          -- unidentified
  POST /drop     success + in_zone

THREADING: MuJoCo's offscreen renderer owns an OpenGL context that is bound to
the thread which created it. Rendering from any other thread hangs forever --
and FastAPI runs sync endpoints on a worker threadpool, so the naive version
deadlocks on the very first /look. Everything therefore runs on one dedicated
simulation thread and requests are marshalled to it through a queue. This also
serialises the simulation for free, which is what the brief wants: the clock
advances only during /move and /turn.

Admin routes are prefixed /admin and are NEVER called by the agent.
"""
from __future__ import annotations

import io
import json
import queue
import random
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .recorder import write_clip
from .sim import Sim
from .world import DROP_ZONE_HALF, default_scene, random_scene


class SimWorker:
    """Owns the Sim and its GL context on a single thread."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self.command = ""
        self.obstacles = False
        self.clutter = "none"
        self.seed: int | None = None
        self._thread = threading.Thread(target=self._loop, name="sim", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("simulation thread failed to start")

    def _loop(self) -> None:
        self.sim = Sim()                 # created HERE, so the GL context lives here
        self._ready.set()
        while True:
            fn, box = self._jobs.get()
            try:
                box.put(("ok", fn(self)))
            except Exception as e:        # noqa: BLE001 -- surfaced to the caller
                box.put(("err", e))

    def call(self, fn: Callable[["SimWorker"], Any], timeout: float = 120.0) -> Any:
        box: queue.Queue = queue.Queue()
        self._jobs.put((fn, box))
        kind, value = box.get(timeout=timeout)
        if kind == "err":
            raise value
        return value


app = FastAPI(title="Membrane Fetch Engine", version="1.0.0")
worker = SimWorker()


class RunLock:
    """Exclusive control of the robot.

    The brief requires that nobody touches keyboard or mouse during a judged
    run. That is not enforceable by convention: an operator console left open in
    a browser polls and can drive the robot, and an agent run sharing the engine
    with it produces a nonsense trace -- a turn of 2 degrees that moves the
    heading by 176, objects delivered that the agent never touched. It happened.

    While a run holds the lock, every mutating endpoint requires its token.
    """

    def __init__(self) -> None:
        self.token: str | None = None
        self.owner: str = ""
        self.expires: float = 0.0

    def active(self) -> bool:
        return self.token is not None and time.time() < self.expires

    def claim(self, owner: str, ttl_s: float) -> dict[str, Any]:
        if self.active():
            raise HTTPException(409, f"engine is already claimed by {self.owner!r}")
        self.token = uuid.uuid4().hex
        self.owner, self.expires = owner, time.time() + ttl_s
        return {"token": self.token, "owner": owner,
                "expires_in_s": round(ttl_s, 1)}

    def release(self, token: str | None) -> dict[str, Any]:
        if self.active() and token != self.token:
            raise HTTPException(403, "wrong token")
        self.token, self.owner, self.expires = None, "", 0.0
        return {"released": True}

    def check(self, token: str | None) -> None:
        if self.active() and token != self.token:
            raise HTTPException(
                409, f"engine is locked by {self.owner!r}; manual control is disabled "
                     f"while a run is in progress")


lock = RunLock()


def guard(x_run_token: str | None = Header(default=None)) -> None:
    """Dependency on every endpoint that changes the world."""
    lock.check(x_run_token)


class MoveBody(BaseModel):
    distance: float = Field(..., description="metres; negative reverses")


class TurnBody(BaseModel):
    degrees: float = Field(..., description="degrees; positive turns right")


class ResetBody(BaseModel):
    """Omitted fields mean UNCHANGED, not "back to the default scene".

    They used to carry concrete defaults, so any caller that reset positions
    without naming the layout silently rebuilt the fixed demo scene. The oracle
    does exactly that at the start of every run, which wiped a randomised scene
    before it could be driven -- every clutter run failed with "no object
    labelled ...". A reset must never change the world unless asked.
    """
    command: str = ""
    record: bool = True
    randomise: bool = False
    seed: int | None = None          # omit for a different scene every time
    n_objects: int = 5               # the PDF allows 4 to 6
    obstacles: bool | None = None    # None -> keep the current layout
    clutter: str | None = None       # None -> keep the current layout


# ------------------------------------------------------------ agent-facing --
@app.get("/state")
def get_state() -> dict[str, Any]:
    return worker.call(lambda w: w.sim.state_payload())


@app.get("/look")
def get_look() -> Response:
    jpg = worker.call(lambda w: w.sim.look())
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/depth")
def get_depth() -> dict[str, Any]:
    return {"distance": worker.call(lambda w: w.sim.depth())}


@app.post("/move")
def post_move(body: MoveBody, _: None = Depends(guard)) -> dict[str, Any]:
    return worker.call(lambda w: w.sim.move(body.distance))


@app.post("/turn")
def post_turn(body: TurnBody, _: None = Depends(guard)) -> dict[str, Any]:
    return worker.call(lambda w: w.sim.turn(body.degrees))


@app.post("/pickup")
def post_pickup(_: None = Depends(guard)) -> dict[str, Any]:
    return worker.call(lambda w: w.sim.pickup())


@app.post("/drop")
def post_drop(_: None = Depends(guard)) -> dict[str, Any]:
    return worker.call(lambda w: w.sim.drop())


# -------------------------------------------------------------------- admin --
def _reset(w: SimWorker, body: ResetBody) -> dict[str, Any]:
    """Reset state in place when the scene is unchanged; rebuild (closing the
    old GL context first) when the layout actually differs.

    A rebuild is the expensive, fragile path -- a second Renderer without
    closing the first corrupts the context -- so it happens only when the scene
    genuinely changes, never on a plain reset between runs.
    """
    obstacles = w.obstacles if body.obstacles is None else body.obstacles
    clutter = w.clutter if body.clutter is None else body.clutter

    rebuilt = False
    want_new_scene = body.randomise or obstacles != w.obstacles or clutter != w.clutter

    if want_new_scene:
        seed = body.seed if body.seed is not None else random.randrange(1, 10**9)
        scene = (random_scene(seed=seed, n_objects=body.n_objects,
                              clutter=clutter, with_obstacles=obstacles)
                 if body.randomise else default_scene(with_obstacles=obstacles))
        w.sim.close()
        w.sim = Sim(scene=scene, record_video=body.record)
        w.obstacles, w.clutter = obstacles, clutter
        w.seed = seed if body.randomise else None
        rebuilt = True
    else:
        w.sim.record_video = body.record
        w.sim.reset_state()

    w.command = body.command
    return {"ok": True, "obstacles": obstacles, "recording": body.record,
            "model_rebuilt": rebuilt, "randomised": body.randomise,
            "seed": w.seed, "clutter": w.clutter,
            "n_objects": len(w.sim.scene.objects)}


@app.post("/admin/reset")
def admin_reset(body: ResetBody, _: None = Depends(guard)) -> dict[str, Any]:
    return worker.call(lambda w: _reset(w, body))


@app.get("/admin/outcome")
def admin_outcome() -> dict[str, Any]:
    """Engine-side ground truth, for OUR regression testing only. The agent never
    calls this and it never appears in any Claude payload."""
    return worker.call(lambda w: {
        "result": w.sim.outcome.result,
        "dropped_label": w.sim.outcome.dropped_label,
        "state": w.sim.state,
        "retry_count": w.sim.retries,
        "video_frames": len(w.sim.video_frames),
    })


@app.get("/admin/next_action")
def admin_next_action(target: str, _: None = Depends(guard)) -> dict[str, Any]:
    """OPERATOR CONSOLE ONLY. Computes the next oracle action toward `target`
    from ground truth, so the browser console can animate a scripted fetch one
    step at a time.

    This is not part of the agent's world. agent/loop.py never calls it, and
    audit/check_leakage.py fails any run whose log mentions /admin/.
    """
    import math

    def job(w: SimWorker) -> dict[str, Any]:
        sim = w.sim
        obj = next((o for o in sim.scene.objects if o.label == target), None)
        if obj is None:
            return {"done": True, "reason": f"unknown target {target!r}"}

        holding = sim.holding is not None
        tx, ty = (sim.scene.drop_zone if holding else
                  tuple(float(v) for v in sim.data.xpos[sim.obj_bids[obj.name]][:2]))
        standoff = 0.30 if holding else 0.42

        gap = math.hypot(tx - sim.x, ty - sim.y)
        bearing = math.degrees(math.atan2(-(ty - sim.y), tx - sim.x))
        rel = ((bearing - sim.heading + 180) % 360) - 180

        if holding and gap <= standoff + 0.06:
            return {"tool": "drop", "args": {}, "note": "on the patch; releasing"}
        if not holding and gap <= standoff + 0.06:
            return {"tool": "pickup", "args": {}, "note": "in reach; grasping"}
        if abs(rel) > 2.5:
            return {"tool": "turn", "args": {"degrees": round(rel, 2)},
                    "note": f"turning {rel:+.1f} deg to line up"}
        return {"tool": "move", "args": {"distance": round(gap - standoff, 3)},
                "note": f"closing {gap - standoff:.2f} m"}

    return worker.call(job)


@app.get("/admin/targets")
def admin_targets() -> dict[str, Any]:
    """Operator console: the labels a scripted fetch can be pointed at."""
    return worker.call(lambda w: {"targets": [o.label for o in w.sim.scene.objects]})


@app.get("/admin/layout")
def admin_layout() -> dict[str, Any]:
    """OPERATOR ONLY. Scene geometry for drawing the dashboard. Never reaches
    the agent: audit/check_leakage.py fails any run whose log mentions /admin/."""
    def job(w: SimWorker) -> dict[str, Any]:
        return {
            "seed": w.seed, "clutter": w.clutter, "obstacles": bool(w.sim.scene.obstacles),
            "room_half": 4.0, "drop_zone_half": DROP_ZONE_HALF,
            "drop_zone": list(w.sim.scene.drop_zone),
            "robot": [round(w.sim.x, 3), round(w.sim.y, 3), round(w.sim.heading, 1)],
            "objects": [
                {"label": o.label, "kind": o.kind, "colour": o.colour,
                 "height": o.height,
                 "pos": [round(float(v), 3) for v in w.sim.data.xpos[w.sim.obj_bids[o.name]][:2]],
                 "held": w.sim.holding == o.name}
                for o in w.sim.scene.objects
            ],
        }

    return worker.call(job)


@app.get("/admin/truth")
def admin_truth() -> dict[str, Any]:
    """DEV ONLY ground truth: object positions and labels.

    This exists so scripts/oracle_run.py can compute an optimal trajectory and
    give us a directness baseline to score Claude against. It is NOT part of the
    agent's world: agent/loop.py never calls it, and audit/check_leakage.py fails
    the run if this path appears anywhere in a run log.
    """
    def job(w: SimWorker) -> dict[str, Any]:
        return {
            "objects": [
                {"name": o.name, "label": o.label, "kind": o.kind, "colour": o.colour,
                 "pos": [round(float(v), 3) for v in w.sim.data.xpos[w.sim.obj_bids[o.name]][:2]],
                 "height": o.height}
                for o in w.sim.scene.objects
            ],
            "drop_zone": list(w.sim.scene.drop_zone),
            "robot": [w.sim.x, w.sim.y, w.sim.heading],
        }

    return worker.call(job)


@app.post("/admin/clip")
def admin_clip(out: str = "runs/clip.mp4") -> dict[str, Any]:
    def job(w: SimWorker) -> dict[str, Any]:
        if not w.sim.video_frames:
            raise HTTPException(400, "no frames recorded")
        return write_clip(w.sim.video_frames, w.command, out)

    return worker.call(job, timeout=300.0)


@app.get("/admin/overhead")
def admin_overhead() -> Response:
    from PIL import Image

    arr = worker.call(lambda w: w.sim.overhead_frame())
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/", include_in_schema=False)
def console() -> Response:
    """Operator console. For driving and watching the robot during development
    and rehearsal; the agent does not use it."""
    html = (Path(__file__).parent / "console.html").read_text()
    return Response(content=html, media_type="text/html")


# ---- launching the real agent -------------------------------------------
# The brief requires three processes: engine, agent, Claude API. The engine must
# never call the model itself, so this does not run the agent in-process -- it
# SPAWNS `python -m agent.loop` exactly as a terminal would, and reports progress
# by reading the run's JSONL. The agent claims the run lock on its own.
_agent_proc: dict[str, Any] = {"proc": None, "run_id": None, "started": 0.0}


class AgentRunBody(BaseModel):
    command: str
    provider: str = "gemini"
    model: str = ""
    max_steps: int = 60
    rpm: int = 12
    obstacles: bool = False


@app.post("/admin/agent_run")
def admin_agent_run(body: AgentRunBody) -> dict[str, Any]:
    proc = _agent_proc["proc"]
    if proc is not None and proc.poll() is None:
        raise HTTPException(409, f"an agent run is already in progress ({_agent_proc['run_id']})")
    if lock.active():
        raise HTTPException(409, f"engine is locked by {lock.owner!r}")
    if not body.command.strip():
        raise HTTPException(400, "command is empty")

    run_id = f"ui{uuid.uuid4().hex[:12]}"
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-X", "utf8", "-u", "-m", "agent.loop",
           "--command", body.command.strip(),
           "--provider", body.provider,
           "--max-steps", str(body.max_steps),
           "--rpm", str(body.rpm),
           "--run-id", run_id]
    if body.model.strip():
        cmd += ["--model", body.model.strip()]
    if body.obstacles:
        cmd.append("--obstacles")

    # Persist startup/API failures: a pipe was consumed by the first status
    # poll, so refreshing the console erased the only useful error message.
    run_dir = root.parent / "runs" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "agent.stdout.log").open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(cmd, cwd=str(root), stdout=output,
                                stderr=subprocess.STDOUT, text=True)
    _agent_proc.update(proc=proc, run_id=run_id, started=time.time())
    return {"started": True, "run_id": run_id, "command": body.command.strip(),
            "provider": body.provider, "model": body.model or "(provider default)"}


@app.get("/admin/agent_status")
def admin_agent_status(run_id: str = "") -> dict[str, Any]:
    rid = run_id or _agent_proc["run_id"]
    if not rid:
        return {"run_id": None, "running": False, "steps": []}
    proc = _agent_proc["proc"]
    running = proc is not None and proc.poll() is None and rid == _agent_proc["run_id"]

    log = Path(__file__).resolve().parents[2] / "runs" / f"run_{rid}" / "log.jsonl"
    steps, final = [], None
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in d:
                steps.append({"step": d["step"], "tool": d["tool"],
                              "args": {k: v for k, v in (d.get("args") or {}).items()},
                              "result": d.get("result"),
                              "reason": d.get("claude_reasoning", "")})
            elif d.get("event") == "run_end":
                final = d

    tail = ""
    output = log.parent / "agent.stdout.log"
    if not running and output.exists():
        tail = output.read_text(encoding="utf-8", errors="replace")[-2000:]
    returncode = proc.poll() if proc is not None and rid == _agent_proc["run_id"] else None
    progress = {}
    progress_path = log.parent / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"run_id": rid, "running": running, "steps": steps, "final": final,
            "stdout_tail": tail, "returncode": returncode,
            "progress": progress,
            "elapsed_s": round(time.time() - _agent_proc["started"], 1)}


@app.post("/admin/agent_stop")
def admin_agent_stop() -> dict[str, Any]:
    proc = _agent_proc["proc"]
    if proc is None or proc.poll() is not None:
        return {"stopped": False, "reason": "no run in progress"}
    proc.terminate()
    lock.release(lock.token)       # the agent cannot release it once killed
    return {"stopped": True, "run_id": _agent_proc["run_id"]}


@app.get("/admin/providers")
def admin_providers() -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.env import load as load_env
    from agent.vlm import DEFAULT_MODEL, available

    load_env()
    return {"available": available(), "defaults": DEFAULT_MODEL}


@app.post("/admin/claim")
def admin_claim(owner: str = "agent", ttl_s: float = 900.0) -> dict[str, Any]:
    return lock.claim(owner, ttl_s)


@app.post("/admin/release")
def admin_release(x_run_token: str | None = Header(default=None)) -> dict[str, Any]:
    return lock.release(x_run_token)


@app.get("/admin/lock")
def admin_lock() -> dict[str, Any]:
    return {"locked": lock.active(), "owner": lock.owner if lock.active() else None,
            "expires_in_s": round(max(0.0, lock.expires - time.time()), 1) if lock.active() else 0}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "engine": "mujoco", "sim_thread_alive": worker._thread.is_alive()}


@app.exception_handler(Exception)
async def unhandled(request, exc):   # never hand the agent an HTML error page
    return JSONResponse(status_code=500, content={"detail": str(exc)[:300]})
