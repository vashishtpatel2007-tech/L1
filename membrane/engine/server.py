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
import queue
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
from .world import default_scene


class SimWorker:
    """Owns the Sim and its GL context on a single thread."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self.command = ""
        self.obstacles = False
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
    command: str = ""
    obstacles: bool = False
    record: bool = True


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
    """Reset state in place when the scene structure is unchanged; only rebuild
    (closing the old GL context first) when the obstacle layout actually differs."""
    rebuilt = False
    if body.obstacles != w.obstacles:
        w.sim.close()
        w.sim = Sim(scene=default_scene(with_obstacles=body.obstacles), record_video=body.record)
        w.obstacles = body.obstacles
        rebuilt = True
    else:
        w.sim.record_video = body.record
        w.sim.reset_state()
    w.command = body.command
    return {"ok": True, "obstacles": body.obstacles, "recording": body.record,
            "model_rebuilt": rebuilt}


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
