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
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
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
def post_move(body: MoveBody) -> dict[str, Any]:
    return worker.call(lambda w: w.sim.move(body.distance))


@app.post("/turn")
def post_turn(body: TurnBody) -> dict[str, Any]:
    return worker.call(lambda w: w.sim.turn(body.degrees))


@app.post("/pickup")
def post_pickup() -> dict[str, Any]:
    return worker.call(lambda w: w.sim.pickup())


@app.post("/drop")
def post_drop() -> dict[str, Any]:
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
def admin_reset(body: ResetBody) -> dict[str, Any]:
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "engine": "mujoco", "sim_thread_alive": worker._thread.is_alive()}


@app.exception_handler(Exception)
async def unhandled(request, exc):   # never hand the agent an HTML error page
    return JSONResponse(status_code=500, content={"detail": str(exc)[:300]})
