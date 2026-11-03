"""FastAPI entrypoint.

    uv run uvicorn main:app --reload --port 8000     (from apps/api)

Docs at /docs -- hand that URL to a judge who wants to poke the backend.
"""
from __future__ import annotations

import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ai import registry
from core.config import settings
from core.db import init_db
from core.ws import hub
from routers import ai as ai_router
from routers import auth, demo, items, metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    registry.warmup()   # pay model-load cost at boot, never mid-demo
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.TAGLINE,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STARTED = time.time()


@app.middleware("http")
async def timing(request, call_next):
    """Response-time header. Judges notice a fast API; prove it."""
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled(request, exc):
    """Never show a judge a raw stack trace."""
    return JSONResponse(status_code=500, content={"error": "Internal error", "detail": str(exc)[:300]})


@app.get("/health", tags=["system"])
def health():
    return {
        "ok": True,
        "app": settings.APP_NAME,
        "env": settings.ENV,
        "uptime_s": round(time.time() - STARTED, 1),
        "offline_mode": settings.OFFLINE,
        "ws_clients": hub.count,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        await ws.send_json({"kind": "hello", "data": {"app": settings.APP_NAME}})
        while True:
            await ws.receive_text()   # keepalive / client pings
    except WebSocketDisconnect:
        await hub.disconnect(ws)
    except Exception:
        await hub.disconnect(ws)


for r in (auth.router, items.router, ai_router.router, demo.router, metrics.router):
    app.include_router(r)

# Serve the built frontend from the API in production, so the whole demo is ONE
# process on ONE port behind ONE tunnel URL. Build with: cd apps/web && npm run build
_dist = settings.DATA_DIR.parent / "apps" / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
