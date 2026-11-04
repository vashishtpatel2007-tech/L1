"""HTTP contract tests against a LIVE engine.

Deliberately not FastAPI's TestClient: importing engine.server builds a second
MuJoCo Renderer in this process, and a second GL context corrupts the first so
every later frame renders as noise. These tests therefore talk to a running
server over HTTP, which is also how the agent and the judges see it.

    uvicorn engine.server:app --port 8080     # in another terminal
    pytest -m integration
"""
from __future__ import annotations

import httpx
import pytest

BASE = "http://127.0.0.1:8080"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    try:
        c = httpx.Client(base_url=BASE, timeout=120.0)
        c.get("/health").raise_for_status()
    except Exception:
        pytest.skip("engine not running on :8080")
    c.post("/admin/reset", json={"command": "test", "obstacles": False, "record": False})
    yield c
    c.close()


def test_all_seven_endpoints_exist(client):
    assert client.get("/state").status_code == 200
    assert client.get("/look").status_code == 200
    assert client.get("/depth").status_code == 200
    assert client.post("/move", json={"distance": 0.1}).status_code == 200
    assert client.post("/turn", json={"degrees": 5}).status_code == 200
    assert client.post("/pickup").status_code == 200
    assert client.post("/drop").status_code == 200


def test_state_exposes_no_object_data(client):
    d = client.get("/state").json()
    assert set(d) == {"state", "x", "y", "heading", "holding", "retry_count"}
    assert isinstance(d["holding"], bool)


def test_look_serves_a_640x480_jpeg(client):
    r = client.get("/look")
    assert r.headers["content-type"] == "image/jpeg"
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(r.content)).size == (640, 480)


def test_repeated_look_calls_stay_valid(client):
    """Regression: the renderer deadlocked on the first call under FastAPI's
    worker threadpool, and a rebuilt context returned noise thereafter."""
    sizes = [len(client.get("/look").content) for _ in range(12)]
    assert all(s > 5000 for s in sizes), f"a frame came back degenerate: {sizes}"


def test_depth_returns_only_a_distance(client):
    d = client.get("/depth").json()
    assert set(d) == {"distance"}
    assert d["distance"] is None or 0 < d["distance"] <= 10


def test_pickup_never_identifies_anything(client):
    d = client.post("/pickup").json()
    assert set(d) <= {"success", "reason", "state"}


def test_a_missing_route_returns_json_not_html(client):
    r = client.get("/objects")
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]


def test_reset_is_deterministic(client):
    """The demo must look identical on every run."""
    frames = []
    for _ in range(3):
        client.post("/admin/reset", json={"command": "x", "record": False})
        frames.append(client.get("/look").content)
    assert frames[0] == frames[1] == frames[2], "reset did not reproduce the same frame"
