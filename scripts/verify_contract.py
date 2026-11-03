#!/usr/bin/env python3
"""Validate a LIVE backend against contract/api_contract.json.

This is the merge gate. Two people building separately only works if a machine,
not a human, decides whether the pieces fit.

    python scripts/verify_contract.py                        # default localhost:8000
    python scripts/verify_contract.py --base http://localhost:9000
    python scripts/verify_contract.py --strict               # also fail on extra fields

Exit code 0 = safe to merge. Non-zero = do NOT merge, read the failures.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contract" / "api_contract.json"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# Sample request bodies per endpoint id. Kept here so the contract file stays
# a pure description of shapes.
SAMPLES: dict[str, dict[str, Any]] = {
    "create_item": {"title": "Contract probe", "body": "Created by verify_contract.py", "category": "health",
                    "status": "open", "severity": 3, "score": 0.5, "lat": 28.61, "lon": 77.21, "meta": {"probe": True}},
    "patch_item": {"status": "in_progress"},
    "triage": {"text": "Sewage has been leaking into the drinking water line for 5 days and three children are now sick. Nobody responded despite four calls."},
    "summarise": {"text": "The drain outside the stall has overflowed for three weeks. Four calls went unanswered. Children walk through standing water to school."},
    "rag_ask": {"query": "what water problems are reported?", "k": 3},
    "seed": {"n": 30, "reset": True},
    "offline_toggle": {"offline": False},
    # Unique each run: the gate must be re-runnable any number of times.
    "signup": {"email": f"probe-{uuid.uuid4().hex[:10]}@demo.in", "password": "probe12345", "name": "Probe"},
    "login": {"email": "admin@demo.in", "password": "demo1234"},
}


class Result:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.warned: list[tuple[str, str]] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  {GREEN}PASS{RESET}  {name}")

    def fail(self, name: str, why: str) -> None:
        self.failed.append((name, why))
        print(f"  {RED}FAIL{RESET}  {name}\n        {RED}{why}{RESET}")

    def warn(self, name: str, why: str) -> None:
        self.warned.append((name, why))
        print(f"  {YELLOW}WARN{RESET}  {name}\n        {YELLOW}{why}{RESET}")


def call(base: str, method: str, path: str, body: dict | None = None, token: str | None = None):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else None), None
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw), None
        except Exception:
            return e.code, None, f"non-JSON error body: {raw[:160]}"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {str(e)[:160]}"


def type_ok(value: Any, spec: str) -> bool:
    """spec: str | int | num | bool | obj | list[...] , trailing ? means nullable."""
    spec = spec.strip()
    if spec.endswith("?"):
        if value is None:
            return True
        spec = spec[:-1].strip()
    if value is None:
        return False
    if spec.startswith("list"):
        return isinstance(value, list)
    return {
        "str": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "num": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda v: isinstance(v, bool),
        "obj": lambda v: isinstance(v, dict),
    }.get(spec, lambda v: True)(value)


def check_shape(payload: Any, shape: dict[str, str], label: str, res: Result, strict: bool) -> None:
    if not isinstance(payload, dict):
        res.fail(label, f"expected a JSON object, got {type(payload).__name__}")
        return
    problems = []
    for key, spec in shape.items():
        if key.startswith("$"):
            continue
        if key not in payload:
            if not spec.endswith("?"):
                problems.append(f"missing required field '{key}' ({spec})")
            continue
        if not type_ok(payload[key], spec):
            problems.append(f"field '{key}' expected {spec}, got {type(payload[key]).__name__} = {payload[key]!r:.60}")
    if strict:
        extra = [k for k in payload if k not in shape]
        if extra:
            problems.append(f"unexpected extra fields: {extra}")
    if problems:
        res.fail(label, "; ".join(problems))
    else:
        res.ok(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--strict", action="store_true", help="also fail on fields not in the contract")
    ap.add_argument("--contract", default=str(CONTRACT))
    args = ap.parse_args()

    spec = json.loads(Path(args.contract).read_text())
    models = spec.get("models", {})
    res = Result()

    print(f"\n{DIM}contract {spec['version']} -> {args.base}{RESET}\n")
    print("=== REACHABILITY ===")
    status, _, err = call(args.base, "GET", "/health")
    if err or status != 200:
        print(f"  {RED}FAIL{RESET}  server not reachable at {args.base}: {err or f'HTTP {status}'}")
        print(f"\n{RED}Cannot continue. Start the backend first.{RESET}\n")
        return 2
    print(f"  {GREEN}PASS{RESET}  server reachable\n")

    # Seed first so list/stats endpoints have data to return.
    call(args.base, "POST", "/demo/seed", {"n": 30, "reset": True})

    token = None
    _, login_body, _ = call(args.base, "POST", "/auth/login", SAMPLES["login"])
    if isinstance(login_body, dict):
        token = login_body.get("token")

    # Grab an existing id up front so path-param endpoints are actually tested
    # rather than skipped because POST happens later in the contract order.
    created_id = None
    _, listing, _ = call(args.base, "GET", "/items?limit=1")
    if isinstance(listing, dict) and listing.get("items"):
        created_id = listing["items"][0].get("id")

    print("=== ENDPOINTS ===")
    for ep in spec["endpoints"]:
        eid, method, path = ep["id"], ep["method"], ep["path"]

        if "{id}" in path:
            if created_id is None:
                res.warn(f"{method} {path}", "skipped: no item id available yet")
                continue
            path = path.replace("{id}", str(created_id))

        body = SAMPLES.get(eid) if method in ("POST", "PATCH", "PUT") else None
        label = f"{method} {ep['path']}"
        status, payload, err = call(args.base, method, path, body, token if ep.get("auth") else None)

        if err:
            res.fail(label, err)
            continue

        expected_status = ep.get("status", 200)
        if status != expected_status:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            res.fail(label, f"expected HTTP {expected_status}, got {status}" + (f" -- {detail}" if detail else ""))
            continue

        if eid == "create_item" and isinstance(payload, dict):
            created_id = payload.get("id")

        if "response_is" in ep:
            check_shape(payload, models[ep["response_is"]], f"{label}  [{ep['response_is']}]", res, args.strict)
        elif "response" in ep:
            check_shape(payload, ep["response"], label, res, args.strict)
            # validate the element shape of list[Item] fields
            for key, decl in ep["response"].items():
                if decl == "list[Item]" and isinstance(payload, dict) and payload.get(key):
                    check_shape(payload[key][0], models["Item"], f"{label}  [{key}[0] as Item]", res, args.strict)
        else:
            res.ok(label)

    # ---- behavioural guarantees that shapes alone cannot express ----
    print("\n=== BEHAVIOUR ===")

    # /items/stats must not be swallowed by /items/{id}
    status, payload, _ = call(args.base, "GET", "/items/stats")
    if status == 200 and isinstance(payload, dict) and "by_category" in payload:
        res.ok("GET /items/stats resolves before /items/{id}")
    else:
        res.fail("GET /items/stats resolves before /items/{id}",
                 "route ordering bug: register /items/stats BEFORE /items/{id}")

    # triage must survive the LLM being unreachable
    call(args.base, "POST", "/demo/offline", {"offline": True})
    status, payload, err = call(args.base, "POST", "/ai/triage", SAMPLES["triage"])
    if status == 200 and isinstance(payload, dict) and payload.get("category") and payload.get("severity"):
        res.ok("POST /ai/triage still works with the LLM offline")
    else:
        res.fail("POST /ai/triage still works with the LLM offline",
                 f"returned HTTP {status} {str(payload)[:120]} -- it MUST fall back to a rules engine")
    call(args.base, "POST", "/demo/offline", {"offline": False})

    # seeding must be deterministic
    call(args.base, "POST", "/demo/seed", {"n": 30, "reset": True})
    _, a, _ = call(args.base, "GET", "/items?limit=10&sort=created_at&order=asc")
    call(args.base, "POST", "/demo/seed", {"n": 30, "reset": True})
    _, b, _ = call(args.base, "GET", "/items?limit=10&sort=created_at&order=asc")
    if isinstance(a, dict) and isinstance(b, dict):
        ta = [i.get("title") for i in a.get("items", [])]
        tb = [i.get("title") for i in b.get("items", [])]
        if ta and ta == tb:
            res.ok("POST /demo/seed is deterministic")
        else:
            res.fail("POST /demo/seed is deterministic",
                     "two seeds produced different records; use a fixed random seed so the demo looks identical every run")

    # errors must be JSON, never HTML
    status, payload, err = call(args.base, "GET", "/items/99999999")
    if status in (404, 400) and isinstance(payload, dict) and "detail" in payload:
        res.ok("missing record returns JSON {detail}")
    elif err and "non-JSON" in err:
        res.fail("missing record returns JSON {detail}", "returned an HTML error page; the frontend cannot parse that")
    else:
        res.warn("missing record returns JSON {detail}", f"got HTTP {status} {str(payload)[:90]}")

    # CORS preflight must be permitted
    status, _, err = call(args.base, "OPTIONS", "/items")
    if status in (200, 204, 405):
        res.ok("CORS/OPTIONS does not error")
    else:
        res.warn("CORS/OPTIONS does not error", f"HTTP {status}")

    # ---- verdict ----
    total = len(res.passed) + len(res.failed)
    print(f"\n{'=' * 52}")
    print(f"  {len(res.passed)}/{total} passed"
          + (f", {len(res.failed)} FAILED" if res.failed else "")
          + (f", {len(res.warned)} warnings" if res.warned else ""))
    if res.failed:
        print(f"\n  {RED}DO NOT MERGE.{RESET} Fix these, then re-run:")
        for name, why in res.failed:
            print(f"    - {name}: {why}")
        print()
        return 1
    print(f"\n  {GREEN}CONTRACT SATISFIED - SAFE TO MERGE{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
