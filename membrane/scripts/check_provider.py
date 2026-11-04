#!/usr/bin/env python3
"""Verify a provider end to end before trusting it with a run.

Paste a key into membrane/.env, run this, and know in one call whether it
works -- rather than discovering it mid-demo through a run that silently does
nothing. It makes ONE request with a real camera frame and the real tool
schema, so a pass here means the agent will work.

    python scripts/check_provider.py                 # every configured provider
    python scripts/check_provider.py --provider qwen
    python scripts/check_provider.py --provider qwen --model qwen-vl-plus

Exit 0 when at least one provider answered with a usable tool call.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import vlm                                   # noqa: E402
from agent.env import load as load_env                  # noqa: E402
from agent.loop import SYSTEM, TOOLS                    # noqa: E402

ENGINE = "http://127.0.0.1:8080"

#: Common failures and what to actually do about them. A bare HTTP code sends
#: people hunting; the cause is nearly always one of these five.
GENERAL_HINTS = [
    ("401", "the key was rejected. Check it copied whole, with no stray spaces or quotes."),
    ("403", "the key is valid but not entitled to this model, or the account needs the "
            "service enabled in its console. A revoked key also gives 403."),
    ("429", "out of quota or rate-limited. Wait, or use a different provider."),
    ("insufficient", "the account has no credit or trial quota left."),
]

#: A 404 means completely different things per provider, and a generic hint
#: sends people to the wrong console.
PROVIDER_HINTS = {
    "qwen": [("404", "that model name is not on this endpoint. Try "
                     "--model qwen-vl-plus, or set DASHSCOPE_BASE_URL to "
                     "https://dashscope.aliyuncs.com/compatible-mode/v1 if your "
                     "account is in the mainland region rather than international.")],
    "local": [("404", "the model is not downloaded. Run: ollama pull qwen2.5vl:7b"),
              ("connection", "the Ollama daemon is not running. Run: ollama serve")],
    "gemini": [("404", "that model was retired. The default chain already falls back; "
                       "pass --model to pick another."),
               ("429", "per-model quota. Wait a minute, or pass --model gemini-3.1-flash-lite."),
               ("400", "the image was rejected or the model needs a different payload. "
                       "Retry with the engine running so /look is a real 640x480 JPEG.")],
    "openrouter": [("404", "unknown model slug. Check the exact id on openrouter.ai/models.")],
    "anthropic": [("404", "unknown model id. Use claude-sonnet-5-5.")],
    "groq": [("404", "unknown model id. Check console.groq.com/docs/models.")],
}


def frame() -> bytes:
    """A real camera frame if the engine is up, else a tiny valid JPEG."""
    try:
        with urllib.request.urlopen(f"{ENGINE}/look", timeout=10) as r:
            return r.read()
    except Exception:
        import base64
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
            "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAf/AABEIAAEAAQMBIgACEQEDEQH/xAAfAAABBQEB"
            "AQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/2gAIAQEAAD8A0s8g/9k=")


def explain(err: str, provider: str) -> str:
    low = err.lower()
    for needle, hint in PROVIDER_HINTS.get(provider, []):
        if needle in low:
            return hint
    for needle, hint in GENERAL_HINTS:
        if needle in low:
            return hint
    return "unexpected failure; the provider's own message is above."


def check(provider: str, model: str | None, jpg: bytes) -> bool:
    try:
        client = vlm.build(provider, model, rpm=0)
    except Exception as e:
        print(f"  {provider:11} NOT CONFIGURED  {e}")
        return False

    label = f"{provider}/{getattr(client, 'model', '?')}"
    text = ("COMMAND: Bring the tallest object\n"
            "step 1 of 32 | heading 0.0 deg | holding false | depth_ahead 2.4 | "
            "collisions 0 | consecutive turns 0\n\nPREVIOUS STEPS:\n(none)")
    t0 = time.time()
    try:
        d = client.decide(SYSTEM, TOOLS, jpg, text)
    except Exception as e:
        print(f"  {label:34} FAIL  {str(e)[:150]}")
        print(f"              -> {explain(str(e), provider)}")
        return False
    dt = time.time() - t0

    if d.error:
        print(f"  {label:34} FAIL  {d.error[:150]}")
        print(f"              -> {explain(d.error, provider)}")
        return False
    if not d.tool:
        print(f"  {label:34} FAIL  answered, but returned no tool call")
        print("              -> the model ignored tool_choice; try a different model.")
        return False

    reason = (d.args or {}).get("reason") or d.text or ""
    print(f"  {label:34} OK    {dt:5.1f}s  {d.tool}("
          f"{json.dumps({k: v for k, v in (d.args or {}).items() if k != 'reason'})[:48]})")
    print(f"              tokens in/out {d.usage.get('input', 0)}/{d.usage.get('output', 0)}"
          f"   cost ${vlm.cost_usd(client, d.usage):.4f}")
    if reason:
        print(f"              said: {reason[:110]}")
    return True


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="default: every configured one")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    avail = vlm.available()
    names = [a.provider] if a.provider else [n for n, ok in avail.items() if ok]
    if not names:
        print("No provider is configured. Put a key in membrane/.env, for example:")
        print("    DASHSCOPE_API_KEY=sk-...")
        return 2

    jpg = frame()
    live = len(jpg) > 2000
    print(f"\nchecking with {'a live camera frame' if live else 'a placeholder image '
          '(engine not running)'}, {len(jpg)} bytes\n")

    ok = 0
    for n in names:
        if not a.provider and not avail.get(n):
            continue
        ok += check(n, a.model, jpg)
    print()
    if ok:
        print(f"{ok} provider(s) ready. Run one with:")
        print(f"    python -m agent.loop --command \"Bring the tallest object\" "
              f"--provider {names[0]}")
        return 0
    print("No provider answered. Fix the errors above before running the agent.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
