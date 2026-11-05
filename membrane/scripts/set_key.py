#!/usr/bin/env python3
"""Paste the Anthropic key straight into .env, then prove it works.

Typing a hundred-character key by hand does not work. It mixes 0 with O, 1
with I and l, and a single wrong character gives the same "invalid x-api-key"
as no key at all -- with nothing to say which character was wrong. This takes
the key from the clipboard or a paste, cleans it up, tests it against the real
API, and only writes .env if the API accepts it.

    python scripts/set_key.py                 # paste when prompted
    python scripts/set_key.py --clipboard     # take it from the clipboard

The key is read without echoing and is never printed back.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import ssl
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
API = "https://api.anthropic.com/v1/messages"

#: Every Anthropic key prefix seen in the wild. "sk-ant-api03-" is what the
#: documentation shows; "sk-ant-usr-" is what the Console hands out for a
#: personal key and works exactly the same way, as an x-api-key. "sk-ant-oat"
#: is a Claude Code sign-in token and is NOT an API key.
API_PREFIXES = ("sk-ant-api", "sk-ant-usr", "sk-ant-svc", "sk-ant-admin")
NOT_AN_API_KEY = ("sk-ant-oat",)


def clean(raw: str) -> str:
    """Strip everything a paste can pick up and a key never contains.

    Keys arrive wrapped across lines, with quotes, with a leading
    "ANTHROPIC_API_KEY=", and sometimes with a zero-width character from a
    web page. A key is only letters, digits, hyphen and underscore, so
    anything else is packaging.
    """
    raw = raw.strip().strip('"').strip("'")
    if "=" in raw.split("sk-ant")[0]:
        raw = raw.split("=", 1)[1]
    return re.sub(r"[^A-Za-z0-9_-]", "", raw)


def from_clipboard() -> str:
    for cmd in (["pbpaste"], ["xclip", "-selection", "clipboard", "-o"], ["wl-paste"]):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=5).stdout
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    return ""


def check(key: str) -> tuple[bool, str]:
    """Ask the API whether it accepts this key. One tiny call."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    body = {"model": "claude-sonnet-5-5", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "x-api-key": key,
        "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=40, context=ctx):
            return True, "the API accepted it"
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read().decode() or "{}").get("error", {}).get("message", "")
        if e.code == 401:
            return False, ("the API rejected the key itself. Copy it again from "
                           "console.anthropic.com -- a key can only be shown once, "
                           "so if it was not saved you need to create a new one.")
        if e.code == 400 and "credit" in detail.lower():
            # The key is genuine; the account simply has nothing to spend.
            return True, ("the key is valid, but the account has no credit. "
                          "Add credit at console.anthropic.com/settings/billing.")
        if e.code == 404:
            return True, (f"the key works, but this account cannot use that model "
                          f"({detail[:80]}). Set a different one with --model.")
        return False, f"HTTP {e.code}: {detail[:120]}"
    except Exception as e:
        return False, f"could not reach the API: {type(e).__name__}: {e}"


def write(key: str) -> None:
    lines = []
    if ENV.exists():
        lines = [l for l in ENV.read_text().splitlines()
                 if not l.startswith("ANTHROPIC_API_KEY=")]
    lines = [l for l in lines if l.strip()]
    lines.insert(0, "# Claude (Anthropic). The only provider this project uses.")
    lines.insert(1, f"ANTHROPIC_API_KEY={key}")
    ENV.write_text("\n".join(lines) + "\n")
    os.chmod(ENV, stat.S_IRUSR | stat.S_IWUSR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clipboard", action="store_true",
                    help="read the key from the system clipboard")
    a = ap.parse_args()

    raw = from_clipboard() if a.clipboard else getpass.getpass(
        "Paste the Anthropic API key (it will not be shown), then press Enter:\n> ")
    key = clean(raw)

    if not key:
        print("  nothing to read. Copy the key, then run this again.")
        return 1
    print(f"\n  read {len(key)} characters, starting {key[:11]}...")

    if key.startswith(NOT_AN_API_KEY):
        print("  that is a Claude Code sign-in token, not an API key. Create an "
              "API key at console.anthropic.com/settings/keys.")
        return 1
    if not key.startswith(API_PREFIXES):
        print(f"  that does not look like an Anthropic key: it should start with "
              f"one of {', '.join(API_PREFIXES)}.")
        return 1
    if len(key) < 90:
        print("  that is shorter than any Anthropic key, so some of it is "
              "missing. Copy the whole thing, including the end.")
        return 1

    print("  testing it against the API...")
    ok, why = check(key)
    print(f"  {why}")
    if not ok:
        print("\n  .env left unchanged.")
        return 1

    write(key)
    print(f"\n  saved to {ENV} (owner-only, git-ignored).")
    print("  run the agent with:  python -m agent.loop "
          "--command \"Bring the tallest object\" --provider anthropic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
