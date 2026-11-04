"""Minimal .env loader.

Keys arrive on a USB stick or in a chat message at 2am; editing shell profiles
then is a mistake waiting to happen. Drop them in membrane/.env instead.
Values already in the real environment always win, so nothing is overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_FILES = [
    Path(__file__).resolve().parents[1] / ".env",      # membrane/.env
    Path(__file__).resolve().parents[2] / ".env",      # repo root .env
]


def load() -> list[str]:
    """Load any .env files that exist. Returns the names of keys that were set."""
    added: list[str] = []
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v
                added.append(k)
    return added
