"""Single place to ask 'what AI can this deployment actually do right now?'

GET /ai/health hits this. Put it on a slide. Judges respect a system that
knows and reports its own degraded modes.
"""
from __future__ import annotations

from typing import Any

from core.config import settings
from . import llm, rag, speech, tabular, vision

CAPABILITIES = {"llm": llm, "rag": rag, "vision": vision, "tabular": tabular, "speech": speech}


def health() -> dict[str, Any]:
    out: dict[str, Any] = {"offline_mode": settings.OFFLINE, "capabilities": {}}
    for name, mod in CAPABILITIES.items():
        try:
            out["capabilities"][name] = mod.available()
        except Exception as e:
            out["capabilities"][name] = {"ok": False, "error": str(e)[:200]}
    return out


def warmup() -> None:
    """Pay import/weight-load cost at boot, not during the live demo."""
    try:
        vision.load()
    except Exception:
        pass
