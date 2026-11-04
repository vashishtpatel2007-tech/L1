"""Claude client with budget discipline, plus a zero-cost mock.

Budget rules from the brief, implemented here:
  * one image per step, latest only; earlier steps collapse to one-line text
  * prompt caching on the system prompt and tool schemas
  * per-run and cumulative spend tracking against the $40 target
  * Haiku and mock modes so HTTP plumbing is never debugged on Sonnet's dime
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import certifi

    _CTX: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = None

API_URL = "https://api.anthropic.com/v1/messages"
SPEND_FILE = Path(__file__).resolve().parents[2] / "runs" / "spend.json"

# Published per-million-token prices, used only to track the $40 target.
PRICES = {
    "claude-sonnet-5-5": (3.00, 15.00),
    "claude-haiku-5-5": (1.00, 5.00),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, model: str, u: dict[str, Any]) -> None:
        i = int(u.get("input_tokens", 0))
        o = int(u.get("output_tokens", 0))
        cr = int(u.get("cache_read_input_tokens", 0))
        self.input_tokens += i
        self.output_tokens += o
        self.cache_read += cr
        self.calls += 1
        pin, pout = PRICES.get(model, (3.0, 15.0))
        self.cost_usd += (i / 1e6) * pin + (o / 1e6) * pout + (cr / 1e6) * pin * 0.1


@dataclass
class Claude:
    model: str = "claude-sonnet-5-5"
    mock: bool = False
    max_tokens: int = 700
    usage: Usage = field(default_factory=Usage)
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))

    # ------------------------------------------------------------- spend --
    @staticmethod
    def total_spend() -> float:
        if SPEND_FILE.exists():
            try:
                return float(json.loads(SPEND_FILE.read_text()).get("total_usd", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def commit_spend(self) -> dict[str, Any]:
        SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
        total = self.total_spend() + self.usage.cost_usd
        runs = 0
        if SPEND_FILE.exists():
            try:
                runs = int(json.loads(SPEND_FILE.read_text()).get("runs", 0))
            except Exception:
                runs = 0
        payload = {"total_usd": round(total, 4), "runs": runs + 1,
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        SPEND_FILE.write_text(json.dumps(payload, indent=2))
        return payload

    # -------------------------------------------------------------- call --
    def call(self, system: str, tools: list[dict[str, Any]],
             messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Returns {text, tool_name, tool_input, stop_reason}."""
        if self.mock:
            raise RuntimeError("mock mode must be handled by the caller's policy")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Cache the static preamble; it is identical on every step of every run.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools,
            "tool_choice": {"type": "any"},   # the brief wants exactly one tool call per step
            "messages": messages,
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        last_err = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90, context=_CTX) as r:
                    out = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:300]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_err) from e
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_err) from e

        self.usage.add(self.model, out.get("usage", {}))
        text, tool_name, tool_input, tool_id = "", None, None, None
        for block in out.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_name, tool_input, tool_id = block["name"], block.get("input", {}), block["id"]
        return {"text": text, "tool_name": tool_name, "tool_input": tool_input,
                "tool_id": tool_id, "stop_reason": out.get("stop_reason"),
                "raw_assistant": out.get("content", [])}


def image_block(jpg: bytes) -> dict[str, Any]:
    """The brief requires the frame as an image content block, not a description."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": base64.b64encode(jpg).decode()},
    }
