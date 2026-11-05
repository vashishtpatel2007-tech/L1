"""Provider-agnostic vision-language backend.

WHY THIS EXISTS: the brief supplies one Claude key per team and judges a Claude
tool-use loop, so Claude is what ships. But the key does not arrive until the
event, and the single biggest unknown is whether a vision model can identify the
target from OUR frames with OUR prompt. That question is answerable with any
competent VLM, for free, right now.

So the loop talks to this interface and the provider is a flag. Develop against
Gemini, ship on Claude, change nothing but `--provider`.

The prompt, the tool set, the firewall and the control loop are identical across
providers -- only the wire format differs. That is the whole point: what gets
validated today is what runs tomorrow.
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
from typing import Any

try:
    import certifi

    _CTX: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:          # pragma: no cover
    _CTX = None

TIMEOUT = 90
RETRY_CODES = {408, 429, 500, 502, 503, 504, 529}

#: Requests per minute to stay under, per provider. Free Gemini tiers are tight
#: (5-15 rpm) and a 429 mid-run is indistinguishable from a broken agent, so the
#: loop paces itself rather than discovering the limit the hard way.
DEFAULT_RPM = {"anthropic": 0, "mock": 0}      # 0 = unpaced

_last_call: dict[str, float] = {}


def _pace(provider: str, rpm: int) -> None:
    """Block until at least 60/rpm seconds have passed since the last call."""
    if rpm <= 0:
        return
    gap = 60.0 / rpm
    prev = _last_call.get(provider, 0.0)
    wait = gap - (time.time() - prev)
    if wait > 0:
        time.sleep(wait)
    _last_call[provider] = time.time()


@dataclass
class Decision:
    """One step's output, identical in shape whichever provider produced it."""
    tool: str | None
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    error: str | None = None


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
        **headers,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=req_headers)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=_CTX) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            last = f"HTTP {e.code}: {detail}"
            if e.code in RETRY_CODES and attempt < 3:
                # 429 means a per-minute quota, so wait on that timescale; the
                # usual 1/2/4s backoff just burns the remaining attempts.
                time.sleep((14, 22, 32)[attempt] if e.code == 429 else 2 ** attempt)
                continue
            raise RuntimeError(last) from e
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last) from e
    raise RuntimeError(last or "request failed")


# ------------------------------------------------------------------- Mock --
class MockVLM:
    """Pure local physics, geometry, and computer vision mock. Zero external API calls."""

    name = "mock"
    env_key = ""
    prices = {}

    def __init__(self, model: str = "geometry", max_tokens: int = 700,
                 rpm: int | None = None) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.rpm = 0
        self.key = "local_math"

    def decide(self, system: str, tools: list[dict], image_jpeg: bytes, text: str) -> Decision:
        return Decision(tool=None, provider=self.name, model=self.model)


# ----------------------------------------------------------------- Claude --
class AnthropicVLM:
    """What ships. Organizers supply a Sonnet 5.5 key with vision enabled."""

    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    # (input, output) USD per million tokens, for budget tracking only
    prices = {"claude-sonnet-5-5": (3.0, 15.0), "claude-haiku-5-5": (1.0, 5.0)}

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 700,
                 rpm: int | None = None) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.rpm = DEFAULT_RPM.get(self.name, 0) if rpm is None else rpm
        self.key = os.environ.get(self.env_key, "")

    def decide(self, system: str, tools: list[dict], image_jpeg: bytes, text: str) -> Decision:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Cached: the preamble is byte-identical on every step of every run.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools,
            "tool_choice": {"type": "any"},     # exactly one tool call per step
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(image_jpeg).decode()}},
                {"type": "text", "text": text},
            ]}],
        }
        _pace(self.name, self.rpm)
        out = _post("https://api.anthropic.com/v1/messages",
                    {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body)
        d = Decision(tool=None, provider=self.name, model=self.model)
        for block in out.get("content", []):
            if block.get("type") == "text":
                d.text += block.get("text", "")
            elif block.get("type") == "tool_use":
                d.tool, d.args = block["name"], block.get("input") or {}
        u = out.get("usage", {})
        d.usage = {"input": int(u.get("input_tokens", 0)),
                   "output": int(u.get("output_tokens", 0)),
                   "cache_read": int(u.get("cache_read_input_tokens", 0))}
        return d


# ----------------------------------------------------------------- Gemini --





class FallbackVLM:
    """Tries several models in order, moving on when one is unavailable.

    Newer Gemini models return 503 under load and free tiers hand out 429s per
    model, so a single-model client stalls a run on a condition that has nothing
    to do with the agent. A 404 means the model is gone, so it is demoted for
    the rest of the run. A 429/503 is per-minute quota: try the next model this
    step, but keep the first one eligible next step or the chain dies after one
    burst and later steps report "all models unavailable: None".
    """

    def __init__(self, backends: list) -> None:
        if not backends:
            raise ValueError("no backends")
        self.backends = backends
        self.dead: set[str] = set()

    @property
    def active(self):
        for b in self.backends:
            if b.model not in self.dead:
                return b
        return self.backends[0]

    name = property(lambda self: self.active.name)
    model = property(lambda self: self.active.model)
    prices = property(lambda self: self.active.prices)

    def decide(self, system, tools, image_jpeg, text) -> Decision:
        last = None
        for b in self.backends:
            if b.model in self.dead:
                continue
            try:
                d = b.decide(system, tools, image_jpeg, text)
            except RuntimeError as e:
                last = str(e)
                low = last.lower()
                if "404" in last or "no longer available" in low:
                    self.dead.add(b.model)
                    continue
                if any(c in last for c in ("503", "429")):
                    continue
                raise
            if d.error:
                last = d.error
                low = last.lower()
                if "404" in last or "no longer available" in low:
                    self.dead.add(b.model)
                    continue
            return d
        return Decision(tool=None, error=f"all models unavailable: {(last or 'no remaining models')[:200]}")


#: The brief names Claude as the reasoning model, so Claude is what ships.
#: Earlier revisions carried Gemini, Groq, Qwen, OpenRouter, GitHub Models,
#: Mistral and a local Ollama backend as stand-ins while no Anthropic key was
#: available. Each had its own auth, tool-schema and rate-limit quirks, and
#: keeping seven half-exercised paths alive is seven ways for the real one to
#: break unnoticed. `mock` remains: it is the zero-API path that drives the
#: offline validation sweeps, and it calls nothing.
PROVIDERS = {"mock": MockVLM, "anthropic": AnthropicVLM}
#: A comma-separated preference order, tried left to right. Sonnet is the
#: default because it reads the scene accurately at a fraction of Opus's cost
#: per step, and a fetch is thirty-odd steps.
DEFAULT_MODEL = {
    "mock": "geometry",
    "anthropic": "claude-sonnet-5-5,claude-opus-5-5,claude-haiku-5-5",
}


def build(provider: str, model: str | None = None, max_tokens: int = 700,
          rpm: int | None = None):
    """`model` may be a comma-separated preference order, tried left to right."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    cls = PROVIDERS[provider]
    names = [m.strip() for m in (model or DEFAULT_MODEL[provider]).split(",") if m.strip()]
    backends = [cls(n, max_tokens, rpm) for n in names]
    if not backends[0].key:
        raise RuntimeError(
            f"{cls.env_key} is not set. Export it, or put it in .env and re-launch the engine.")
    return backends[0] if len(backends) == 1 else FallbackVLM(backends)


def cost_usd(vlm, usage: dict[str, int]) -> float:
    pin, pout = vlm.prices.get(vlm.model, (3.0, 15.0))
    return (usage.get("input", 0) / 1e6) * pin + (usage.get("output", 0) / 1e6) * pout



def available() -> dict[str, bool]:
    """Which providers are usable right now, without revealing any key.

    `local` is reported available only when the daemon actually answers, since
    an installed-but-stopped Ollama is not a usable provider.
    """
    # Load .env defensively: a caller that has not done so would be told a key
    # is missing when it is sitting in the file, and the console would grey out
    # a provider that works perfectly well.
    try:
        from .env import load as _load_env

        _load_env()
    except Exception:
        pass

    # `mock` needs no key: it is the offline path and calls nothing.
    return {name: True if name == "mock" else bool(os.environ.get(cls.env_key))
            for name, cls in PROVIDERS.items()}
