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
DEFAULT_RPM = {"anthropic": 0, "gemini": 10, "groq": 25}   # 0 = unpaced

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
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers})
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
        self.rpm = DEFAULT_RPM[self.name] if rpm is None else rpm
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
class GeminiVLM:
    """Free tier, vision + function calling. Used to validate the pipeline
    before the Claude key exists; never the submitted configuration."""

    name = "gemini"
    env_key = "GEMINI_API_KEY"
    prices = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.0-flash": (0.10, 0.40)}

    def __init__(self, model: str = "gemini-2.5-flash", max_tokens: int = 700,
                 rpm: int | None = None) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.rpm = DEFAULT_RPM[self.name] if rpm is None else rpm
        self.key = os.environ.get(self.env_key) or os.environ.get("GOOGLE_API_KEY", "")

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        """Anthropic tool schema -> Gemini function declarations.

        Gemini rejects a parameters object with an empty `properties`, so a
        no-argument tool must declare no parameters at all.
        """
        decls = []
        for t in tools:
            fn: dict[str, Any] = {"name": t["name"], "description": t.get("description", "")}
            schema = t.get("input_schema") or {}
            if schema.get("properties"):
                fn["parameters"] = {
                    "type": "object",
                    "properties": {
                        k: {kk: vv for kk, vv in v.items() if kk in ("type", "description")}
                        for k, v in schema["properties"].items()
                    },
                    **({"required": schema["required"]} if schema.get("required") else {}),
                }
            decls.append(fn)
        return [{"function_declarations": decls}]

    def decide(self, system: str, tools: list[dict], image_jpeg: bytes, text: str) -> Decision:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(image_jpeg).decode()}},
                {"text": text},
            ]}],
            "tools": self._tools(tools),
            "tool_config": {"function_calling_config": {"mode": "ANY"}},
            "generationConfig": {"maxOutputTokens": self.max_tokens, "temperature": 0.2},
        }
        _pace(self.name, self.rpm)
        out = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.key}, body)

        d = Decision(tool=None, provider=self.name, model=self.model)
        cands = out.get("candidates") or []
        if not cands:
            d.error = f"no candidates: {str(out)[:200]}"
            return d
        for part in cands[0].get("content", {}).get("parts", []) or []:
            if "text" in part:
                d.text += part["text"]
            if "functionCall" in part:
                fc = part["functionCall"]
                d.tool, d.args = fc.get("name"), dict(fc.get("args") or {})
        m = out.get("usageMetadata", {})
        d.usage = {"input": int(m.get("promptTokenCount", 0)),
                   "output": int(m.get("candidatesTokenCount", 0)),
                   "cache_read": int(m.get("cachedContentTokenCount", 0))}
        return d


# ------------------------------------------------------------------- Groq --
class GroqVLM:
    """OpenAI-compatible vision + tools. A second free fallback."""

    name = "groq"
    env_key = "GROQ_API_KEY"
    prices = {"meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34)}

    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                 max_tokens: int = 700, rpm: int | None = None) -> None:
        self.model, self.max_tokens = model, max_tokens
        self.rpm = DEFAULT_RPM[self.name] if rpm is None else rpm
        self.key = os.environ.get(self.env_key, "")

    def decide(self, system: str, tools: list[dict], image_jpeg: bytes, text: str) -> Decision:
        oa_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
        }} for t in tools]
        b64 = base64.b64encode(image_jpeg).decode()
        body = {
            "model": self.model, "max_tokens": self.max_tokens, "tools": oa_tools,
            "tool_choice": "required",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": text},
                ]},
            ],
        }
        _pace(self.name, self.rpm)
        out = _post("https://api.groq.com/openai/v1/chat/completions",
                    {"Authorization": f"Bearer {self.key}"}, body)
        d = Decision(tool=None, provider=self.name, model=self.model)
        msg = (out.get("choices") or [{}])[0].get("message", {})
        d.text = msg.get("content") or ""
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            d.tool = fn.get("name")
            try:
                d.args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                d.args = {}
        u = out.get("usage", {})
        d.usage = {"input": int(u.get("prompt_tokens", 0)),
                   "output": int(u.get("completion_tokens", 0)), "cache_read": 0}
        return d


class FallbackVLM:
    """Tries several models in order, moving on when one is unavailable.

    Newer Gemini models return 503 under load and free tiers hand out 429s per
    model, so a single-model client stalls a run on a condition that has nothing
    to do with the agent. A model that fails is demoted for the rest of the run
    rather than retried every step.
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
                return b.decide(system, tools, image_jpeg, text)
            except RuntimeError as e:
                last = str(e)
                if any(c in last for c in ("503", "429", "404")):
                    self.dead.add(b.model)      # demote for the rest of the run
                    continue
                raise
        return Decision(tool=None, error=f"all models unavailable: {str(last)[:200]}")


PROVIDERS = {"anthropic": AnthropicVLM, "gemini": GeminiVLM, "groq": GroqVLM}
DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5",
                 "gemini": "gemini-2.5-flash",
                 "groq": "meta-llama/llama-4-scout-17b-16e-instruct"}


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
    """Which providers are usable right now, without revealing any key."""
    return {name: bool(os.environ.get(cls.env_key)
                       or (name == "gemini" and os.environ.get("GOOGLE_API_KEY")))
            for name, cls in PROVIDERS.items()}
