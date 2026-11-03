"""Multi-provider LLM with a deterministic offline fallback.

Order: Anthropic -> OpenAI -> Groq -> Gemini -> offline stub.
Never raises. Always returns text. That property is what saves your demo.
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Any

from core.config import settings

TIMEOUT = 45

# macOS python.org builds ship without a usable CA bundle, which makes every
# HTTPS call fail with CERTIFICATE_VERIFY_FAILED. Bind to certifi explicitly so
# this works on any machine without running Install Certificates.command.
try:
    import certifi

    _SSL_CTX: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None

# Populated on every failure so /ai/health can tell you WHY you are in fallback.
LAST_ERROR: dict[str, str] = {}


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as r:
        return json.loads(r.read().decode())


# ---------------- providers ----------------
def _anthropic(prompt: str, system: str, max_tokens: int) -> str:
    out = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
        {
            "model": settings.LLM_MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return "".join(b.get("text", "") for b in out.get("content", []))


def _openai_like(url: str, key: str, model: str, prompt: str, system: str, max_tokens: int) -> str:
    out = _post(
        url,
        {"Authorization": f"Bearer {key}"},
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        },
    )
    return out["choices"][0]["message"]["content"]


def _gemini(prompt: str, system: str, max_tokens: int) -> str:
    out = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.GEMINI_API_KEY}",
        {},
        {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        },
    )
    return out["candidates"][0]["content"]["parts"][0]["text"]


# ---------------- offline fallback ----------------
_STOP = set("the a an and or of to in for on with is are was were be been it this that as at by from".split())


def _offline(prompt: str, system: str) -> str:
    """Extractive summary + keyword surface. Looks deliberate, not broken."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prompt) if len(s.strip()) > 20]
    words = [w.lower().strip(".,!?;:()[]\"'") for w in prompt.split()]
    freq: dict[str, int] = {}
    for w in words:
        if len(w) > 3 and w not in _STOP:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:6]

    def score(s: str) -> int:
        low = s.lower()
        return sum(freq.get(t, 0) for t in top if t in low)

    picked = sorted(sentences, key=score, reverse=True)[:3] or [prompt[:240]]
    return (
        "[local mode] Key points:\n"
        + "\n".join(f"- {s}" for s in picked)
        + ("\n\nSalient terms: " + ", ".join(top) if top else "")
    )


# ---------------- public API ----------------
def available() -> dict[str, Any]:
    provs = [
        n
        for n, k in (
            ("anthropic", settings.ANTHROPIC_API_KEY),
            ("openai", settings.OPENAI_API_KEY),
            ("groq", settings.GROQ_API_KEY),
            ("gemini", settings.GEMINI_API_KEY),
        )
        if k
    ]
    return {
        "ok": True,
        "providers": provs,
        "mode": "offline" if (settings.OFFLINE or not provs) else provs[0],
        "ssl": "certifi" if _SSL_CTX else "system-default",
        "last_error": LAST_ERROR or None,
    }


def complete(prompt: str, system: str = "You are a concise, helpful assistant.", max_tokens: int = 800) -> str:
    if settings.OFFLINE:
        return _offline(prompt, system)
    attempts = []
    if settings.ANTHROPIC_API_KEY:
        attempts.append(("anthropic", lambda: _anthropic(prompt, system, max_tokens)))
    if settings.OPENAI_API_KEY:
        attempts.append(("openai", lambda: _openai_like("https://api.openai.com/v1/chat/completions", settings.OPENAI_API_KEY, "gpt-4o-mini", prompt, system, max_tokens)))
    if settings.GROQ_API_KEY:
        attempts.append(("groq", lambda: _openai_like("https://api.groq.com/openai/v1/chat/completions", settings.GROQ_API_KEY, settings.GROQ_MODEL, prompt, system, max_tokens)))
    if settings.GEMINI_API_KEY:
        attempts.append(("gemini", lambda: _gemini(prompt, system, max_tokens)))

    for name, fn in attempts:
        try:
            text = fn()
            if text and text.strip():
                LAST_ERROR.pop(name, None)
                return text
        except urllib.error.HTTPError as e:
            LAST_ERROR[name] = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:
            LAST_ERROR[name] = f"{type(e).__name__}: {str(e)[:200]}"
    return _offline(prompt, system)


def complete_json(prompt: str, schema_hint: str, system: str = "Reply with JSON only.") -> dict[str, Any]:
    """LLM -> dict, with brace-extraction repair. Returns {} rather than exploding."""
    raw = complete(f"{prompt}\n\nReturn ONLY valid JSON matching: {schema_hint}", system=system)
    for candidate in (raw, raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw else ""):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return {}
