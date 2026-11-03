"""Speech-to-text. Optional: only needed for voice-first PS statements
(accessibility, farmer helplines, field reporting in low-literacy settings).

Install at the venue only if you need it:  uv pip install faster-whisper
"""
from __future__ import annotations

from typing import Any

_model = None


def load(size: str = "tiny") -> bool:
    global _model
    if _model is not None:
        return True
    try:
        from faster_whisper import WhisperModel

        _model = WhisperModel(size, device="cpu", compute_type="int8")
        return True
    except Exception:
        _model = None
        return False


def transcribe(path: str, language: str | None = None) -> dict[str, Any]:
    if not load():
        return {"ok": False, "error": "faster-whisper not installed. Run: uv pip install faster-whisper"}
    segments, info = _model.transcribe(path, language=language, vad_filter=True)
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    return {"ok": True, "language": info.language, "duration": round(info.duration, 2), "text": " ".join(s["text"] for s in segs), "segments": segs}


def available() -> dict[str, Any]:
    try:
        import faster_whisper  # noqa

        return {"ok": True, "installed": True, "loaded": _model is not None}
    except Exception:
        return {"ok": True, "installed": False, "hint": "uv pip install faster-whisper"}
