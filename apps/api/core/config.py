"""Central config. Everything has a safe default so the app boots with zero setup."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class Settings:
    # --- identity ---
    APP_NAME: str = _env("APP_NAME", "Hackathon Kit")
    TAGLINE: str = _env("TAGLINE", "Rename me in .env once the PS drops")

    # --- runtime ---
    ENV: str = _env("ENV", "dev")
    SECRET_KEY: str = _env("SECRET_KEY", "dev-secret-change-me")
    TOKEN_TTL_MIN: int = int(_env("TOKEN_TTL_MIN", "720"))

    # --- storage ---
    DATA_DIR: Path = ROOT / "data"
    DB_URL: str = _env("DB_URL", f"sqlite:///{ROOT / 'data' / 'app.db'}")
    MODEL_DIR: Path = ROOT / "data" / "models"
    INDEX_DIR: Path = ROOT / "data" / "index"
    UPLOAD_DIR: Path = ROOT / "data" / "raw" / "uploads"

    # --- AI providers (all optional) ---
    ANTHROPIC_API_KEY: str = _env("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: str = _env("OPENAI_API_KEY")
    GROQ_API_KEY: str = _env("GROQ_API_KEY")
    GEMINI_API_KEY: str = _env("GEMINI_API_KEY")
    LLM_MODEL: str = _env("LLM_MODEL", "claude-sonnet-5-5")
    GROQ_MODEL: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Hard offline switch -- flip this if venue wifi dies mid-demo.
    OFFLINE: bool = _env("OFFLINE", "0") in ("1", "true", "yes")

    EMBED_MODEL: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VISION_WEIGHTS: str = _env("VISION_WEIGHTS", str(ROOT / "data" / "models" / "cnn.pt"))

    CORS_ORIGINS: list[str] = ["*"]

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.MODEL_DIR, self.INDEX_DIR, self.UPLOAD_DIR):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def has_llm(self) -> bool:
        if self.OFFLINE:
            return False
        return any([self.ANTHROPIC_API_KEY, self.OPENAI_API_KEY, self.GROQ_API_KEY, self.GEMINI_API_KEY])


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
