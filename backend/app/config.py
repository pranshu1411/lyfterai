from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KeyEntry:
    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    key_pool: list[KeyEntry] = field(default_factory=list)
    default_model: str = "groq/openai/gpt-oss-20b"
    default_system_prompt: str = "You are a helpful, concise assistant."
    database_url: str = "sqlite:///./app.db"
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    enable_failure_simulation: bool = False
    cache_enabled: bool = True


def load_settings() -> Settings:
    raw_keys = os.getenv("LLM_API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    default_model = (
        os.getenv("LLM_MODEL", "groq/openai/gpt-oss-20b").strip()
        or "groq/openai/gpt-oss-20b"
    )
    raw_models = os.getenv("LLM_MODELS", "").strip()
    models = [m.strip() for m in raw_models.split(",") if m.strip()] if raw_models else []

    key_pool: list[KeyEntry] = []
    for idx, api_key in enumerate(keys):
        model_for_key = models[idx] if idx < len(models) else default_model
        key_pool.append(KeyEntry(key_id=f"key_{idx + 1}", api_key=api_key, model=model_for_key))

    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise assistant.",
    ).strip()

    database_url = os.getenv("DATABASE_URL", "sqlite:///./app.db").strip() or "sqlite:///./app.db"

    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    allowed_origins = [o.strip().rstrip("/") for o in raw_origins.split(",") if o.strip()]

    enable_sim = os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() == "true"

    return Settings(
        key_pool=key_pool,
        default_model=default_model,
        default_system_prompt=default_system_prompt,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_sim,
        cache_enabled=True,
    )
