"""Application configuration loaded from environment variables via pydantic-settings."""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime settings with safe defaults so the app starts without any keys set."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "anthropic", "groq"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "duckdb:///./data/local.duckdb"

    # ── Cache ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    context_ttl_seconds: int = 3600

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # ── GCS ──────────────────────────────────────────────────────────────────
    google_application_credentials: str = ""


settings = Settings()
