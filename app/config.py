from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
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
    llm_provider: Literal["ollama", "openai", "anthropic", "groq", "grok"] = "ollama"
    ollama_base_url: str = ""
    ollama_model: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = "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"
    xai_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"


    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "duckdb:///./data/local.duckdb"

    # ── Cache ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    context_ttl_seconds: int = 3600   # 1 hour default

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # ── GCS ──────────────────────────────────────────────────────────────────
    google_application_credentials: str = ""


settings = Settings()
