"""Gushwork Inbound AEO Voice Agent — Configuration."""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised settings loaded from .env or Render env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Retell AI ──────────────────────────────────────────
    retell_api_key: str = ""
    retell_agent_id: str = ""
    retell_webhook_secret: str = ""

    # ── LLM ────────────────────────────────────────────────
    llm_provider: str = "groq"  # "gemini" or "groq"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Cal.com ────────────────────────────────────────────
    calcom_api_key: str = ""
    calcom_event_type_id: int = 0

    # ── CRM ────────────────────────────────────────────────
    hubspot_api_key: str = ""
    crm_provider: str = "hubspot"

    # ── Server ─────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = int(os.environ.get("PORT", "8000"))  # Render sets PORT
    webhook_base_url: str = "http://localhost:8000"

    # ── LLM Tuning ─────────────────────────────────────────
    llm_temperature: float = 0.2
    llm_max_tokens: int = 300
    llm_context_messages: int = 16
    vad_silence_ms: int = 400

    # ── Pipeline Runtime Controls ──────────────────────────
    call_state_ttl_seconds: int = 7200
    audit_query_timeout_seconds: float = 20.0
    audit_cache_ttl_seconds: int = 3600

    # ── LangSmith Observability ────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "gushwork-sdr"
    langsmith_tracing: bool = True


settings = Settings()

# ── LangSmith env var injection ────────────────────────────
# LangSmith reads from env vars, so we inject them from our config.
if settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
elif settings.langsmith_tracing:
    # Enable tracing even without key (logs locally)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
