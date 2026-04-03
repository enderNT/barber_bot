from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    llm_provider: str = "openai_compatible"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: int | None = None
    llm_temperature: float | None = None

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: int = 30
    openai_temperature: float = 0.1
    state_router_confidence_threshold: float = 0.62
    state_router_retrieval_threshold: float = 0.58
    summary_refresh_turn_threshold: int = 4
    summary_refresh_char_threshold: int = 900
    memory_search_limit: int = 3

    barbershop_config_path: Path = Field(default=Path("config/barbershop.json"))

    memory_backend: Literal["in_memory", "postgres"] = "in_memory"
    checkpoint_backend: Literal["memory", "postgres"] = "postgres"
    postgres_dsn: str | None = None
    memory_embedding_model: str = "text-embedding-3-small"
    memory_embedding_dims: int = 1536

    qdrant_enabled: bool = False
    qdrant_simulate: bool = True
    qdrant_base_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "barbershop_knowledge"
    qdrant_timeout_seconds: int = 10
    qdrant_top_k: int = 5
    qdrant_vector_size: int = 8

    chatwoot_reply_enabled: bool = False
    chatwoot_api_base_url: str | None = None
    chatwoot_api_token: str | None = None
    chatwoot_account_id: str | None = None

    tracer_enabled: bool = False
    tracer_app_key: str = "barbershop-assistant"
    tracer_batch_size: int = 25
    tracer_flush_interval_seconds: float = 2.0
    tracer_projectors_enabled: bool = True

    @property
    def resolved_llm_provider(self) -> str:
        return self.llm_provider.strip().lower()

    @property
    def resolved_llm_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key

    @property
    def resolved_llm_base_url(self) -> str | None:
        return self.llm_base_url or self.openai_base_url

    @property
    def resolved_llm_model(self) -> str:
        return (self.llm_model or self.openai_model).strip()

    @property
    def resolved_llm_timeout_seconds(self) -> int:
        return self.llm_timeout_seconds or self.openai_timeout_seconds

    @property
    def resolved_llm_temperature(self) -> float | None:
        if self.llm_temperature is not None:
            return self.llm_temperature
        return self.openai_temperature


@lru_cache
def get_settings() -> Settings:
    return Settings()
