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

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: int = 30
    openai_temperature: float = 0.1
    semantic_router_debug: bool = False
    router_input_debug: bool = False

    clinic_config_path: Path = Field(default=Path("config/clinic.json"))

    memory_backend: Literal["in_memory", "mem0_local", "mem0_platform"] = "in_memory"
    mem0_api_key: str | None = None
    mem0_org_id: str | None = None
    mem0_project_id: str | None = None

    qdrant_enabled: bool = False
    qdrant_simulate: bool = True
    qdrant_base_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "clinic_knowledge"
    qdrant_timeout_seconds: int = 10
    qdrant_top_k: int = 5
    qdrant_vector_size: int = 8

    chatwoot_reply_enabled: bool = False
    chatwoot_api_base_url: str | None = None
    chatwoot_api_token: str | None = None
    chatwoot_account_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
