from __future__ import annotations

import logging

from fastapi import FastAPI

from app.graph.workflow import ClinicWorkflow
from app.services.agent import ClinicAgentService
from app.services.chatwoot import ChatwootClient
from app.services.clinic_config import ClinicConfigLoader
from app.services.llm import ClinicLLMService, OpenAIClient
from app.services.memory import build_memory_store
from app.services.router import ClinicIntentRouterService
from app.services.qdrant import QdrantRetrievalService
from app.settings import get_settings
from app.webhooks.routes import build_webhook_router


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    clinic_config_loader = ClinicConfigLoader(settings.clinic_config_path)
    openai_client = OpenAIClient(settings)
    router_service = ClinicIntentRouterService(settings)
    llm_service = ClinicLLMService(openai_client)
    memory_store = build_memory_store(settings)
    qdrant_service = QdrantRetrievalService(settings)
    workflow = ClinicWorkflow(router_service, llm_service, memory_store, clinic_config_loader, qdrant_service)
    agent_service = ClinicAgentService(workflow, ChatwootClient(settings))

    app = FastAPI(title="Clinica Assistant", version="0.1.0")
    app.include_router(build_webhook_router(agent_service))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return app
