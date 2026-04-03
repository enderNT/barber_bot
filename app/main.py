from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.graph.workflow import BarbershopWorkflow
from app.observability.flow_logger import configure_flow_logger
from app.services.agent import BarbershopAgentService
from app.services.barbershop_config import BarbershopConfigLoader
from app.services.chatwoot import ChatwootClient
from app.services.llm import BarbershopLLMService, build_llm_provider
from app.services.memory import build_memory_index_config, build_memory_store
from app.services.qdrant import QdrantRetrievalService
from app.services.router import StateRoutingService
from app.settings import Settings, get_settings
from app.tracing import (
    AsyncBatchTraceSink,
    BarbershopTraceNormalizer,
    BarbershopTraceProjector,
    NoopTraceSink,
    build_barbershop_field_policy,
)
from app.webhooks.routes import build_webhook_router


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    configure_flow_logger(getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(title="Barbershop Assistant", version="0.1.0", lifespan=_build_lifespan(settings))
    app.include_router(build_webhook_router())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return app


def _build_lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            store_backend, checkpointer = await _build_persistence_resources(settings, stack)
            trace_sink, trace_normalizer = await _build_trace_resources(settings, stack)
            app.state.agent_service = _build_agent_service(
                settings,
                store_backend,
                checkpointer,
                trace_sink=trace_sink,
                trace_normalizer=trace_normalizer,
            )
            yield

    return lifespan


def _build_agent_service(
    settings: Settings,
    store_backend: Any | None,
    checkpointer: Any | None,
    *,
    trace_sink: Any | None = None,
    trace_normalizer: Any | None = None,
) -> BarbershopAgentService:
    barbershop_config_loader = BarbershopConfigLoader(settings.barbershop_config_path)
    llm_provider = build_llm_provider(settings)
    llm_service = BarbershopLLMService(llm_provider)
    router_service = StateRoutingService(settings, llm_service)
    memory_store = build_memory_store(settings, store_backend)
    qdrant_service = QdrantRetrievalService(settings)
    workflow = BarbershopWorkflow(
        router_service,
        llm_service,
        memory_store,
        barbershop_config_loader,
        qdrant_service,
        settings,
        checkpointer=checkpointer,
        store_backend=store_backend,
        trace_sink=trace_sink,
        trace_normalizer=trace_normalizer,
    )
    return BarbershopAgentService(workflow, ChatwootClient(settings))


async def _build_persistence_resources(settings: Settings, stack: AsyncExitStack) -> tuple[Any | None, Any | None]:
    requires_postgres = settings.memory_backend == "postgres" or settings.checkpoint_backend == "postgres"
    if requires_postgres and not settings.postgres_dsn:
        raise ValueError("POSTGRES_DSN is required when using postgres memory or checkpoint backends.")

    store_backend: Any | None = None
    checkpointer: Any | None = None

    if settings.memory_backend == "postgres":
        store_backend = await _open_postgres_store(settings, stack)
        await store_backend.setup()

    if settings.checkpoint_backend == "postgres":
        checkpointer = await _open_postgres_checkpointer(settings, stack)
        await checkpointer.setup()

    return store_backend, checkpointer


async def _build_trace_resources(settings: Settings, stack: AsyncExitStack) -> tuple[Any, BarbershopTraceNormalizer]:
    trace_normalizer = BarbershopTraceNormalizer()
    if not settings.tracer_enabled:
        return NoopTraceSink(), trace_normalizer

    if not settings.postgres_dsn:
        raise ValueError("POSTGRES_DSN is required when TRACER_ENABLED is true.")

    from app.tracing.postgres import AsyncPostgresTraceRepository

    field_policy = build_barbershop_field_policy()
    projectors = [BarbershopTraceProjector()] if settings.tracer_projectors_enabled else []
    repository = AsyncPostgresTraceRepository(
        settings.postgres_dsn,
        projectors=projectors,
        field_policy=field_policy,
    )
    sink = AsyncBatchTraceSink(
        repository,
        batch_size=settings.tracer_batch_size,
        flush_interval_seconds=settings.tracer_flush_interval_seconds,
    )
    await sink.start()
    stack.push_async_callback(sink.close)
    return sink, trace_normalizer


async def _open_postgres_store(settings: Settings, stack: AsyncExitStack) -> Any:
    from langgraph.store.postgres.aio import AsyncPostgresStore

    store_cm = AsyncPostgresStore.from_conn_string(
        settings.postgres_dsn,
        index=build_memory_index_config(settings),
    )
    return await stack.enter_async_context(store_cm)


async def _open_postgres_checkpointer(settings: Settings, stack: AsyncExitStack) -> Any:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.postgres_dsn)
    return await stack.enter_async_context(checkpointer_cm)
