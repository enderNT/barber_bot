from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from langgraph.store.base import Embeddings
from openai import AsyncOpenAI, OpenAI

from app.models.schemas import MemoryRecord
from app.settings import Settings


class MemoryStore(Protocol):
    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        ...

    async def save_memories(self, contact_id: str, memories: list[MemoryRecord]) -> None:
        ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        del query
        memories = self._store.get(contact_id, [])
        return memories[-limit:]

    async def save_memories(self, contact_id: str, memories: list[MemoryRecord]) -> None:
        snippets = self._store.setdefault(contact_id, [])
        for memory in memories:
            if memory.text not in snippets:
                snippets.append(memory.text)


class OpenAIEmbeddingsAdapter(Embeddings):
    def __init__(self, settings: Settings) -> None:
        client_kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key or settings.resolved_llm_api_key or "sk-placeholder",
        }
        base_url = settings.openai_base_url or settings.resolved_llm_base_url
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self._sync_client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)
        self._model = settings.memory_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._sync_client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._sync_client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._async_client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        response = await self._async_client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding


class PostgresMemoryStore:
    def __init__(self, store: Any) -> None:
        self._store = store

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        items = await self._store.asearch(
            ("memories", contact_id),
            query=query,
            limit=limit,
        )
        memories: list[str] = []
        for item in items:
            text = item.value.get("text")
            if isinstance(text, str) and text:
                memories.append(text)
        return memories

    async def save_memories(self, contact_id: str, memories: list[MemoryRecord]) -> None:
        namespace = ("memories", contact_id)
        for memory in memories:
            await self._store.aput(
                namespace,
                str(uuid4()),
                {
                    "text": memory.text,
                    "kind": memory.kind,
                    "source": memory.source,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                index=["text"],
            )


def build_memory_store(settings: Settings, store_backend: Any | None = None) -> MemoryStore:
    if settings.memory_backend == "postgres":
        if store_backend is None:
            raise ValueError("Postgres memory backend requires a configured LangGraph store.")
        return PostgresMemoryStore(store_backend)
    return InMemoryMemoryStore()


def build_memory_index_config(settings: Settings) -> dict[str, Any]:
    return {
        "dims": settings.memory_embedding_dims,
        "embed": OpenAIEmbeddingsAdapter(settings),
        "fields": ["text"],
    }


def should_store_memory(user_message: str, assistant_message: str, route: str, state: dict[str, Any]) -> list[MemoryRecord]:
    lowered_user = user_message.lower().strip()
    lowered_assistant = assistant_message.lower().strip()
    memories: list[MemoryRecord] = []

    if route == "booking":
        details = state.get("booking_details") or {}
        relevant_bits = []
        for key in ("client_name", "service", "preferred_date", "preferred_time"):
            value = details.get(key)
            if value:
                relevant_bits.append(f"{key}={value}")
        if relevant_bits:
            memories.append(
                MemoryRecord(
                    kind="profile",
                    text="Preferencias de cita en barberia: " + ", ".join(relevant_bits),
                )
            )
        elif lowered_user and not _is_trivial_turn(lowered_user):
            memories.append(
                MemoryRecord(
                    kind="episode",
                    text=f"El usuario solicito apoyo para agendar una cita en barberia: {user_message}",
                )
            )
        return memories

    if _is_trivial_turn(lowered_user):
        return memories

    if _looks_like_persistent_preference(lowered_user):
        memories.append(
            MemoryRecord(
                kind="profile",
                text=f"Preferencia del usuario: {user_message}",
            )
        )
        return memories

    if route == "rag" and lowered_assistant:
        memories.append(
            MemoryRecord(
                kind="episode",
                text=f"Consulta informativa resuelta sobre: {user_message}",
            )
        )
        return memories

    if lowered_assistant and len(lowered_user) >= 18:
        memories.append(
            MemoryRecord(
                kind="episode",
                text=f"Conversacion util: {user_message} -> {assistant_message}",
            )
        )
    return memories


def _is_trivial_turn(user_message: str) -> bool:
    trivial_phrases = {
        "hola",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "gracias",
        "ok",
        "okay",
        "si",
        "no",
    }
    compact = " ".join(user_message.split())
    return compact in trivial_phrases or len(compact) <= 3


def _looks_like_persistent_preference(user_message: str) -> bool:
    preference_markers: Sequence[str] = (
        "prefiero",
        "me gusta",
        "solo por",
        "no puedo",
        "no puedo por",
        "por favor escribeme",
        "escribeme por",
        "contactame por",
        "mi horario ideal",
        "normalmente puedo",
    )
    return any(marker in user_message for marker in preference_markers)
