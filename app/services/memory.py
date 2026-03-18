from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Protocol

from app.models.schemas import MemoryRecord
from app.settings import Settings

logger = logging.getLogger(__name__)


def _unwrap_mem0_results(results: Any) -> Any:
    current = results
    for _ in range(3):
        if not isinstance(current, Mapping):
            return current
        for key in ("results", "memories", "items"):
            if key in current:
                current = current[key]
                break
        else:
            return current
    return current


def _extract_memory_text(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("memory", "text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    for attr in ("memory", "text", "content"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _normalize_mem0_search_results(results: Any, limit: int) -> list[str]:
    raw_items = _unwrap_mem0_results(results)
    if raw_items is None:
        return []
    if isinstance(raw_items, Mapping):
        logger.warning("Mem0 search returned an unexpected mapping shape: keys=%s", sorted(raw_items.keys()))
        return []
    if isinstance(raw_items, (str, bytes)):
        raw_items = [raw_items]
    elif not isinstance(raw_items, Iterable):
        raw_items = [raw_items]

    memories: list[str] = []
    for item in islice(raw_items, limit):
        memory = _extract_memory_text(item)
        if memory:
            memories.append(memory)
    return memories


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


class Mem0LocalMemoryStore:
    def __init__(self) -> None:
        from mem0 import Memory

        self._client = Memory()

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        results = self._client.search(query, filters={"user_id": contact_id}, limit=limit)
        return _normalize_mem0_search_results(results, limit=limit)

    async def save_memories(self, contact_id: str, memories: list[MemoryRecord]) -> None:
        if not memories:
            return
        messages = [{"role": "system", "content": memory.text} for memory in memories]
        self._client.add(messages, user_id=contact_id)


class Mem0PlatformMemoryStore:
    def __init__(self, settings: Settings) -> None:
        from mem0 import MemoryClient

        client_kwargs: dict[str, str] = {}
        if settings.mem0_api_key:
            client_kwargs["api_key"] = settings.mem0_api_key
        if settings.mem0_org_id:
            client_kwargs["org_id"] = settings.mem0_org_id
        if settings.mem0_project_id:
            client_kwargs["project_id"] = settings.mem0_project_id
        self._client = MemoryClient(**client_kwargs)

    async def search(self, contact_id: str, query: str, limit: int = 5) -> list[str]:
        results = self._client.search(query, filters={"user_id": contact_id}, top_k=limit)
        return _normalize_mem0_search_results(results, limit=limit)

    async def save_memories(self, contact_id: str, memories: list[MemoryRecord]) -> None:
        if not memories:
            return
        messages = [{"role": "system", "content": memory.text} for memory in memories]
        self._client.add(messages, user_id=contact_id)


def build_memory_store(settings: Settings) -> MemoryStore:
    try:
        if settings.memory_backend == "mem0_local":
            return Mem0LocalMemoryStore()
        if settings.memory_backend == "mem0_platform":
            return Mem0PlatformMemoryStore(settings)
    except Exception as exc:  # pragma: no cover - depende de entorno externo
        logger.warning("Falling back to in-memory store because mem0 failed to initialize: %s", exc)
    return InMemoryMemoryStore()


def should_store_memory(user_message: str, assistant_message: str, route: str, state: dict[str, Any]) -> list[MemoryRecord]:
    lowered_user = user_message.lower().strip()
    lowered_assistant = assistant_message.lower().strip()
    memories: list[MemoryRecord] = []

    if route == "appointment":
        slots = state.get("appointment_slots") or {}
        relevant_bits = []
        for key in ("patient_name", "reason", "preferred_date", "preferred_time"):
            value = slots.get(key)
            if value:
                relevant_bits.append(f"{key}={value}")
        if relevant_bits:
            memories.append(
                MemoryRecord(
                    kind="profile",
                    text="Preferencias de cita: " + ", ".join(relevant_bits),
                )
            )
        elif lowered_user and not _is_trivial_turn(lowered_user):
            memories.append(
                MemoryRecord(
                    kind="episode",
                    text=f"El usuario solicito apoyo para agendar una cita: {user_message}",
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
    preference_markers = (
        "prefiero",
        "me gusta",
        "solo por",
        "no puedo",
        "no puedo por",
        "por favor escribeme",
        "mejor por",
    )
    return any(marker in user_message for marker in preference_markers)
