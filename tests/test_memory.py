import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.services.memory import PostgresMemoryStore, build_memory_store, should_store_memory
from app.settings import Settings


class FakeAsyncStore:
    def __init__(self):
        self.put_calls = []
        self.search_results = []

    async def aput(self, namespace, key, value, index=None):
        self.put_calls.append((namespace, key, value, index))

    async def asearch(self, namespace_prefix, query=None, limit=10):
        assert namespace_prefix == ("memories", "456")
        assert query == "dolor de cabeza"
        assert limit == 3
        return self.search_results


def test_postgres_memory_store_search_returns_text_snippets():
    store_backend = FakeAsyncStore()
    store_backend.search_results = [
        SimpleNamespace(value={"text": "Antecedente A"}),
        SimpleNamespace(value={"text": "Antecedente B"}),
        SimpleNamespace(value={"kind": "profile"}),
    ]
    store = PostgresMemoryStore(store_backend)

    memories = asyncio.run(store.search("456", "dolor de cabeza", limit=3))

    assert memories == ["Antecedente A", "Antecedente B"]


def test_postgres_memory_store_persists_expected_payload_shape():
    store_backend = FakeAsyncStore()
    store = PostgresMemoryStore(store_backend)
    memories = should_store_memory(
        "Quiero cita para corte y barba manana",
        "Perfecto, lo paso a recepcion",
        "booking",
        {
            "booking_details": {
                "client_name": "Juan Perez",
                "service": "corte y barba",
                "preferred_date": "manana",
                "preferred_time": "10 am",
            }
        },
    )

    asyncio.run(store.save_memories("456", memories))

    assert len(store_backend.put_calls) == 1
    namespace, key, value, index = store_backend.put_calls[0]
    assert namespace == ("memories", "456")
    assert key
    assert value["kind"] == "profile"
    assert value["source"] == "stateful-flow"
    assert value["text"].startswith("Preferencias de cita en barberia:")
    assert datetime.fromisoformat(value["created_at"]).tzinfo == UTC
    assert index == ["text"]


def test_build_memory_store_requires_backend_when_postgres_is_enabled():
    settings = Settings(memory_backend="postgres")

    with pytest.raises(ValueError, match="configured LangGraph store"):
        build_memory_store(settings)


def test_should_store_memory_skips_trivial_turns():
    memories = should_store_memory("hola", "Hola, te ayudo con gusto", "conversation", {})

    assert memories == []


def test_should_store_memory_persists_booking_facts():
    memories = should_store_memory(
        "Quiero cita para corte y barba manana",
        "Perfecto, lo paso a recepcion",
        "booking",
        {
            "booking_details": {
                "client_name": "Juan Perez",
                "service": "corte y barba",
                "preferred_date": "manana",
                "preferred_time": "10 am",
            }
        },
    )

    assert memories
    assert memories[0].kind == "profile"
