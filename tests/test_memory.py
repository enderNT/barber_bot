import asyncio

from app.services.memory import (
    Mem0LocalMemoryStore,
    Mem0PlatformMemoryStore,
    _normalize_mem0_search_results,
)


def test_normalize_mem0_search_results_accepts_v2_dict_shape():
    results = {
        "results": [
            {"memory": "Paciente prefiere horario matutino"},
            {"memory": "Tiene seguro activo"},
            {"text": "Dato alternativo"},
        ]
    }

    normalized = _normalize_mem0_search_results(results, limit=2)

    assert normalized == [
        "Paciente prefiere horario matutino",
        "Tiene seguro activo",
    ]


def test_normalize_mem0_search_results_accepts_list_shape():
    results = [
        {"memory": "Primera memoria"},
        {"memory": "Segunda memoria"},
    ]

    normalized = _normalize_mem0_search_results(results, limit=5)

    assert normalized == ["Primera memoria", "Segunda memoria"]


def test_mem0_local_search_normalizes_dict_results():
    class FakeLocalClient:
        def search(self, query, filters, limit):
            assert query == "dolor de cabeza"
            assert filters == {"user_id": "456"}
            assert limit == 3
            return {"results": [{"memory": "Antecedente A"}, {"memory": "Antecedente B"}]}

    store = object.__new__(Mem0LocalMemoryStore)
    store._client = FakeLocalClient()

    memories = asyncio.run(store.search("456", "dolor de cabeza", limit=3))

    assert memories == ["Antecedente A", "Antecedente B"]


def test_mem0_platform_search_normalizes_dict_results():
    class FakePlatformClient:
        def search(self, query, filters, top_k):
            assert query == "agendar cita"
            assert filters == {"user_id": "789"}
            assert top_k == 2
            return {"results": [{"memory": "Prefiere WhatsApp"}, {"memory": "No disponible por la tarde"}]}

    store = object.__new__(Mem0PlatformMemoryStore)
    store._client = FakePlatformClient()

    memories = asyncio.run(store.search("789", "agendar cita", limit=2))

    assert memories == ["Prefiere WhatsApp", "No disponible por la tarde"]
