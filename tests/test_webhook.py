from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import get_settings


def build_test_client(monkeypatch) -> TestClient:
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    client = TestClient(create_app())
    get_settings.cache_clear()
    return client


def test_healthcheck(monkeypatch):
    client = build_test_client(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chatwoot_webhook_accepts_immediately(monkeypatch):
    client = build_test_client(monkeypatch)
    payload = {
        "content": "Necesito una cita",
        "conversation": {"id": 321},
        "contact": {"id": 654, "name": "Maria"},
    }

    response = client.post("/webhooks/chatwoot", json=payload)

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "conversation_id": "321"}
