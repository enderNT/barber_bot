from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.settings import get_settings
from app.webhooks.routes import build_webhook_router


def build_test_client(monkeypatch) -> TestClient:
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("MEMORY_BACKEND", "in_memory")
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    client = TestClient(create_app())
    get_settings.cache_clear()
    return client


def test_healthcheck(monkeypatch):
    with build_test_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chatwoot_webhook_accepts_immediately(monkeypatch):
    payload = {
        "event": "message_created",
        "message_type": "incoming",
        "content": "Necesito una cita",
        "conversation": {"id": 321},
        "contact": {"id": 654, "name": "Maria"},
    }

    with build_test_client(monkeypatch) as client:
        response = client.post("/webhooks/chatwoot", json=payload)

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "conversation_id": "321"}


def test_chatwoot_webhook_ignores_outgoing_messages(monkeypatch):
    class FakeAgentService:
        async def process_webhook(self, payload, flow_id=None):
            raise AssertionError("Outgoing events should not be processed")

    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr("app.webhooks.routes.asyncio.create_task", fake_create_task)

    app = FastAPI()
    app.state.agent_service = FakeAgentService()
    app.include_router(build_webhook_router())
    client = TestClient(app)

    response = client.post(
        "/webhooks/chatwoot",
        json={
            "event": "message_created",
            "message_type": "outgoing",
            "content": "respuesta del bot",
            "conversation": {"id": 321},
            "contact": {"id": 654, "name": "Maria"},
        },
    )

    assert response.status_code == 202
    assert response.json() == {"status": "ignored", "conversation_id": "321"}
    assert scheduled == []


def test_create_app_requires_postgres_dsn_when_postgres_backends_are_enabled(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("MEMORY_BACKEND", "postgres")
    monkeypatch.setenv("CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("POSTGRES_DSN", "")

    with pytest.raises(ValueError, match="POSTGRES_DSN"):
        with TestClient(create_app()):
            pass

    get_settings.cache_clear()
