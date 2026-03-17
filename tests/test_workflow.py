import asyncio

from app.graph.workflow import ClinicWorkflow
from app.models.schemas import AppointmentIntentPayload, ChatwootWebhook, IntentDecision
from app.services.clinic_config import ClinicConfigLoader


class FakeRouterService:
    async def route_intent(self, user_message, memories, clinic_context):
        del memories, clinic_context
        if "cita" in user_message.lower():
            return IntentDecision(intent="appointment_intent", confidence=0.9, reason="test")
        if "horario" in user_message.lower():
            return IntentDecision(intent="rag", confidence=0.85, reason="test")
        return IntentDecision(intent="conversation", confidence=0.8, reason="test")


class FakeLLMService:
    async def build_conversation_reply(self, user_message, memories, clinic_context):
        del memories, clinic_context
        return f"Respuesta para: {user_message}"

    async def build_rag_reply(self, user_message, memories, clinic_context):
        del memories, clinic_context
        return f"RAG para: {user_message}"

    async def extract_appointment_intent(self, user_message, memories, clinic_context, contact_name):
        del memories, clinic_context, contact_name
        payload = AppointmentIntentPayload(
            patient_name="Juan Perez",
            reason="medicina general",
            preferred_date="manana",
            preferred_time="10 am",
            missing_fields=[],
            should_handoff=True,
            confidence=0.9,
        )
        return payload, f"Solicitud lista: {user_message}"


class FakeMemoryStore:
    def __init__(self):
        self.saved = []

    async def search(self, contact_id, query, limit=5):
        del contact_id, query, limit
        return ["Recuerdo util"]

    async def save_exchange(self, contact_id, user_message, assistant_message):
        self.saved.append((contact_id, user_message, assistant_message))


class FakeQdrantService:
    async def build_context(self, *args, **kwargs):
        del args, kwargs
        return ""


def build_webhook(message: str) -> ChatwootWebhook:
    return ChatwootWebhook(
        content=message,
        conversation={"id": 123},
        contact={"id": 456, "name": "Juan Perez"},
    )


def test_workflow_routes_to_conversation():
    memory = FakeMemoryStore()
    workflow = ClinicWorkflow(
        FakeRouterService(),
        FakeLLMService(),
        memory,
        ClinicConfigLoader(config_path="config/clinic.json"),  # type: ignore[arg-type]
        FakeQdrantService(),
    )

    result = asyncio.run(workflow.run(build_webhook("Cuales son sus servicios?")))

    assert result["intent"] == "conversation"
    assert result["response_text"] == "Respuesta para: Cuales son sus servicios?"
    assert result["handoff_required"] is False
    assert memory.saved


def test_workflow_routes_to_rag():
    memory = FakeMemoryStore()
    workflow = ClinicWorkflow(
        FakeRouterService(),
        FakeLLMService(),
        memory,
        ClinicConfigLoader(config_path="config/clinic.json"),  # type: ignore[arg-type]
        FakeQdrantService(),
    )

    result = asyncio.run(workflow.run(build_webhook("Cuales son sus horarios?")))

    assert result["intent"] == "rag"
    assert result["response_text"] == "RAG para: Cuales son sus horarios?"
    assert result["handoff_required"] is False
    assert memory.saved


def test_workflow_routes_to_appointment():
    memory = FakeMemoryStore()
    workflow = ClinicWorkflow(
        FakeRouterService(),
        FakeLLMService(),
        memory,
        ClinicConfigLoader(config_path="config/clinic.json"),  # type: ignore[arg-type]
        FakeQdrantService(),
    )

    result = asyncio.run(workflow.run(build_webhook("Quiero una cita de medicina general manana a las 10 am")))

    assert result["intent"] == "appointment_intent"
    assert result["handoff_required"] is True
    assert result["appointment_payload"]["preferred_time"] == "10 am"
    assert memory.saved
