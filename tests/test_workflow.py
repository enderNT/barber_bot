import asyncio

from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import BarbershopWorkflow
from app.models.schemas import BookingIntentPayload, ChatwootWebhook, StateRoutingDecision
from app.services.barbershop_config import BarbershopConfigLoader
from app.services.router import StateRoutingService
from app.settings import Settings


class FakeLLMService:
    def __init__(self):
        self.summary_calls = 0

    async def classify_state_route(self, routing_packet, guard_hint=None):
        del guard_hint
        message = routing_packet.user_message.lower()
        if "cita" in message or "corte" in message or routing_packet.active_goal == "booking":
            return StateRoutingDecision(
                next_node="booking",
                intent="booking",
                confidence=0.9,
                needs_retrieval=False,
                state_update={"active_goal": "booking", "stage": "collecting_booking_details"},
                reason="test",
            )
        if "horario" in message or "precio" in message:
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.85,
                needs_retrieval=True,
                state_update={"active_goal": "information", "stage": "lookup"},
                reason="test",
            )
        return StateRoutingDecision(
            next_node="conversation",
            intent="conversation",
            confidence=0.8,
            needs_retrieval=False,
            state_update={"active_goal": "conversation", "stage": "open"},
            reason="test",
        )

    async def build_conversation_reply(self, user_message, memories):
        del memories
        return f"Respuesta para: {user_message}"

    async def build_rag_reply(self, user_message, memories, barbershop_context):
        del memories, barbershop_context
        return f"RAG para: {user_message}"

    async def extract_booking_intent(
        self, user_message, memories, barbershop_context, contact_name, current_details=None, pending_question=None
    ):
        del memories, barbershop_context, contact_name, pending_question
        current_details = current_details or {}
        payload = BookingIntentPayload(
            client_name=current_details.get("client_name", "Juan Perez"),
            service=current_details.get("service", "corte clasico"),
            preferred_date="manana" if "manana" in user_message.lower() else current_details.get("preferred_date"),
            preferred_time="10 am" if "10" in user_message else current_details.get("preferred_time"),
            missing_fields=[] if ("manana" in user_message.lower() and "10" in user_message) else ["preferred_time"],
            should_handoff=True,
            confidence=0.9,
        )
        return payload, f"Solicitud lista: {user_message}"

    async def build_state_summary(self, current_summary, user_message, assistant_message, active_goal, stage):
        self.summary_calls += 1
        return f"{current_summary} | {active_goal}:{stage} | {user_message} -> {assistant_message}".strip(" |")


class FakeMemoryStore:
    def __init__(self):
        self.saved = []

    async def search(self, contact_id, query, limit=5):
        del contact_id, query, limit
        return ["Recuerdo util", "Prefiere horario vespertino"]

    async def save_memories(self, contact_id, memories):
        self.saved.append((contact_id, [memory.model_dump() for memory in memories]))


class FakeQdrantService:
    def __init__(self):
        self.calls = 0

    async def build_context(self, *args, **kwargs):
        self.calls += 1
        del args, kwargs
        return "Contexto RAG simulado"


def build_webhook(message: str, conversation_id: int = 123) -> ChatwootWebhook:
    return ChatwootWebhook(
        content=message,
        conversation={"id": conversation_id},
        contact={"id": 456, "name": "Juan Perez"},
        event="message_created",
        message_type="incoming",
    )


def build_workflow():
    llm = FakeLLMService()
    settings = Settings(
        _env_file=None,
        llm_api_key=None,
        openai_api_key=None,
        memory_backend="in_memory",
        checkpoint_backend="memory",
    )
    router = StateRoutingService(settings, llm)
    memory = FakeMemoryStore()
    qdrant = FakeQdrantService()
    workflow = BarbershopWorkflow(
        router,
        llm,
        memory,
        BarbershopConfigLoader(config_path="config/barbershop.json"),  # type: ignore[arg-type]
        qdrant,
        settings,
    )
    return workflow, memory, qdrant, llm


def build_workflow_with_checkpointer(checkpointer):
    llm = FakeLLMService()
    settings = Settings(_env_file=None, memory_backend="in_memory", checkpoint_backend="memory")
    router = StateRoutingService(settings, llm)
    memory = FakeMemoryStore()
    qdrant = FakeQdrantService()
    workflow = BarbershopWorkflow(
        router,
        llm,
        memory,
        BarbershopConfigLoader(config_path="config/barbershop.json"),  # type: ignore[arg-type]
        qdrant,
        settings,
        checkpointer=checkpointer,
    )
    return workflow


def test_workflow_routes_to_conversation():
    workflow, memory, qdrant, llm = build_workflow()

    result = asyncio.run(workflow.run(build_webhook("Necesito informacion general sobre la barberia")))

    assert result["next_node"] == "conversation"
    assert result["response_text"] == "Respuesta para: Necesito informacion general sobre la barberia"
    assert result["handoff_required"] is False
    assert qdrant.calls == 0
    assert memory.saved
    assert llm.summary_calls >= 1


def test_workflow_routes_to_rag():
    workflow, memory, qdrant, _ = build_workflow()

    result = asyncio.run(workflow.run(build_webhook("Cuales son sus horarios?")))

    assert result["next_node"] == "rag"
    assert result["response_text"] == "RAG para: Cuales son sus horarios?"
    assert result["handoff_required"] is False
    assert qdrant.calls == 1
    assert memory.saved


def test_workflow_keeps_booking_state_across_turns():
    workflow, memory, qdrant, llm = build_workflow()
    conversation_id = 777

    first = asyncio.run(workflow.run(build_webhook("Quiero una cita para corte", conversation_id=conversation_id)))
    second = asyncio.run(workflow.run(build_webhook("manana a las 10", conversation_id=conversation_id)))

    assert first["next_node"] == "booking"
    assert first["stage"] == "collecting_booking_details"
    assert second["next_node"] == "booking"
    assert second["stage"] == "ready_for_handoff"
    assert second["booking_details"]["preferred_time"] == "10 am"
    assert qdrant.calls == 0
    assert memory.saved
    assert llm.summary_calls >= 1


def test_workflow_persists_state_when_recreated_with_shared_checkpointer():
    checkpointer = MemorySaver()
    first_workflow = build_workflow_with_checkpointer(checkpointer)
    second_workflow = build_workflow_with_checkpointer(checkpointer)
    conversation_id = 778

    first = asyncio.run(first_workflow.run(build_webhook("Quiero una cita para corte", conversation_id=conversation_id)))
    second = asyncio.run(second_workflow.run(build_webhook("manana a las 10", conversation_id=conversation_id)))

    assert first["stage"] == "collecting_booking_details"
    assert second["stage"] == "ready_for_handoff"
    assert second["booking_details"]["preferred_time"] == "10 am"
