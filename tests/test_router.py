import pytest

from app.models.schemas import RoutingPacket, StateRoutingDecision
from app.services.router import StateRoutingService
from app.settings import Settings


class FakeLLMService:
    def __init__(self, decision: StateRoutingDecision | None = None):
        self.decision = decision or StateRoutingDecision(
            next_node="conversation",
            intent="conversation",
            confidence=0.4,
            needs_retrieval=False,
            state_update={},
            reason="llm",
        )
        self.calls = []

    async def classify_state_route(self, routing_packet: RoutingPacket, guard_hint=None):
        self.calls.append((routing_packet, guard_hint))
        return self.decision


def build_service(decision: StateRoutingDecision | None = None) -> StateRoutingService:
    return StateRoutingService(Settings(openai_api_key=None), FakeLLMService(decision))


@pytest.mark.asyncio
async def test_router_keeps_appointment_follow_up_in_appointment():
    service = build_service()

    decision = await service.route_state(
        user_message="mañana",
        conversation_summary="",
        active_goal="appointment",
        stage="collecting_slots",
        pending_action="collecting_slots",
        pending_question="Necesito la fecha preferida.",
        appointment_slots={"reason": "dermatologia"},
        last_tool_result="",
        last_user_message="quiero una cita",
        last_assistant_message="",
        memories=[],
    )

    assert decision.next_node == "appointment"
    assert decision.intent == "appointment"


@pytest.mark.asyncio
async def test_router_routes_rag_requests_without_llm():
    service = build_service()

    decision = await service.route_state(
        user_message="Cuales son sus horarios y precios?",
        conversation_summary="",
        active_goal="conversation",
        stage="open",
        pending_action="",
        pending_question="",
        appointment_slots={},
        last_tool_result="",
        last_user_message="",
        last_assistant_message="",
        memories=[],
    )

    assert decision.next_node == "rag"
    assert decision.needs_retrieval is True


@pytest.mark.asyncio
async def test_router_uses_llm_when_no_guard_matches():
    decision = StateRoutingDecision(
        next_node="conversation",
        intent="conversation",
        confidence=0.77,
        needs_retrieval=False,
        state_update={"active_goal": "conversation"},
        reason="llm",
    )
    service = build_service(decision)

    routed = await service.route_state(
        user_message="Tengo una duda sobre algo especifico",
        conversation_summary="Seguimos hablando de otra cosa",
        active_goal="conversation",
        stage="open",
        pending_action="",
        pending_question="",
        appointment_slots={},
        last_tool_result="",
        last_user_message="",
        last_assistant_message="",
        memories=["Prefiere horario vespertino"],
    )

    assert routed.intent == "conversation"
    assert routed.confidence == pytest.approx(0.77)


def test_router_summarizes_memories_to_three_items():
    service = build_service()

    summarized = service.summarize_memories(
        [
            "Prefiere que le escriban por WhatsApp y atiende por la tarde.",
            "No puede recibir llamadas en horario laboral.",
            "Consulta anterior: dermatologia.",
            "Memoria extra que ya no debe entrar.",
        ]
    )

    assert len(summarized) == 3
    assert summarized[0].startswith("Prefiere que le escriban")
