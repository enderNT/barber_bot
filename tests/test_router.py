from app.services.router import ClinicIntentRouterService
from app.settings import Settings


def test_router_falls_back_to_appointment_intent_without_semantic_router():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    decision = service._fallback_route("Quiero agendar una cita con dermatologia")

    assert decision.intent == "appointment_intent"
    assert decision.confidence >= 0.7


def test_router_falls_back_to_rag_without_semantic_router():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    decision = service._fallback_route("Cuales son sus horarios?")

    assert decision.intent == "rag"
    assert decision.confidence >= 0.6


def test_router_falls_back_to_conversation_without_semantic_router():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    decision = service._fallback_route("Hola")

    assert decision.intent == "conversation"
    assert decision.confidence >= 0.5
