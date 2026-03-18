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


def test_router_builds_enriched_input_with_memories_and_context():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    router_input = service._build_router_input(
        "Quiero una cita para manana",
        ["Prefiere horario vespertino", "Ya fue paciente de dermatologia"],
    )

    assert "Mensaje actual:" in router_input
    assert "Memoria relevante del usuario:" in router_input
    assert "Prefiere horario vespertino" in router_input


def test_router_input_ignores_clinic_context_for_all_messages():
    service = ClinicIntentRouterService(Settings(openai_api_key=None))

    router_input = service._build_router_input(
        "Cuales son sus horarios y que doctores atienden?",
        ["Pregunta seguido por medicina general"],
    )

    assert "Referencia clinica:" not in router_input
    assert "Servicios:" not in router_input
    assert "Doctores:" not in router_input
    assert "Horarios:" not in router_input
