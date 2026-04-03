from __future__ import annotations

import logging
import re
from typing import Any

from app.models.schemas import RoutingPacket, StateRoutingDecision
from app.observability.flow_logger import substep
from app.services.llm import BarbershopLLMService
from app.settings import Settings
from app.tracing import capture_trace_fragment

logger = logging.getLogger(__name__)


class StateRoutingService:
    def __init__(self, settings: Settings, llm_service: BarbershopLLMService) -> None:
        self._settings = settings
        self._llm_service = llm_service

    async def route_state(
        self,
        *,
        user_message: str,
        conversation_summary: str,
        active_goal: str,
        stage: str,
        pending_action: str,
        pending_question: str,
        booking_details: dict[str, Any],
        last_tool_result: str,
        last_user_message: str,
        last_assistant_message: str,
        memories: list[str],
    ) -> StateRoutingDecision:
        routing_packet = RoutingPacket(
            user_message=_compact_text(user_message, 400),
            conversation_summary=_compact_text(conversation_summary, 500),
            active_goal=_compact_text(active_goal, 80),
            stage=_compact_text(stage, 80),
            pending_action=_compact_text(pending_action, 120),
            pending_question=_compact_text(pending_question, 200),
            booking_details={key: _compact_text(str(value), 120) for key, value in booking_details.items()},
            last_tool_result=_compact_text(last_tool_result, 280),
            last_user_message=_compact_text(last_user_message, 280),
            last_assistant_message=_compact_text(last_assistant_message, 280),
            memories=[_compact_text(memory, 160) for memory in memories[:3]],
        )
        guard_hint = self._deterministic_guard(routing_packet)
        if guard_hint is not None:
            substep("state_router_guard", "OK", guard_hint.reason)
            capture_trace_fragment(
                "router_guard",
                {
                    "decision": guard_hint.model_dump(),
                    "reason": guard_hint.reason,
                },
                label="route_state",
            )
            return guard_hint

        decision = await self._llm_service.classify_state_route(routing_packet)
        if decision.next_node == "rag" and not decision.needs_retrieval:
            decision.needs_retrieval = True
        capture_trace_fragment(
            "router_llm_decision",
            {
                "decision": decision.model_dump(),
                "reason": decision.reason,
            },
            label="route_state",
        )
        substep(
            "state_router_llm",
            "OK",
            f"next={decision.next_node} intent={decision.intent} confidence={decision.confidence:.2f}",
        )
        return decision

    def summarize_memories(self, memories: list[str]) -> list[str]:
        summarized: list[str] = []
        for memory in memories[:3]:
            compact = _compact_text(memory, 140)
            if compact:
                summarized.append(compact)
        return summarized

    def _deterministic_guard(self, routing_packet: RoutingPacket) -> StateRoutingDecision | None:
        user_message = routing_packet.user_message.lower().strip()
        if not user_message:
            return StateRoutingDecision(
                next_node="conversation",
                intent="conversation",
                confidence=0.3,
                needs_retrieval=False,
                state_update={},
                reason="empty-message",
            )

        if self._booking_follow_up(routing_packet, user_message):
            return StateRoutingDecision(
                next_node="booking",
                intent="booking",
                confidence=0.95,
                needs_retrieval=False,
                state_update={
                    "active_goal": "booking",
                    "stage": "collecting_booking_details",
                    "pending_action": "collecting_booking_details",
                },
                reason="booking-follow-up",
            )

        if self._explicit_booking_request(user_message):
            return StateRoutingDecision(
                next_node="booking",
                intent="booking",
                confidence=0.92,
                needs_retrieval=False,
                state_update={
                    "active_goal": "booking",
                    "stage": "collecting_booking_details",
                    "pending_action": "collecting_booking_details",
                },
                reason="booking-request",
            )

        if self._explicit_rag_request(user_message):
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.86,
                needs_retrieval=True,
                state_update={
                    "active_goal": "information",
                    "stage": "lookup",
                },
                reason="information-request",
            )

        if self._is_simple_conversation(user_message):
            return StateRoutingDecision(
                next_node="conversation",
                intent="conversation",
                confidence=0.9,
                needs_retrieval=False,
                state_update={
                    "active_goal": routing_packet.active_goal or "conversation",
                    "stage": routing_packet.stage or "open",
                },
                reason="simple-conversation",
            )

        return None

    def _booking_follow_up(self, routing_packet: RoutingPacket, user_message: str) -> bool:
        active_booking = routing_packet.active_goal == "booking" or routing_packet.stage in {
            "collecting_booking_details",
            "ready_for_handoff",
        }
        if not active_booking:
            return False
        if routing_packet.pending_question:
            return True
        if routing_packet.booking_details and len(user_message) <= 40:
            return True
        return bool(
            re.search(
                r"\b(si|sí|no|claro|mañana|manana|hoy|tarde|noche|am|pm|con|barba|corte|fade|\d{1,2}:\d{2}|\d{1,2}\s?am|\d{1,2}\s?pm)\b",
                user_message,
            )
        )

    def _explicit_booking_request(self, user_message: str) -> bool:
        booking_keywords = (
            "cita",
            "agendar",
            "agendo",
            "reservar",
            "turno",
            "corte",
            "barba",
            "afeitado",
            "fade",
            "barbero",
            "paquete",
        )
        return any(keyword in user_message for keyword in booking_keywords)

    def _explicit_rag_request(self, user_message: str) -> bool:
        rag_keywords = (
            "horario",
            "horarios",
            "precio",
            "costo",
            "costos",
            "servicio",
            "servicios",
            "barbero",
            "barberos",
            "direccion",
            "ubicacion",
            "ubicación",
            "ubicados",
            "política",
            "politica",
            "pago",
            "pagos",
        )
        return any(keyword in user_message for keyword in rag_keywords)

    def _is_simple_conversation(self, user_message: str) -> bool:
        compact = " ".join(user_message.split())
        if compact in {"hola", "buenas", "buenos dias", "buenas tardes", "gracias", "ok", "okay", "si", "sí"}:
            return True
        if len(compact) <= 6:
            return True
        return any(marker in compact for marker in ("hola", "gracias", "perfecto", "entendido"))


def _compact_text(value: str, max_len: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
