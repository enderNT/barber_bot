from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BookingIntentPayload(BaseModel):
    client_name: str | None = None
    service: str | None = None
    preferred_date: str | None = None
    preferred_time: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    should_handoff: bool = True
    confidence: float = 0.0


class RoutingPacket(BaseModel):
    user_message: str
    conversation_summary: str = ""
    active_goal: str = ""
    stage: str = ""
    pending_action: str = ""
    pending_question: str = ""
    booking_details: dict[str, Any] = Field(default_factory=dict)
    last_tool_result: str = ""
    last_user_message: str = ""
    last_assistant_message: str = ""
    memories: list[str] = Field(default_factory=list)


class StateRoutingDecision(BaseModel):
    next_node: Literal["conversation", "rag", "booking"] = "conversation"
    intent: str = "conversation"
    confidence: float = 0.0
    needs_retrieval: bool = False
    state_update: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class RouteStateUpdate(BaseModel):
    active_goal: str | None = None
    stage: str | None = None
    pending_action: str | None = None
    pending_question: str | None = None
    booking_details: dict[str, Any] | None = None
    conversation_summary: str | None = None
    last_tool_result: str | None = None
    last_user_message: str | None = None
    last_assistant_message: str | None = None
    turn_count: int | None = None
    clear_last_tool_result: bool = False
    clear_pending_action: bool = False
    clear_pending_question: bool = False
    clear_booking_details: bool = False


class MemoryRecord(BaseModel):
    kind: Literal["profile", "episode"]
    text: str
    source: str = "stateful-flow"


class GraphTurnOutcome(BaseModel):
    next_node: Literal["conversation", "rag", "booking"] = "conversation"
    response_text: str = ""
    intent: str = "conversation"
    confidence: float = 0.0
    needs_retrieval: bool = False
    handoff_required: bool = False
    booking_payload: dict[str, Any] = Field(default_factory=dict)
    routing_reason: str = ""


class BarbershopConfig(BaseModel):
    barbershop_name: str
    timezone: str
    services: list[dict[str, Any]] = Field(default_factory=list)
    barbers: list[dict[str, Any]] = Field(default_factory=list)
    hours: dict[str, str] = Field(default_factory=dict)
    policies: dict[str, str] = Field(default_factory=dict)

    def to_context_text(self) -> str:
        services = "\n".join(
            f"- {service.get('name')}: {service.get('duration_minutes', 'N/D')} min, {service.get('price', 'N/D')}"
            for service in self.services
        )
        barbers = "\n".join(
            f"- {barber.get('name')} ({', '.join(barber.get('services', [])) or 'Sin servicios'}): {barber.get('availability_notes', 'Sin nota')}"
            for barber in self.barbers
        )
        hours = "\n".join(f"- {day}: {schedule}" for day, schedule in self.hours.items())
        policies = "\n".join(f"- {name}: {value}" for name, value in self.policies.items())
        return (
            f"Barberia: {self.barbershop_name}\n"
            f"Zona horaria: {self.timezone}\n"
            f"Servicios:\n{services or '- Sin servicios'}\n"
            f"Barberos:\n{barbers or '- Sin barberos'}\n"
            f"Horarios:\n{hours or '- Sin horarios'}\n"
            f"Politicas:\n{policies or '- Sin politicas'}"
        )


class ChatwootWebhook(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: str | None = None
    content: str | None = None
    message_type: str | None = None
    account: dict[str, Any] = Field(default_factory=dict)
    conversation: dict[str, Any] = Field(default_factory=dict)
    contact: dict[str, Any] = Field(default_factory=dict)
    sender: dict[str, Any] = Field(default_factory=dict)
    inbox: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    additional_attributes: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def conversation_id(self) -> str:
        raw = self.conversation.get("id") or self.additional_attributes.get("conversation_id") or "unknown-conversation"
        return str(raw)

    @property
    def account_id(self) -> str | None:
        raw = (
            self.account.get("id")
            or self.conversation.get("account_id")
            or self.additional_attributes.get("account_id")
            or self.meta.get("account", {}).get("id")
            or next(
                (
                    message.get("account_id")
                    for message in reversed(self.messages)
                    if message.get("account_id")
                ),
                None,
            )
        )
        return str(raw) if raw is not None else None

    @property
    def is_incoming_message_event(self) -> bool:
        return (
            (self.event or "").strip().lower() == "message_created"
            and (self.message_type or "").strip().lower() == "incoming"
        )

    @property
    def contact_id(self) -> str:
        raw = self.contact.get("id") or self.sender.get("id") or self.meta.get("sender", {}).get("id") or "unknown-contact"
        return str(raw)

    @property
    def contact_name(self) -> str:
        return (
            self.contact.get("name")
            or self.sender.get("name")
            or self.meta.get("sender", {}).get("name")
            or "Cliente"
        )

    @property
    def latest_message(self) -> str:
        if self.content:
            return self.content.strip()
        for message in reversed(self.messages):
            content = message.get("content")
            if content:
                return str(content).strip()
        return ""
