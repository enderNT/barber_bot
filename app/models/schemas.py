from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AppointmentIntentPayload(BaseModel):
    patient_name: str | None = None
    reason: str | None = None
    preferred_date: str | None = None
    preferred_time: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    should_handoff: bool = True
    confidence: float = 0.0


class IntentDecision(BaseModel):
    intent: Literal["conversation", "rag", "appointment_intent"] = "conversation"
    confidence: float = 0.0
    reason: str = ""


class ClinicConfig(BaseModel):
    clinic_name: str
    timezone: str
    services: list[dict[str, Any]] = Field(default_factory=list)
    doctors: list[dict[str, Any]] = Field(default_factory=list)
    hours: dict[str, str] = Field(default_factory=dict)
    policies: dict[str, str] = Field(default_factory=dict)

    def to_context_text(self) -> str:
        services = "\n".join(
            f"- {service.get('name')}: {service.get('duration_minutes', 'N/D')} min, {service.get('price', 'N/D')}"
            for service in self.services
        )
        doctors = "\n".join(
            f"- {doctor.get('name')} ({doctor.get('specialty')}): {doctor.get('availability_notes', 'Sin nota')}"
            for doctor in self.doctors
        )
        hours = "\n".join(f"- {day}: {schedule}" for day, schedule in self.hours.items())
        policies = "\n".join(f"- {name}: {value}" for name, value in self.policies.items())
        return (
            f"Clinica: {self.clinic_name}\n"
            f"Zona horaria: {self.timezone}\n"
            f"Servicios:\n{services or '- Sin servicios'}\n"
            f"Doctores:\n{doctors or '- Sin doctores'}\n"
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
    def contact_id(self) -> str:
        raw = self.contact.get("id") or self.sender.get("id") or self.meta.get("sender", {}).get("id") or "unknown-contact"
        return str(raw)

    @property
    def contact_name(self) -> str:
        return (
            self.contact.get("name")
            or self.sender.get("name")
            or self.meta.get("sender", {}).get("name")
            or "Paciente"
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
