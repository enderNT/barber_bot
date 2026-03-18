from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.models.schemas import AppointmentIntentPayload, RoutingPacket, StateRoutingDecision
from app.observability.flow_logger import mark_error, step, substep
from app.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        client_kwargs: dict[str, Any] = {"timeout": settings.openai_timeout_seconds}
        client_kwargs["api_key"] = settings.openai_api_key or "sk-placeholder"
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url.rstrip("/")
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = settings.openai_model
        self._temperature = settings.openai_temperature

    def _chat_request_kwargs(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        selected_temperature = self._temperature if temperature is None else temperature
        if selected_temperature is not None and self._model_supports_temperature():
            request_kwargs["temperature"] = selected_temperature
        return request_kwargs

    def _model_supports_temperature(self) -> bool:
        normalized_model = self._model.strip().lower()
        unsupported_models = {"gpt-5"}
        unsupported_prefixes = ("gpt-5-mini", "gpt-5-nano", "gpt-5-")
        return normalized_model not in unsupported_models and not normalized_model.startswith(
            unsupported_prefixes
        )

    async def chat_text(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        step("2.2.1 openai_chat_completion", "RUN", f"model={self._model}")
        try:
            response = await self._client.chat.completions.create(
                **self._chat_request_kwargs(messages=messages, temperature=temperature),
            )
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 openai_chat_completion", "OK", f"response_chars={len(content)}")
            return content
        except Exception as exc:
            mark_error("2.2.1 openai_chat_completion", exc)
            raise

    async def chat_json(self, messages: list[dict[str, str]], temperature: float | None = None) -> dict[str, Any]:
        step("2.2.1 openai_chat_completion", "RUN", f"model={self._model} json_mode=True")
        try:
            request_kwargs = self._chat_request_kwargs(
                messages=messages, temperature=temperature
            )
            request_kwargs["response_format"] = {"type": "json_object"}
            response = await self._client.chat.completions.create(**request_kwargs)
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 openai_chat_completion", "OK", f"response_chars={len(content)}")
            return _extract_json(content)
        except Exception as exc:
            mark_error("2.2.1 openai_chat_completion", exc)
            raise


class ClinicLLMService:
    def __init__(self, client: OpenAIClient) -> None:
        self._client = client

    async def build_conversation_reply(self, user_message: str, memories: list[str]) -> str:
        system_prompt = (
            "Eres un asistente de una clinica. Responde de forma breve, clara y natural. "
            "Ayuda con la conversacion general y solo canaliza con recepcion si falta un dato."
        )
        user_prompt = (
            f"Memorias relevantes: {memories}\n"
            f"Pregunta del usuario: {user_message}\n"
            "Responde en espanol de forma breve y amigable."
        )
        try:
            substep("conversation_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            return await self._client.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("OpenAI conversation failed, using deterministic fallback: %s", exc)
            substep("conversation_fallback", "WARN", "mensaje deterministico")
            return (
                "Puedo ayudarte con informacion general de la clinica y con solicitudes de cita. "
                "Si tu pregunta depende de un dato no disponible, la canalizo con recepcion."
            )

    async def build_rag_reply(self, user_message: str, memories: list[str], clinic_context: str) -> str:
        system_prompt = (
            "Eres un asistente clinico en modo RAG. Usa solo el contexto entregado y no inventes informacion. "
            "Si falta informacion, dilo claramente y escala con recepcion."
        )
        user_prompt = (
            f"Contexto recuperado:\n{clinic_context}\n"
            f"Memoria conversacional: {memories}\n"
            f"Pregunta: {user_message}\n"
            "Responde breve y accionable en espanol."
        )
        try:
            substep("rag_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            return await self._client.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("OpenAI rag failed, using deterministic fallback: %s", exc)
            substep("rag_fallback", "WARN", "RAG degradado a respuesta segura")
            return (
                "Puedo responder con la informacion disponible de la clinica. "
                "Si necesitas un dato que no aparece en el contexto actual, lo canalizo con recepcion."
            )

    async def build_state_summary(
        self,
        current_summary: str,
        user_message: str,
        assistant_message: str,
        active_goal: str,
        stage: str,
    ) -> str:
        if not user_message.strip() and not assistant_message.strip():
            return current_summary

        system_prompt = (
            "Actualiza un resumen corto de estado conversacional. "
            "Mantente en una o dos frases. No repitas texto inutil."
        )
        user_prompt = (
            f"Resumen actual: {current_summary or 'n/a'}\n"
            f"Objetivo activo: {active_goal or 'n/a'}\n"
            f"Etapa: {stage or 'n/a'}\n"
            f"Ultimo mensaje del usuario: {user_message}\n"
            f"Ultima respuesta del asistente: {assistant_message}\n"
            "Devuelve solo el resumen actualizado."
        )
        try:
            substep("summary_prompt_compose", "OK", f"summary_chars={len(current_summary)}")
            return await self._client.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Summary refresh failed, using compact fallback: %s", exc)
            substep("summary_fallback", "WARN", "resumen compacto")
            fragments = [current_summary.strip(), f"Usuario: {user_message.strip()}", f"Asistente: {assistant_message.strip()}"]
            return " ".join(fragment for fragment in fragments if fragment).strip()

    async def classify_state_route(
        self,
        routing_packet: RoutingPacket,
        guard_hint: dict[str, Any] | None = None,
    ) -> StateRoutingDecision:
        system_prompt = (
            "Eres un clasificador de estado para un asistente de clinica. "
            "Debes devolver JSON estricto con next_node, intent, confidence, needs_retrieval, state_update y reason. "
            "Los valores permitidos para next_node son conversation, rag, appointment. "
            "Usa guards y el estado para decidir continuidad conversacional."
        )
        user_prompt = json.dumps(
            {
                "routing_packet": routing_packet.model_dump(),
                "guard_hint": guard_hint or {},
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            substep("state_router_prompt_compose", "OK", f"packet_chars={len(user_prompt)}")
            payload = await self._client.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            decision = StateRoutingDecision.model_validate(payload)
            substep(
                "state_router_json_parse",
                "OK",
                f"next_node={decision.next_node} confidence={decision.confidence:.2f}",
            )
            return decision
        except Exception as exc:
            logger.warning("State classification failed, using safe fallback: %s", exc)
            substep("state_router_fallback", "WARN", "clasificador degradado")
            return self._fallback_state_route(routing_packet, guard_hint or {})

    async def extract_appointment_intent(
        self,
        user_message: str,
        memories: list[str],
        clinic_context: str,
        contact_name: str,
        current_slots: dict[str, Any] | None = None,
        pending_question: str | None = None,
    ) -> tuple[AppointmentIntentPayload, str]:
        system_prompt = (
            "Extrae intencion de cita. Devuelve JSON estricto con llaves: "
            "patient_name, reason, preferred_date, preferred_time, missing_fields, should_handoff, confidence."
        )
        user_prompt = (
            f"Nombre de contacto: {contact_name}\n"
            f"Memorias relevantes: {memories}\n"
            f"Slots actuales: {current_slots or {}}\n"
            f"Pendiente: {pending_question or 'n/a'}\n"
            f"Contexto clinico:\n{clinic_context}\n"
            f"Mensaje: {user_message}\n"
            "Si faltan datos, listalos en missing_fields."
        )
        try:
            substep("appointment_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            payload = await self._client.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            appointment = AppointmentIntentPayload.model_validate(payload)
            substep("appointment_json_parse", "OK")
        except Exception as exc:
            logger.warning("OpenAI appointment extraction failed, using heuristic fallback: %s", exc)
            substep("appointment_fallback", "WARN", "extraccion heuristica")
            appointment = self._fallback_appointment(user_message, contact_name, current_slots=current_slots or {})
        reply = self._build_appointment_reply(appointment)
        return appointment, reply

    def _fallback_appointment(
        self, user_message: str, contact_name: str, current_slots: dict[str, Any] | None = None
    ) -> AppointmentIntentPayload:
        current_slots = current_slots or {}
        lowered = user_message.lower()
        reason = None
        for specialty in ("pediatria", "medicina general", "dermatologia", "ginecologia", "cardiologia"):
            if specialty in lowered:
                reason = specialty
                break
        date_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4}|manana|hoy|lunes|martes|miercoles|jueves|viernes|sabado)\b", lowered)
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm))\b", lowered)
        patient_name = (
            current_slots.get("patient_name")
            or (contact_name if contact_name and contact_name != "Paciente" else None)
        )
        reason = current_slots.get("reason") or reason
        preferred_date = current_slots.get("preferred_date") or (date_match.group(1) if date_match else None)
        preferred_time = current_slots.get("preferred_time") or (time_match.group(1) if time_match else None)
        missing_fields = []
        if not patient_name:
            missing_fields.append("patient_name")
        if not reason:
            missing_fields.append("reason")
        if not preferred_date:
            missing_fields.append("preferred_date")
        if not preferred_time:
            missing_fields.append("preferred_time")
        return AppointmentIntentPayload(
            patient_name=patient_name,
            reason=reason,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            missing_fields=missing_fields,
            should_handoff=True,
            confidence=0.65,
        )

    def _build_appointment_reply(self, appointment: AppointmentIntentPayload) -> str:
        if appointment.missing_fields:
            field_names = {
                "patient_name": "nombre del paciente",
                "reason": "motivo o especialidad",
                "preferred_date": "fecha preferida",
                "preferred_time": "hora preferida",
            }
            missing = ", ".join(field_names.get(field, field) for field in appointment.missing_fields)
            return (
                "Puedo dejar lista tu solicitud de cita. "
                f"Para continuar necesito: {missing}. "
                "En cuanto los compartas, genero el hand-off para recepcion."
            )
        return (
            "Ya tengo lo necesario para preparar tu solicitud de cita. "
            "La pasare a recepcion con el motivo y la preferencia de fecha/hora para confirmacion."
        )

    def _fallback_state_route(
        self, routing_packet: RoutingPacket, guard_hint: dict[str, Any]
    ) -> StateRoutingDecision:
        user_message = routing_packet.user_message.lower()
        if guard_hint.get("force_node") == "appointment":
            return StateRoutingDecision(
                next_node="appointment",
                intent="appointment",
                confidence=0.88,
                needs_retrieval=False,
                state_update=guard_hint.get("state_update", {}),
                reason="guard-hint",
            )
        if guard_hint.get("force_node") == "rag":
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.84,
                needs_retrieval=True,
                state_update=guard_hint.get("state_update", {}),
                reason="guard-hint",
            )
        if any(word in user_message for word in ("cita", "agendar", "reservar", "consulta", "turno")):
            return StateRoutingDecision(
                next_node="appointment",
                intent="appointment",
                confidence=0.74,
                needs_retrieval=False,
                state_update={"active_goal": "appointment", "stage": "collecting_slots"},
                reason="heuristic-fallback",
            )
        if any(word in user_message for word in ("horario", "precio", "costo", "servicio", "doctor", "especialidad")):
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.66,
                needs_retrieval=True,
                state_update={"active_goal": "information", "stage": "lookup"},
                reason="heuristic-fallback",
            )
        return StateRoutingDecision(
            next_node="conversation",
            intent="conversation",
            confidence=0.58,
            needs_retrieval=False,
            state_update={"active_goal": routing_packet.active_goal or "conversation", "stage": routing_packet.stage or "open"},
            reason="heuristic-fallback",
        )


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
