from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Protocol, TypedDict

from openai import AsyncOpenAI

from app.models.schemas import BookingIntentPayload, RoutingPacket, StateRoutingDecision
from app.observability.flow_logger import mark_error, step, substep
from app.settings import Settings

logger = logging.getLogger(__name__)


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def chat_text(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> str: ...

    async def chat_json(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> dict[str, Any]: ...


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings) -> None:
        client_kwargs: dict[str, Any] = {"timeout": settings.resolved_llm_timeout_seconds}
        client_kwargs["api_key"] = settings.resolved_llm_api_key or "sk-placeholder"
        if settings.resolved_llm_base_url:
            client_kwargs["base_url"] = settings.resolved_llm_base_url.rstrip("/")
        self._client = AsyncOpenAI(**client_kwargs)
        self._provider_name = settings.resolved_llm_provider
        self._model = settings.resolved_llm_model
        self._temperature = settings.resolved_llm_temperature

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    def _chat_request_kwargs(
        self, messages: list[LLMMessage], temperature: float | None = None
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

    async def chat_text(self, messages: list[LLMMessage], temperature: float | None = None) -> str:
        step(
            "2.2.1 llm_chat_completion",
            "RUN",
            f"provider={self.provider_name} model={self.model_name}",
        )
        try:
            response = await self._client.chat.completions.create(
                **self._chat_request_kwargs(messages=messages, temperature=temperature),
            )
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 llm_chat_completion", "OK", f"response_chars={len(content)}")
            return content
        except Exception as exc:
            mark_error("2.2.1 llm_chat_completion", exc)
            raise

    async def chat_json(self, messages: list[LLMMessage], temperature: float | None = None) -> dict[str, Any]:
        step(
            "2.2.1 llm_chat_completion",
            "RUN",
            f"provider={self.provider_name} model={self.model_name} json_mode=True",
        )
        request_kwargs = self._chat_request_kwargs(messages=messages, temperature=temperature)
        request_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 llm_chat_completion", "OK", f"response_chars={len(content)}")
            return _extract_json(content)
        except Exception as exc:
            if _should_retry_with_json_schema(exc):
                substep("llm_json_schema_retry", "WARN", "fallback a response_format=json_schema")
                response = await self._client.chat.completions.create(
                    **self._json_schema_request_kwargs(messages=messages, temperature=temperature),
                )
                content = (response.choices[0].message.content or "").strip()
                step("2.2.1 llm_chat_completion", "OK", f"response_chars={len(content)}")
                return _extract_json(content)
            mark_error("2.2.1 llm_chat_completion", exc)
            raise

    def _json_schema_request_kwargs(
        self, messages: list[LLMMessage], temperature: float | None = None
    ) -> dict[str, Any]:
        request_kwargs = self._chat_request_kwargs(messages=messages, temperature=temperature)
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        }
        return request_kwargs


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider_name = settings.resolved_llm_provider
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    raise ValueError(f"Unsupported llm provider: {provider_name}")


class BarbershopLLMService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def build_conversation_reply(self, user_message: str, memories: list[str]) -> str:
        system_prompt = (
            "Eres un asistente de una barberia. Responde de forma breve, clara y natural. "
            "Ayuda con conversacion general, servicios y citas. Si falta un dato operativo, canaliza con recepcion."
        )
        user_prompt = (
            f"Memorias relevantes: {memories}\n"
            f"Pregunta del usuario: {user_message}\n"
            "Responde en espanol de forma breve y amigable."
        )
        try:
            substep("conversation_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            return await self._provider.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("LLM conversation failed, using deterministic fallback: %s", exc)
            substep("conversation_fallback", "WARN", "mensaje deterministico")
            return (
                "Puedo ayudarte con informacion general de la barberia y con solicitudes de cita. "
                "Si tu pregunta depende de un dato no disponible, la canalizo con recepcion."
            )

    async def build_rag_reply(self, user_message: str, memories: list[str], barbershop_context: str) -> str:
        system_prompt = (
            "Eres un asistente de barberia en modo RAG. Usa solo el contexto entregado y no inventes informacion. "
            "Si falta informacion, dilo claramente y escala con recepcion."
        )
        user_prompt = (
            f"Contexto recuperado:\n{barbershop_context}\n"
            f"Memoria conversacional: {memories}\n"
            f"Pregunta: {user_message}\n"
            "Responde breve y accionable en espanol."
        )
        try:
            substep("rag_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            return await self._provider.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("LLM rag failed, using deterministic fallback: %s", exc)
            substep("rag_fallback", "WARN", "RAG degradado a respuesta segura")
            return (
                "Puedo responder con la informacion disponible de la barberia. "
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
            return await self._provider.chat_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Summary refresh failed, using compact fallback: %s", exc)
            substep("summary_fallback", "WARN", "resumen compacto")
            fragments = [
                current_summary.strip(),
                f"Usuario: {user_message.strip()}",
                f"Asistente: {assistant_message.strip()}",
            ]
            return " ".join(fragment for fragment in fragments if fragment).strip()

    async def classify_state_route(
        self,
        routing_packet: RoutingPacket,
        guard_hint: dict[str, Any] | None = None,
    ) -> StateRoutingDecision:
        system_prompt = (
            "Eres un clasificador de estado para un asistente de barberia. "
            "Debes devolver JSON estricto con next_node, intent, confidence, needs_retrieval, state_update y reason. "
            "Los valores permitidos para next_node son conversation, rag, booking. "
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
            payload = await self._provider.chat_json(
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

    async def extract_booking_intent(
        self,
        user_message: str,
        memories: list[str],
        barbershop_context: str,
        contact_name: str,
        current_details: dict[str, Any] | None = None,
        pending_question: str | None = None,
    ) -> tuple[BookingIntentPayload, str]:
        system_prompt = (
            "Extrae intencion de cita para barberia. Devuelve JSON estricto con llaves: "
            "client_name, service, preferred_date, preferred_time, missing_fields, should_handoff, confidence. "
            "Solo pide esos cuatro campos."
        )
        user_prompt = (
            f"Nombre de contacto: {contact_name}\n"
            f"Memorias relevantes: {memories}\n"
            f"Datos actuales: {current_details or {}}\n"
            f"Pendiente: {pending_question or 'n/a'}\n"
            f"Contexto de la barberia:\n{barbershop_context}\n"
            f"Mensaje: {user_message}\n"
            "Si faltan datos, listalos en missing_fields."
        )
        try:
            substep("booking_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            payload = await self._provider.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            booking = BookingIntentPayload.model_validate(payload)
            substep("booking_json_parse", "OK")
        except Exception as exc:
            logger.warning("LLM booking extraction failed, using heuristic fallback: %s", exc)
            substep("booking_fallback", "WARN", "extraccion heuristica")
            booking = self._fallback_booking(
                user_message,
                contact_name,
                current_details=current_details or {},
            )
        reply = self._build_booking_reply(booking)
        return booking, reply

    def _fallback_booking(
        self, user_message: str, contact_name: str, current_details: dict[str, Any] | None = None
    ) -> BookingIntentPayload:
        current_details = current_details or {}
        lowered = user_message.lower()
        service = None
        for known_service in (
            "corte y barba",
            "afeitado",
            "afeitado tradicional",
            "barba",
            "fade",
            "corte clasico",
            "corte",
            "paquete",
        ):
            if known_service in lowered:
                service = known_service
                break
        date_match = re.search(
            r"\b(\d{1,2}/\d{1,2}/\d{2,4}|mañana|manana|hoy|lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado)\b",
            lowered,
        )
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm))\b", lowered)
        client_name = (
            current_details.get("client_name")
            or (contact_name if contact_name and contact_name != "Cliente" else None)
        )
        service = current_details.get("service") or service
        preferred_date = current_details.get("preferred_date") or (date_match.group(1) if date_match else None)
        preferred_time = current_details.get("preferred_time") or (time_match.group(1) if time_match else None)
        missing_fields = []
        if not client_name:
            missing_fields.append("client_name")
        if not service:
            missing_fields.append("service")
        if not preferred_date:
            missing_fields.append("preferred_date")
        if not preferred_time:
            missing_fields.append("preferred_time")
        return BookingIntentPayload(
            client_name=client_name,
            service=service,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            missing_fields=missing_fields,
            should_handoff=True,
            confidence=0.65,
        )

    def _build_booking_reply(self, booking: BookingIntentPayload) -> str:
        if booking.missing_fields:
            field_names = {
                "client_name": "nombre del cliente",
                "service": "servicio",
                "preferred_date": "fecha preferida",
                "preferred_time": "hora preferida",
            }
            missing = ", ".join(field_names.get(field, field) for field in booking.missing_fields)
            return (
                "Puedo dejar lista tu solicitud de cita. "
                f"Para continuar necesito: {missing}. "
                "En cuanto los compartas, genero el hand-off para recepcion."
            )
        return (
            "Ya tengo lo necesario para preparar tu solicitud de cita. "
            "La pasare a recepcion con el servicio y la preferencia de fecha/hora para confirmacion."
        )

    def _fallback_state_route(
        self, routing_packet: RoutingPacket, guard_hint: dict[str, Any]
    ) -> StateRoutingDecision:
        user_message = routing_packet.user_message.lower()
        if guard_hint.get("force_node") == "booking":
            return StateRoutingDecision(
                next_node="booking",
                intent="booking",
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
        if any(
            word in user_message
            for word in ("cita", "agendar", "reservar", "turno", "corte", "barba", "afeitado", "fade", "paquete")
        ):
            return StateRoutingDecision(
                next_node="booking",
                intent="booking",
                confidence=0.74,
                needs_retrieval=False,
                state_update={"active_goal": "booking", "stage": "collecting_booking_details"},
                reason="heuristic-fallback",
            )
        if any(
            word in user_message
            for word in (
                "horario",
                "precio",
                "costo",
                "servicio",
                "servicios",
                "barbero",
                "barberos",
                "ubicacion",
                "ubicación",
                "pago",
                "pagos",
            )
        ):
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
            state_update={
                "active_goal": routing_packet.active_goal or "conversation",
                "stage": routing_packet.stage or "open",
            },
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


def _should_retry_with_json_schema(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format.type" in message and "json_schema" in message
