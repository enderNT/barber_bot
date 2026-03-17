from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.models.schemas import AppointmentIntentPayload
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

    async def chat_text(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        step("2.2.1 openai_chat_completion", "RUN", f"model={self._model}")
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature if temperature is None else temperature,
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
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature if temperature is None else temperature,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            step("2.2.1 openai_chat_completion", "OK", f"response_chars={len(content)}")
            return _extract_json(content)
        except Exception as exc:
            mark_error("2.2.1 openai_chat_completion", exc)
            raise


class ClinicLLMService:
    def __init__(self, client: OpenAIClient) -> None:
        self._client = client

    async def build_conversation_reply(self, user_message: str, memories: list[str], clinic_context: str) -> str:
        system_prompt = (
            "Eres un asistente de una clinica. Responde solo con datos del contexto. "
            "Si no sabes algo, dilo claramente y ofrece canalizar con recepcion. "
            "No inventes precios, horarios ni disponibilidad."
        )
        user_prompt = (
            f"Memorias relevantes: {memories}\n"
            f"Contexto clinico:\n{clinic_context}\n"
            f"Pregunta del usuario: {user_message}\n"
            "Responde en espanol de forma breve, clara y operativa."
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

    async def extract_appointment_intent(
        self, user_message: str, memories: list[str], clinic_context: str, contact_name: str
    ) -> tuple[AppointmentIntentPayload, str]:
        system_prompt = (
            "Extrae intencion de cita. Devuelve JSON estricto con llaves: "
            "patient_name, reason, preferred_date, preferred_time, missing_fields, should_handoff, confidence."
        )
        user_prompt = (
            f"Nombre de contacto: {contact_name}\n"
            f"Memorias relevantes: {memories}\n"
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
            appointment = self._fallback_appointment(user_message, contact_name)
        reply = self._build_appointment_reply(appointment)
        return appointment, reply

    def _fallback_appointment(self, user_message: str, contact_name: str) -> AppointmentIntentPayload:
        lowered = user_message.lower()
        reason = None
        for specialty in ("pediatria", "medicina general", "dermatologia", "ginecologia", "cardiologia"):
            if specialty in lowered:
                reason = specialty
                break
        date_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4}|manana|hoy|lunes|martes|miercoles|jueves|viernes|sabado)\b", lowered)
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm))\b", lowered)
        patient_name = contact_name if contact_name and contact_name != "Paciente" else None
        missing_fields = []
        if not patient_name:
            missing_fields.append("patient_name")
        if not reason:
            missing_fields.append("reason")
        if not date_match:
            missing_fields.append("preferred_date")
        if not time_match:
            missing_fields.append("preferred_time")
        return AppointmentIntentPayload(
            patient_name=patient_name,
            reason=reason,
            preferred_date=date_match.group(1) if date_match else None,
            preferred_time=time_match.group(1) if time_match else None,
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


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
