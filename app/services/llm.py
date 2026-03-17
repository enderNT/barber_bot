from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.models.schemas import AppointmentIntentPayload, IntentDecision
from app.observability.flow_logger import mark_error, step, substep
from app.settings import Settings

logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._model = settings.vllm_model
        self._api_key = settings.vllm_api_key
        self._timeout = settings.llm_timeout_seconds
        self._temperature = settings.llm_temperature

    async def _chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        step("2.2.1 vllm_call", "RUN", f"model={self._model}")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            step("2.2.1 vllm_call", "OK", f"response_chars={len(content)}")
            return content
        except Exception as exc:
            mark_error("2.2.1 vllm_call", exc)
            raise

    async def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        return await self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    async def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = await self.chat_text(system_prompt, user_prompt)
        return _extract_json(content)


class ClinicLLMService:
    def __init__(self, client: VLLMClient) -> None:
        self._client = client

    async def route_intent(self, user_message: str, memories: list[str], clinic_context: str) -> IntentDecision:
        system_prompt = (
            "Clasifica el mensaje del usuario. "
            "Devuelve JSON estricto con las llaves intent, confidence y reason. "
            "intent debe ser 'conversation', 'rag' o 'appointment_intent'. "
            "Usa 'rag' cuando la pregunta requiera buscar detalles concretos de conocimiento institucional."
        )
        user_prompt = (
            f"Mensaje: {user_message}\n"
            f"Memorias: {memories}\n"
            f"Contexto clinico:\n{clinic_context}\n"
            "Si el usuario quiere agendar, reservar o pedir una cita, clasifica como appointment_intent."
        )
        try:
            substep("router_prompt_compose", "OK", f"msg_chars={len(user_message)} memories={len(memories)}")
            payload = await self._client.chat_json(system_prompt, user_prompt)
            substep("router_json_parse", "OK")
            return IntentDecision.model_validate(payload)
        except Exception as exc:
            logger.warning("LLM routing failed, using heuristic fallback: %s", exc)
            substep("router_fallback", "WARN", "heuristic intent detection")
            return self._fallback_route(user_message)

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
            return await self._client.chat_text(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("LLM conversation failed, using deterministic fallback: %s", exc)
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
            return await self._client.chat_text(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("LLM rag failed, using deterministic fallback: %s", exc)
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
            payload = await self._client.chat_json(system_prompt, user_prompt)
            appointment = AppointmentIntentPayload.model_validate(payload)
            substep("appointment_json_parse", "OK")
        except Exception as exc:
            logger.warning("LLM appointment extraction failed, using heuristic fallback: %s", exc)
            substep("appointment_fallback", "WARN", "extraccion heuristica")
            appointment = self._fallback_appointment(user_message, contact_name)
        reply = self._build_appointment_reply(appointment)
        return appointment, reply

    def _fallback_route(self, user_message: str) -> IntentDecision:
        rag_keywords = ("precio", "costo", "horario", "horarios", "doctor", "especialidad", "servicio")
        if any(word in user_message.lower() for word in rag_keywords):
            return IntentDecision(intent="rag", confidence=0.60, reason="heuristic-rag-fallback")
        keywords = ("cita", "agendar", "agendo", "reservar", "consulta", "doctor", "doctora")
        intent = "appointment_intent" if any(word in user_message.lower() for word in keywords) else "conversation"
        confidence = 0.72 if intent == "appointment_intent" else 0.55
        return IntentDecision(intent=intent, confidence=confidence, reason="heuristic-fallback")

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
