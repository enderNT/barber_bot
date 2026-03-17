from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ChatwootWebhook
from app.observability.flow_logger import new_flow_id, step, substep
from app.services.agent import ClinicAgentService

logger = logging.getLogger(__name__)


def build_webhook_router(agent_service: ClinicAgentService) -> APIRouter:
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    @router.post("/chatwoot", status_code=status.HTTP_202_ACCEPTED)
    async def chatwoot_webhook(payload: ChatwootWebhook) -> dict[str, str]:
        flow_id = new_flow_id()
        step(
            "1. webhook_received",
            "OK",
            f"flow={flow_id} conv={payload.conversation_id} contact={payload.contact_id}",
        )
        substep("webhook_payload_validated", "RUN", "validando contenido util")
        if not payload.latest_message:
            substep("webhook_payload_validated", "ERROR", "sin contenido procesable")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Webhook sin contenido util para procesar.",
            )
        substep("webhook_payload_validated", "OK", "contenido detectado")
        substep("webhook_background_dispatch", "RUN", "enviando al flujo asyncrono")
        asyncio.create_task(_safe_process(agent_service, payload, flow_id))
        substep("webhook_background_dispatch", "OK", "task creada")
        return {"status": "accepted", "conversation_id": payload.conversation_id}

    return router


async def _safe_process(agent_service: ClinicAgentService, payload: ChatwootWebhook, flow_id: str) -> None:
    try:
        await agent_service.process_webhook(payload, flow_id=flow_id)
    except Exception as exc:  # pragma: no cover - logging defensivo
        logger.exception("Background webhook processing failed for %s: %s", payload.conversation_id, exc)
