from __future__ import annotations

import logging
from typing import Any

from app.graph.workflow import ClinicWorkflow
from app.models.schemas import ChatwootWebhook
from app.observability.flow_logger import bind_flow, clear_flow, end_flow, mark_error, start_flow, step, substep
from app.services.chatwoot import ChatwootClient

logger = logging.getLogger(__name__)


class ClinicAgentService:
    def __init__(self, workflow: ClinicWorkflow, chatwoot_client: ChatwootClient) -> None:
        self._workflow = workflow
        self._chatwoot_client = chatwoot_client

    async def process_webhook(self, payload: ChatwootWebhook, flow_id: str | None = None) -> dict[str, Any]:
        bind_flow(flow_id or payload.conversation_id, payload.conversation_id)
        start_flow(payload.latest_message)
        try:
            step("2. state_routing_and_graph", "RUN", "ejecutando LangGraph")
            result = await self._workflow.run(payload)
            step(
                "2. state_routing_and_graph",
                "OK",
                f"intent={result.get('intent')} confidence={result.get('confidence', 0):.2f}",
            )

            response_text = result.get("response_text")
            if response_text:
                step("4. outbound_response", "RUN", "enviando respuesta a Chatwoot")
                try:
                    await self._chatwoot_client.send_message(
                        payload.conversation_id,
                        response_text,
                        account_id=payload.account_id,
                    )
                    step("4. outbound_response", "OK", "respuesta enviada o logueada")
                except Exception as exc:  # pragma: no cover - depende de Chatwoot
                    mark_error("4. outbound_response", exc)
                    logger.exception("Failed to send Chatwoot response for %s: %s", payload.conversation_id, exc)
            else:
                substep("4. outbound_response", "WARN", "sin response_text para enviar")

            end_flow("OK", f"branch={result.get('next_node', result.get('intent', 'unknown'))}")
            return result
        except Exception as exc:
            mark_error("flow_execution", exc)
            end_flow("ERROR")
            raise
        finally:
            clear_flow()
