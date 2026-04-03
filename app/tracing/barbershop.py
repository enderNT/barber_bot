from __future__ import annotations

import uuid
from typing import Any

from app.models.schemas import ChatwootWebhook
from app.tracing.policy import AllowlistRedactionPolicy
from app.tracing.types import ProjectedExample, TraceEnvelope, TraceNormalizer, TraceProjector, TraceRecord


class BarbershopTraceNormalizer(TraceNormalizer):
    def normalize_input(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, ChatwootWebhook):
            return {
                "conversation_id": payload.conversation_id,
                "contact_id": payload.contact_id,
                "contact_name": payload.contact_name,
                "message": payload.latest_message,
                "event": payload.event,
                "message_type": payload.message_type,
            }
        return _normalize_payload(payload)

    def normalize_output(self, payload: Any) -> dict[str, Any]:
        return _normalize_payload(payload)

    def normalize_error(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, Exception):
            return {"error_type": type(payload).__name__, "message": str(payload)}
        return _normalize_payload(payload)

    def normalize_fragment(self, kind: str, payload: Any) -> dict[str, Any]:
        normalized = _normalize_payload(payload)
        normalized.setdefault("kind", kind)
        return normalized

    def build_envelope(self, webhook: ChatwootWebhook, *, model_backend: str, model_name: str, app_key: str) -> TraceEnvelope:
        return TraceEnvelope(
            trace_id=uuid.uuid4().hex,
            session_key=webhook.conversation_id,
            actor_key=webhook.contact_id,
            app_key=app_key,
            flow_key="chatwoot_webhook",
            dedupe_key=webhook.dedupe_key,
            component_version="0.1.0",
            model_backend=model_backend,
            model_name=model_name,
        )


class BarbershopTraceProjector(TraceProjector):
    version = "barbershop-v1"

    def project(self, trace_record: TraceRecord) -> list[ProjectedExample]:
        if trace_record.outcome != "success":
            return []
        user_message = trace_record.input_payload.get("message")
        assistant_message = trace_record.output_payload.get("response_text")
        if not user_message or not assistant_message:
            return []
        task_name = f"reply_{trace_record.output_payload.get('next_node', 'conversation')}"
        return [
            ProjectedExample(
                task_name=task_name,
                projector_version=self.version,
                input_payload={"message": user_message, "context": trace_record.input_payload},
                target_payload={"response_text": assistant_message},
                metadata_payload={
                    "trace_id": trace_record.envelope.trace_id,
                    "intent": trace_record.output_payload.get("intent"),
                    "outcome": trace_record.outcome,
                },
                eligibility_reason="success-with-response",
            )
        ]


def build_barbershop_field_policy() -> AllowlistRedactionPolicy:
    return AllowlistRedactionPolicy(
        allowed_keys={
            "conversation_id",
            "contact_id",
            "contact_name",
            "message",
            "event",
            "message_type",
            "response_text",
            "next_node",
            "intent",
            "confidence",
            "needs_retrieval",
            "handoff_required",
            "booking_payload",
            "routing_reason",
            "active_goal",
            "stage",
            "pending_action",
            "pending_question",
            "booking_details",
            "last_tool_result",
            "turn_count",
            "summary_refresh_requested",
            "kind",
            "query",
            "memories_found",
            "decision",
            "reason",
            "provider",
            "model",
            "operation",
            "message_count",
            "response_chars",
            "json_mode",
            "fallback",
            "error_type",
            "message_preview",
            "barbershop_context_chars",
            "missing_fields",
            "projectors",
            "flow_id",
            "latency_ms",
            "route",
            "fragment_count",
        }
    )


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, Exception):
        return {"error_type": type(payload).__name__, "message": str(payload)}
    return {"value": payload}
