from __future__ import annotations

import contextvars
from dataclasses import replace
from typing import Any

from app.tracing.policy import NoopFieldPolicy
from app.tracing.types import (
    FieldPolicy,
    TraceEnvelope,
    TraceFragment,
    TraceNormalizer,
    TraceRecord,
    TraceSink,
    utc_now,
)

_current_trace_context: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "current_trace_context",
    default=None,
)


class TraceContext:
    def __init__(
        self,
        *,
        envelope: TraceEnvelope,
        sink: TraceSink,
        normalizer: TraceNormalizer,
        field_policy: FieldPolicy | None = None,
    ) -> None:
        self._envelope = envelope
        self._sink = sink
        self._normalizer = normalizer
        self._field_policy = field_policy or NoopFieldPolicy()
        self._input_payload: dict[str, Any] = {}
        self._fragments: list[TraceFragment] = []
        self._output_payload: dict[str, Any] = {}
        self._error_payload: dict[str, Any] = {}
        self._metrics_payload: dict[str, Any] = {}
        self._tags: dict[str, Any] = {}
        self._extra_payload: dict[str, Any] = {}
        self._token: contextvars.Token[TraceContext | None] | None = None
        self._finalized = False

    @property
    def trace_id(self) -> str:
        return self._envelope.trace_id

    @property
    def envelope(self) -> TraceEnvelope:
        return self._envelope

    def start(self, envelope: TraceEnvelope | None = None) -> TraceContext:
        if envelope is not None:
            self._envelope = envelope
        self._token = _current_trace_context.set(self)
        return self

    def capture_input(self, payload: Any) -> None:
        self._input_payload = self._sanitize(self._normalizer.normalize_input(payload))

    def capture_fragment(
        self,
        kind: str,
        payload: Any,
        order: int | None = None,
        *,
        label: str = "",
        latency_ms: int | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        fragment = TraceFragment(
            kind=kind,
            order=order or len(self._fragments) + 1,
            label=label,
            payload=self._sanitize(self._normalizer.normalize_fragment(kind, payload)),
            latency_ms=latency_ms,
            token_usage=self._sanitize(token_usage or {}),
            created_at=utc_now(),
        )
        self._fragments.append(fragment)

    def capture_output(self, payload: Any) -> None:
        self._output_payload = self._sanitize(self._normalizer.normalize_output(payload))

    def capture_error(self, error_payload: Any) -> None:
        self._error_payload = self._sanitize(self._normalizer.normalize_error(error_payload))

    async def finalize(
        self,
        outcome: str,
        *,
        metrics_payload: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> TraceRecord:
        if self._finalized:
            raise RuntimeError("TraceContext already finalized.")
        self._finalized = True
        self._metrics_payload = self._sanitize(metrics_payload or {})
        self._tags = self._sanitize(tags or {})
        self._extra_payload = self._sanitize(extra_payload or {})
        record = TraceRecord(
            envelope=replace(self._envelope),
            input_payload=dict(self._input_payload),
            fragments=list(self._fragments),
            output_payload=dict(self._output_payload),
            error_payload=dict(self._error_payload),
            metrics_payload=dict(self._metrics_payload),
            tags=dict(self._tags),
            extra_payload=dict(self._extra_payload),
            completed_at=utc_now(),
            outcome=outcome,
        )
        await self._sink.enqueue(record)
        return record

    def detach(self) -> None:
        if self._token is not None:
            _current_trace_context.reset(self._token)
            self._token = None

    def _sanitize(self, payload: Any) -> Any:
        sanitized = self._field_policy.sanitize(payload)
        return sanitized if isinstance(sanitized, dict) else {"value": sanitized} if sanitized is not None else {}


def get_current_trace_context() -> TraceContext | None:
    return _current_trace_context.get()


def capture_trace_fragment(
    kind: str,
    payload: Any,
    *,
    label: str = "",
    latency_ms: int | None = None,
    token_usage: dict[str, Any] | None = None,
) -> None:
    trace_context = get_current_trace_context()
    if trace_context is None:
        return
    trace_context.capture_fragment(
        kind,
        payload,
        label=label,
        latency_ms=latency_ms,
        token_usage=token_usage,
    )
