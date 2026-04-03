from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


JSONDict = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class TraceEnvelope:
    trace_id: str
    parent_trace_id: str | None = None
    session_key: str | None = None
    actor_key: str | None = None
    app_key: str | None = None
    flow_key: str | None = None
    dedupe_key: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    component_version: str | None = None
    model_backend: str | None = None
    model_name: str | None = None


@dataclass(slots=True)
class TraceFragment:
    kind: str
    order: int
    label: str = ""
    payload: JSONDict = field(default_factory=dict)
    latency_ms: int | None = None
    token_usage: JSONDict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class TraceRecord:
    envelope: TraceEnvelope
    input_payload: JSONDict = field(default_factory=dict)
    fragments: list[TraceFragment] = field(default_factory=list)
    output_payload: JSONDict = field(default_factory=dict)
    error_payload: JSONDict = field(default_factory=dict)
    metrics_payload: JSONDict = field(default_factory=dict)
    tags: JSONDict = field(default_factory=dict)
    extra_payload: JSONDict = field(default_factory=dict)
    completed_at: datetime | None = None
    outcome: str = "unknown"


@dataclass(slots=True)
class ProjectedExample:
    task_name: str
    projector_version: str
    input_payload: JSONDict
    target_payload: JSONDict
    metadata_payload: JSONDict = field(default_factory=dict)
    eligibility_reason: str = ""
    split: str = "train"
    quality_label: str | None = None
    created_at: datetime = field(default_factory=utc_now)


class FieldPolicy(Protocol):
    def sanitize(self, payload: Any) -> Any: ...


class TraceSink(Protocol):
    async def start(self) -> None: ...

    async def enqueue(self, trace_record: TraceRecord) -> None: ...

    async def close(self) -> None: ...


class TraceRepository(Protocol):
    async def setup(self) -> None: ...

    async def persist_batch(self, trace_records: list[TraceRecord]) -> None: ...

    async def close(self) -> None: ...


class TraceNormalizer(Protocol):
    def normalize_input(self, payload: Any) -> JSONDict: ...

    def normalize_output(self, payload: Any) -> JSONDict: ...

    def normalize_error(self, payload: Any) -> JSONDict: ...

    def normalize_fragment(self, kind: str, payload: Any) -> JSONDict: ...


class TraceProjector(Protocol):
    def project(self, trace_record: TraceRecord) -> list[ProjectedExample]: ...
