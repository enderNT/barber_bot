from __future__ import annotations

import json
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.tracing.policy import NoopFieldPolicy
from app.tracing.types import FieldPolicy, ProjectedExample, TraceProjector, TraceRecord, TraceRepository


class AsyncPostgresTraceRepository(TraceRepository):
    def __init__(
        self,
        dsn: str,
        *,
        projectors: list[TraceProjector] | None = None,
        field_policy: FieldPolicy | None = None,
    ) -> None:
        self._pool = AsyncConnectionPool(conninfo=dsn, open=False)
        self._projectors = projectors or []
        self._field_policy = field_policy or NoopFieldPolicy()
        self._setup_done = False

    async def setup(self) -> None:
        if self._setup_done:
            return
        await self._pool.open()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trace_turns (
                        trace_id TEXT PRIMARY KEY,
                        parent_trace_id TEXT,
                        session_key TEXT,
                        actor_key TEXT,
                        app_key TEXT,
                        flow_key TEXT,
                        dedupe_key TEXT UNIQUE,
                        started_at TIMESTAMPTZ NOT NULL,
                        completed_at TIMESTAMPTZ,
                        outcome TEXT NOT NULL,
                        component_version TEXT,
                        model_backend TEXT,
                        model_name TEXT,
                        has_error BOOLEAN NOT NULL DEFAULT FALSE,
                        projector_summary TEXT,
                        input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        output_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        error_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        metrics_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        tags JSONB NOT NULL DEFAULT '{}'::jsonb,
                        extra_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                    );
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trace_fragments (
                        trace_id TEXT NOT NULL REFERENCES trace_turns(trace_id) ON DELETE CASCADE,
                        "order" INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        label TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL,
                        latency_ms INTEGER,
                        token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (trace_id, "order")
                    );
                    """
                )
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trace_examples (
                        trace_id TEXT NOT NULL REFERENCES trace_turns(trace_id) ON DELETE CASCADE,
                        task_name TEXT NOT NULL,
                        projector_version TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        split TEXT NOT NULL DEFAULT 'train',
                        quality_label TEXT,
                        eligibility_reason TEXT NOT NULL DEFAULT '',
                        input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        target_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        metadata_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (trace_id, task_name, projector_version)
                    );
                    """
                )
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_turns_session_key ON trace_turns(session_key);")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_turns_actor_key ON trace_turns(actor_key);")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_turns_flow_key ON trace_turns(flow_key);")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_turns_started_at ON trace_turns(started_at);")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_turns_outcome ON trace_turns(outcome);")
            await conn.commit()
        self._setup_done = True

    async def persist_batch(self, trace_records: list[TraceRecord]) -> None:
        if not trace_records:
            return
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                for trace_record in trace_records:
                    trace_id = await self._upsert_trace_turn(cur, trace_record)
                    for fragment in trace_record.fragments:
                        await cur.execute(
                            """
                            INSERT INTO trace_fragments (
                                trace_id,
                                "order",
                                kind,
                                label,
                                created_at,
                                latency_ms,
                                token_usage,
                                payload
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (trace_id, "order") DO UPDATE SET
                                kind = EXCLUDED.kind,
                                label = EXCLUDED.label,
                                created_at = EXCLUDED.created_at,
                                latency_ms = EXCLUDED.latency_ms,
                                token_usage = EXCLUDED.token_usage,
                                payload = EXCLUDED.payload
                            """,
                            (
                                trace_id,
                                fragment.order,
                                fragment.kind,
                                fragment.label,
                                fragment.created_at,
                                fragment.latency_ms,
                                Jsonb(fragment.token_usage),
                                Jsonb(fragment.payload),
                            ),
                        )
                    for example in self._build_examples(trace_record):
                        await self._upsert_trace_example(cur, trace_id, example)
            await conn.commit()

    async def close(self) -> None:
        await self._pool.close()

    async def _upsert_trace_turn(self, cur: Any, trace_record: TraceRecord) -> str:
        summary = self._projector_summary(trace_record)
        envelope = trace_record.envelope
        has_error = bool(trace_record.error_payload) or trace_record.outcome == "error"
        params = (
            envelope.trace_id,
            envelope.parent_trace_id,
            envelope.session_key,
            envelope.actor_key,
            envelope.app_key,
            envelope.flow_key,
            envelope.dedupe_key,
            envelope.started_at,
            trace_record.completed_at,
            trace_record.outcome,
            envelope.component_version,
            envelope.model_backend,
            envelope.model_name,
            has_error,
            summary,
            Jsonb(trace_record.input_payload),
            Jsonb(trace_record.output_payload),
            Jsonb(trace_record.error_payload),
            Jsonb(trace_record.metrics_payload),
            Jsonb(trace_record.tags),
            Jsonb(trace_record.extra_payload),
        )
        if envelope.dedupe_key:
            await cur.execute(
                """
                INSERT INTO trace_turns (
                    trace_id,
                    parent_trace_id,
                    session_key,
                    actor_key,
                    app_key,
                    flow_key,
                    dedupe_key,
                    started_at,
                    completed_at,
                    outcome,
                    component_version,
                    model_backend,
                    model_name,
                    has_error,
                    projector_summary,
                    input_payload,
                    output_payload,
                    error_payload,
                    metrics_payload,
                    tags,
                    extra_payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dedupe_key) DO UPDATE SET
                    completed_at = EXCLUDED.completed_at,
                    outcome = EXCLUDED.outcome,
                    component_version = EXCLUDED.component_version,
                    model_backend = EXCLUDED.model_backend,
                    model_name = EXCLUDED.model_name,
                    has_error = EXCLUDED.has_error,
                    projector_summary = EXCLUDED.projector_summary,
                    input_payload = EXCLUDED.input_payload,
                    output_payload = EXCLUDED.output_payload,
                    error_payload = EXCLUDED.error_payload,
                    metrics_payload = EXCLUDED.metrics_payload,
                    tags = EXCLUDED.tags,
                    extra_payload = EXCLUDED.extra_payload
                RETURNING trace_id
                """,
                params,
            )
            row = await cur.fetchone()
            return row["trace_id"]

        await cur.execute(
            """
            INSERT INTO trace_turns (
                trace_id,
                parent_trace_id,
                session_key,
                actor_key,
                app_key,
                flow_key,
                dedupe_key,
                started_at,
                completed_at,
                outcome,
                component_version,
                model_backend,
                model_name,
                has_error,
                projector_summary,
                input_payload,
                output_payload,
                error_payload,
                metrics_payload,
                tags,
                extra_payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trace_id) DO UPDATE SET
                completed_at = EXCLUDED.completed_at,
                outcome = EXCLUDED.outcome,
                component_version = EXCLUDED.component_version,
                model_backend = EXCLUDED.model_backend,
                model_name = EXCLUDED.model_name,
                has_error = EXCLUDED.has_error,
                projector_summary = EXCLUDED.projector_summary,
                input_payload = EXCLUDED.input_payload,
                output_payload = EXCLUDED.output_payload,
                error_payload = EXCLUDED.error_payload,
                metrics_payload = EXCLUDED.metrics_payload,
                tags = EXCLUDED.tags,
                extra_payload = EXCLUDED.extra_payload
            """,
            params,
        )
        return envelope.trace_id

    async def _upsert_trace_example(self, cur: Any, trace_id: str, example: ProjectedExample) -> None:
        await cur.execute(
            """
            INSERT INTO trace_examples (
                trace_id,
                task_name,
                projector_version,
                created_at,
                split,
                quality_label,
                eligibility_reason,
                input_payload,
                target_payload,
                metadata_payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trace_id, task_name, projector_version) DO UPDATE SET
                created_at = EXCLUDED.created_at,
                split = EXCLUDED.split,
                quality_label = EXCLUDED.quality_label,
                eligibility_reason = EXCLUDED.eligibility_reason,
                input_payload = EXCLUDED.input_payload,
                target_payload = EXCLUDED.target_payload,
                metadata_payload = EXCLUDED.metadata_payload
            """,
            (
                trace_id,
                example.task_name,
                example.projector_version,
                example.created_at,
                example.split,
                example.quality_label,
                example.eligibility_reason,
                Jsonb(self._sanitize_json(example.input_payload)),
                Jsonb(self._sanitize_json(example.target_payload)),
                Jsonb(self._sanitize_json(example.metadata_payload)),
            ),
        )

    def _build_examples(self, trace_record: TraceRecord) -> list[ProjectedExample]:
        examples: list[ProjectedExample] = []
        for projector in self._projectors:
            examples.extend(projector.project(trace_record))
        return examples

    def _sanitize_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        sanitized = self._field_policy.sanitize(payload)
        if not isinstance(sanitized, dict):
            return {"value": json.loads(json.dumps(sanitized))}
        return sanitized

    def _projector_summary(self, trace_record: TraceRecord) -> str:
        if not self._projectors:
            return "projectors-disabled"
        names = [type(projector).__name__ for projector in self._projectors]
        return ",".join(names[:8])
