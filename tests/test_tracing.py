import asyncio

from app.graph.workflow import BarbershopWorkflow
from app.models.schemas import BookingIntentPayload, ChatwootWebhook, StateRoutingDecision
from app.services.barbershop_config import BarbershopConfigLoader
from app.services.router import StateRoutingService
from app.settings import Settings
from app.tracing import AsyncBatchTraceSink, BarbershopTraceNormalizer, BarbershopTraceProjector, TraceContext
from app.tracing.barbershop import build_barbershop_field_policy
from app.tracing.types import ProjectedExample, TraceEnvelope, TraceRecord, TraceRepository


class RecordingRepository(TraceRepository):
    def __init__(self, *, projectors=None, field_policy=None):
        self.projectors = projectors or []
        self.field_policy = field_policy or build_barbershop_field_policy()
        self.batches: list[list[TraceRecord]] = []
        self.records: list[TraceRecord] = []
        self.examples: list[ProjectedExample] = []
        self.started = False
        self.closed = False

    async def setup(self) -> None:
        self.started = True

    async def persist_batch(self, trace_records: list[TraceRecord]) -> None:
        self.batches.append(list(trace_records))
        self.records.extend(trace_records)
        for trace_record in trace_records:
            for projector in self.projectors:
                for example in projector.project(trace_record):
                    example.input_payload = self.field_policy.sanitize(example.input_payload)
                    example.target_payload = self.field_policy.sanitize(example.target_payload)
                    example.metadata_payload = self.field_policy.sanitize(example.metadata_payload)
                    self.examples.append(example)

    async def close(self) -> None:
        self.closed = True


class FakeLLMService:
    async def classify_state_route(self, routing_packet, guard_hint=None):
        del guard_hint
        message = routing_packet.user_message.lower()
        if "horario" in message:
            return StateRoutingDecision(
                next_node="rag",
                intent="rag",
                confidence=0.85,
                needs_retrieval=True,
                state_update={"active_goal": "information", "stage": "lookup"},
                reason="test",
            )
        return StateRoutingDecision(
            next_node="conversation",
            intent="conversation",
            confidence=0.8,
            needs_retrieval=False,
            state_update={"active_goal": "conversation", "stage": "open"},
            reason="test",
        )

    async def build_conversation_reply(self, user_message, memories):
        del memories
        return f"Respuesta para: {user_message}"

    async def build_rag_reply(self, user_message, memories, barbershop_context):
        del memories, barbershop_context
        return f"RAG para: {user_message}"

    async def extract_booking_intent(
        self, user_message, memories, barbershop_context, contact_name, current_details=None, pending_question=None
    ):
        del user_message, memories, barbershop_context, contact_name, current_details, pending_question
        payload = BookingIntentPayload(
            client_name="Juan Perez",
            service="corte",
            preferred_date="manana",
            preferred_time="10 am",
            missing_fields=[],
            should_handoff=True,
            confidence=0.9,
        )
        return payload, "Solicitud lista"

    async def build_state_summary(self, current_summary, user_message, assistant_message, active_goal, stage):
        return f"{current_summary} | {active_goal}:{stage} | {user_message} -> {assistant_message}".strip(" |")


class FakeMemoryStore:
    async def search(self, contact_id, query, limit=5):
        del contact_id, query, limit
        return ["Recuerdo util"]

    async def save_memories(self, contact_id, memories):
        del contact_id, memories


class FakeQdrantService:
    async def build_context(self, *args, **kwargs):
        del args, kwargs
        return "Contexto RAG simulado"


def build_webhook(message: str, conversation_id: int = 123) -> ChatwootWebhook:
    return ChatwootWebhook(
        content=message,
        conversation={"id": conversation_id},
        contact={"id": 456, "name": "Juan Perez"},
        event="message_created",
        message_type="incoming",
        additional_attributes={"message_id": f"msg-{conversation_id}"},
    )


async def _build_sink(*, batch_size=10, flush_interval=5.0, projectors=None):
    repository = RecordingRepository(projectors=projectors, field_policy=build_barbershop_field_policy())
    sink = AsyncBatchTraceSink(
        repository,
        batch_size=batch_size,
        flush_interval_seconds=flush_interval,
    )
    await sink.start()
    return sink, repository


def test_trace_context_persists_successful_turn_and_projected_example():
    async def scenario():
        sink, repository = await _build_sink(projectors=[BarbershopTraceProjector()])
        normalizer = BarbershopTraceNormalizer()
        trace_context = TraceContext(
            envelope=TraceEnvelope(
                trace_id="trace-1",
                session_key="session-1",
                actor_key="actor-1",
                app_key="barbershop",
                flow_key="chatwoot_webhook",
                dedupe_key="msg-1",
            ),
            sink=sink,
            normalizer=normalizer,
            field_policy=build_barbershop_field_policy(),
        ).start()

        trace_context.capture_input(
            {
                "message": "Quiero una cita",
                "conversation_id": "321",
                "contact_id": "654",
                "api_key": "super-secret",
            }
        )
        trace_context.capture_fragment("routing_decision", {"reason": "booking", "token": "private"})
        trace_context.capture_output(
            {"response_text": "Claro, te ayudo", "next_node": "conversation", "intent": "booking"}
        )

        await trace_context.finalize("success", tags={"route": "conversation"})
        trace_context.detach()
        await sink.close()
        return repository

    repository = asyncio.run(scenario())

    assert repository.started is True
    assert repository.closed is True
    assert len(repository.records) == 1
    assert "api_key" not in repository.records[0].input_payload
    assert repository.records[0].fragments[0].order == 1
    assert "token" not in repository.records[0].fragments[0].payload
    assert len(repository.examples) == 1
    assert "api_key" not in repository.examples[0].input_payload["context"]


def test_trace_context_persists_failed_turn_without_examples():
    async def scenario():
        sink, repository = await _build_sink(projectors=[BarbershopTraceProjector()])
        trace_context = TraceContext(
            envelope=TraceEnvelope(trace_id="trace-2"),
            sink=sink,
            normalizer=BarbershopTraceNormalizer(),
            field_policy=build_barbershop_field_policy(),
        ).start()

        trace_context.capture_input({"message": "hola"})
        trace_context.capture_error(RuntimeError("boom"))
        await trace_context.finalize("error")
        trace_context.detach()
        await sink.close()
        return repository

    repository = asyncio.run(scenario())

    assert len(repository.records) == 1
    assert repository.records[0].error_payload["error_type"] == "RuntimeError"
    assert repository.examples == []


def test_batch_sink_flushes_by_batch_size():
    async def scenario():
        sink, repository = await _build_sink(batch_size=2, flush_interval=10.0)
        normalizer = BarbershopTraceNormalizer()

        for trace_id in ("trace-a", "trace-b"):
            trace_context = TraceContext(
                envelope=TraceEnvelope(trace_id=trace_id),
                sink=sink,
                normalizer=normalizer,
                field_policy=build_barbershop_field_policy(),
            ).start()
            trace_context.capture_input({"message": trace_id})
            await trace_context.finalize("success")
            trace_context.detach()

        await asyncio.sleep(0.1)
        await sink.close()
        return repository

    repository = asyncio.run(scenario())

    assert len(repository.batches) >= 1
    assert len(repository.batches[0]) == 2


def test_batch_sink_flushes_by_interval_and_shutdown():
    async def scenario():
        sink, repository = await _build_sink(batch_size=10, flush_interval=0.05)
        trace_context = TraceContext(
            envelope=TraceEnvelope(trace_id="trace-interval"),
            sink=sink,
            normalizer=BarbershopTraceNormalizer(),
            field_policy=build_barbershop_field_policy(),
        ).start()
        trace_context.capture_input({"message": "interval"})
        await trace_context.finalize("success")
        trace_context.detach()

        await asyncio.sleep(0.12)

        second = TraceContext(
            envelope=TraceEnvelope(trace_id="trace-shutdown"),
            sink=sink,
            normalizer=BarbershopTraceNormalizer(),
            field_policy=build_barbershop_field_policy(),
        ).start()
        second.capture_input({"message": "shutdown"})
        await second.finalize("success")
        second.detach()
        await sink.close()
        return repository

    repository = asyncio.run(scenario())
    assert {record.envelope.trace_id for record in repository.records} == {"trace-interval", "trace-shutdown"}


def test_projectors_can_be_disabled_without_disabling_base_capture():
    async def scenario():
        sink, repository = await _build_sink(projectors=[])
        trace_context = TraceContext(
            envelope=TraceEnvelope(trace_id="trace-3"),
            sink=sink,
            normalizer=BarbershopTraceNormalizer(),
            field_policy=build_barbershop_field_policy(),
        ).start()
        trace_context.capture_input({"message": "hola"})
        trace_context.capture_output({"response_text": "hola", "next_node": "conversation"})
        await trace_context.finalize("success")
        trace_context.detach()
        await sink.close()
        return repository

    repository = asyncio.run(scenario())

    assert len(repository.records) == 1
    assert repository.examples == []


def test_workflow_emits_trace_records_with_fragments():
    async def scenario():
        sink, repository = await _build_sink(batch_size=1, projectors=[BarbershopTraceProjector()])
        llm = FakeLLMService()
        settings = Settings(_env_file=None, memory_backend="in_memory", checkpoint_backend="memory")
        workflow = BarbershopWorkflow(
            StateRoutingService(settings, llm),
            llm,
            FakeMemoryStore(),
            BarbershopConfigLoader(config_path="config/barbershop.json"),  # type: ignore[arg-type]
            FakeQdrantService(),
            settings,
            trace_sink=sink,
            trace_normalizer=BarbershopTraceNormalizer(),
        )

        result = await workflow.run(build_webhook("Necesito informacion general"))
        await sink.close()
        return result, repository

    result, repository = asyncio.run(scenario())

    assert result["response_text"] == "Respuesta para: Necesito informacion general"
    assert len(repository.records) == 1
    assert repository.records[0].output_payload["response_text"] == result["response_text"]
    fragment_kinds = [fragment.kind for fragment in repository.records[0].fragments]
    assert "memory_lookup" in fragment_kinds
    assert "routing_decision" in fragment_kinds
    assert "node_result" in fragment_kinds
