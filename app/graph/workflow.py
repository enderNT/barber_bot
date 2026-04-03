from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.models.schemas import ChatwootWebhook
from app.observability.flow_logger import mark_error, step, substep
from app.services.barbershop_config import BarbershopConfigLoader
from app.services.llm import BarbershopLLMService
from app.services.memory import MemoryStore, should_store_memory
from app.services.qdrant import QdrantRetrievalService
from app.services.router import StateRoutingService
from app.settings import Settings
from app.tracing import NoopTraceSink, TraceContext, capture_trace_fragment
from app.tracing.barbershop import BarbershopTraceNormalizer, build_barbershop_field_policy
from app.tracing.types import TraceSink

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    conversation_id: str
    contact_id: str
    contact_name: str
    last_user_message: str
    last_assistant_message: str
    conversation_summary: str
    active_goal: str
    stage: str
    pending_action: str
    pending_question: str
    booking_details: dict[str, Any]
    last_tool_result: str
    memories: list[str]
    next_node: str
    intent: str
    confidence: float
    needs_retrieval: bool
    routing_reason: str
    state_update: dict[str, Any]
    response_text: str
    booking_payload: dict[str, Any]
    handoff_required: bool
    turn_count: int
    summary_refresh_requested: bool


class BarbershopWorkflow:
    def __init__(
        self,
        router_service: StateRoutingService,
        llm_service: BarbershopLLMService,
        memory_store: MemoryStore,
        barbershop_config_loader: BarbershopConfigLoader,
        qdrant_service: QdrantRetrievalService,
        settings: Settings,
        checkpointer: Any | None = None,
        store_backend: Any | None = None,
        trace_sink: TraceSink | None = None,
        trace_normalizer: BarbershopTraceNormalizer | None = None,
    ) -> None:
        self._router_service = router_service
        self._llm_service = llm_service
        self._memory_store = memory_store
        self._barbershop_config_loader = barbershop_config_loader
        self._qdrant_service = qdrant_service
        self._settings = settings
        self._checkpointer = checkpointer or MemorySaver()
        self._store_backend = store_backend
        self._trace_sink = trace_sink or NoopTraceSink()
        self._trace_normalizer = trace_normalizer or BarbershopTraceNormalizer()
        self._trace_field_policy = build_barbershop_field_policy()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("route", self._route)
        graph.add_node("conversation", self._conversation)
        graph.add_node("rag", self._rag)
        graph.add_node("booking", self._booking)
        graph.add_node("finalize_turn", self._finalize_turn)
        graph.add_node("store_memory", self._store_memory)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "route")
        graph.add_conditional_edges(
            "route",
            self._branch_after_route,
            {
                "conversation": "conversation",
                "rag": "rag",
                "booking": "booking",
            },
        )
        graph.add_edge("conversation", "finalize_turn")
        graph.add_edge("rag", "finalize_turn")
        graph.add_edge("booking", "finalize_turn")
        graph.add_edge("finalize_turn", "store_memory")
        graph.add_edge("store_memory", END)
        compile_kwargs: dict[str, Any] = {"checkpointer": self._checkpointer}
        if self._store_backend is not None:
            compile_kwargs["store"] = self._store_backend
        return graph.compile(**compile_kwargs)

    async def run(self, webhook: ChatwootWebhook) -> GraphState:
        trace_context = TraceContext(
            envelope=self._trace_normalizer.build_envelope(
                webhook,
                model_backend=self._settings.resolved_llm_provider,
                model_name=self._settings.resolved_llm_model,
                app_key=self._settings.tracer_app_key,
            ),
            sink=self._trace_sink,
            normalizer=self._trace_normalizer,
            field_policy=self._trace_field_policy,
        ).start()
        initial_state: GraphState = {
            "conversation_id": webhook.conversation_id,
            "contact_id": webhook.contact_id,
            "contact_name": webhook.contact_name,
            "last_user_message": webhook.latest_message,
        }
        config = {"configurable": {"thread_id": webhook.conversation_id}}
        trace_context.capture_input(webhook)
        try:
            result = await self._graph.ainvoke(initial_state, config=config)
            trace_context.capture_output(self._build_trace_output(result))
            await trace_context.finalize(
                "success",
                metrics_payload={
                    "fragment_count": len(result.get("booking_details", {})),
                    "turn_count": result.get("turn_count", 0),
                },
                tags={"route": result.get("next_node", "conversation")},
            )
            return result
        except Exception as exc:
            trace_context.capture_error(exc)
            await trace_context.finalize(
                "error",
                tags={"route": "error"},
            )
            raise
        finally:
            trace_context.detach()

    async def _load_context(self, state: GraphState) -> GraphState:
        try:
            step("2.1 build_context", "RUN", "cargando estado corto y memorias duraderas")
            memories = await self._memory_store.search(
                state["contact_id"],
                query=state.get("last_user_message") or state.get("conversation_summary") or "contexto del usuario",
                limit=self._settings.memory_search_limit,
            )
            capture_trace_fragment(
                "memory_lookup",
                {
                    "query": state.get("last_user_message") or state.get("conversation_summary") or "",
                    "memories_found": len(memories),
                },
                label="load_context",
            )
            substep("mem0_lookup", "OK", f"memories={len(memories)}")
            step("2.1 build_context", "OK")
            turn_count = int(state.get("turn_count", 0)) + 1
            return {
                "turn_count": turn_count,
                "memories": self._router_service.summarize_memories(memories),
            }
        except Exception as exc:
            mark_error("2.1 build_context", exc)
            raise

    async def _route(self, state: GraphState) -> GraphState:
        try:
            step("2.2 state_router", "RUN", "clasificando con estado compacto")
            decision = await self._router_service.route_state(
                user_message=state["last_user_message"],
                conversation_summary=state.get("conversation_summary", ""),
                active_goal=state.get("active_goal", ""),
                stage=state.get("stage", ""),
                pending_action=state.get("pending_action", ""),
                pending_question=state.get("pending_question", ""),
                booking_details=state.get("booking_details", {}),
                last_tool_result=state.get("last_tool_result", ""),
                last_user_message=state.get("last_user_message", ""),
                last_assistant_message=state.get("last_assistant_message", ""),
                memories=state.get("memories", []),
            )
            merged_state = self._apply_state_update(state, decision.state_update)
            merged_state.update(
                {
                    "next_node": decision.next_node,
                    "intent": decision.intent,
                    "confidence": decision.confidence,
                    "needs_retrieval": decision.needs_retrieval,
                    "routing_reason": decision.reason,
                    "state_update": decision.state_update,
                    "summary_refresh_requested": merged_state.get("summary_refresh_requested", False)
                    or merged_state.get("active_goal") != state.get("active_goal"),
                }
            )
            capture_trace_fragment(
                "routing_decision",
                {
                    "decision": {
                        "next_node": decision.next_node,
                        "intent": decision.intent,
                        "confidence": decision.confidence,
                        "needs_retrieval": decision.needs_retrieval,
                    },
                    "reason": decision.reason,
                },
                label="route",
            )
            step(
                "2.2 state_router",
                "OK",
                f"next={decision.next_node} intent={decision.intent} confidence={decision.confidence:.2f}",
            )
            return merged_state
        except Exception as exc:
            mark_error("2.2 state_router", exc)
            raise

    def _branch_after_route(self, state: GraphState) -> str:
        branch = state.get("next_node", "conversation")
        step("3. branch_selection", "OK", f"selected={branch}")
        if branch == "conversation":
            substep("3.a conversation", "OK", "usando nodo conversacional")
        elif branch == "rag":
            substep("3.b rag", "OK", "usando nodo RAG")
        elif branch == "booking":
            substep("3.c booking", "OK", "usando nodo de agendado")
        else:
            substep("3.x unknown_branch", "WARN", f"branch={branch}; fallback a conversation")
            return "conversation"
        return branch

    async def _conversation(self, state: GraphState) -> GraphState:
        try:
            step("3.a.1 conversation_node", "RUN", "generando respuesta")
            response_text = await self._llm_service.build_conversation_reply(
                user_message=state["last_user_message"],
                memories=state.get("memories", []),
            )
            capture_trace_fragment(
                "node_result",
                {"response_text": response_text, "next_node": "conversation"},
                label="conversation",
            )
            step("3.a.1 conversation_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "last_tool_result": "",
                "handoff_required": False,
                "booking_payload": {},
            }
        except Exception as exc:
            mark_error("3.a.1 conversation_node", exc)
            raise

    async def _rag(self, state: GraphState) -> GraphState:
        try:
            step("3.b.1 rag_node", "RUN", "consultando contexto RAG")
            barbershop_context = self._barbershop_config_loader.load().to_context_text()
            substep("barbershop_config", "OK", "config estatica cargada")
            rag_context = await self._qdrant_service.build_context(
                query=state["last_user_message"] or "contexto del usuario",
                contact_id=state["contact_id"],
                barbershop_context=barbershop_context,
                memories=state.get("memories", []),
            )
            substep("qdrant_lookup", "OK", "contexto vectorial preparado")
            capture_trace_fragment(
                "rag_context",
                {
                    "query": state["last_user_message"],
                    "barbershop_context_chars": len(rag_context),
                },
                label="rag",
            )
            response_text = await self._llm_service.build_rag_reply(
                user_message=state["last_user_message"],
                memories=state.get("memories", []),
                barbershop_context=rag_context,
            )
            step("3.b.1 rag_node", "OK", f"chars={len(response_text)}")
            return {
                "last_tool_result": _shorten(rag_context, 240),
                "response_text": response_text,
                "last_assistant_message": response_text,
                "handoff_required": False,
                "booking_payload": {},
            }
        except Exception as exc:
            mark_error("3.b.1 rag_node", exc)
            raise

    async def _booking(self, state: GraphState) -> GraphState:
        try:
            step("3.c.1 booking_node", "RUN", "extrayendo datos de cita")
            barbershop_context = self._barbershop_config_loader.load().to_context_text()
            substep("barbershop_config", "OK", "config estatica cargada")
            booking, response_text = await self._llm_service.extract_booking_intent(
                user_message=state["last_user_message"],
                memories=state.get("memories", []),
                barbershop_context=barbershop_context,
                contact_name=state["contact_name"],
                current_details=state.get("booking_details", {}),
                pending_question=state.get("pending_question"),
            )
            booking_details = _merge_booking_details(state.get("booking_details", {}), booking.model_dump())
            missing_fields = list(booking.missing_fields)
            pending_question = _build_pending_question(missing_fields) if missing_fields else ""
            stage = "collecting_booking_details" if missing_fields else "ready_for_handoff"
            pending_action = "collecting_booking_details" if missing_fields else ""
            if not missing_fields:
                response_text = (response_text + " " + "Tu solicitud quedo lista para recepcion.").strip()
            substep(
                "booking_payload",
                "OK",
                f"missing_fields={len(missing_fields)} handoff={booking.should_handoff}",
            )
            capture_trace_fragment(
                "booking_extraction",
                {
                    "booking_payload": booking.model_dump(),
                    "missing_fields": missing_fields,
                },
                label="booking",
            )
            step("3.c.1 booking_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "last_assistant_message": response_text,
                "booking_details": booking_details,
                "pending_question": pending_question,
                "pending_action": pending_action,
                "active_goal": "booking",
                "stage": stage,
                "last_tool_result": _shorten(
                    f"booking missing={','.join(missing_fields) or 'none'} confidence={booking.confidence:.2f}",
                    200,
                ),
                "handoff_required": booking.should_handoff,
                "booking_payload": booking.model_dump(),
            }
        except Exception as exc:
            mark_error("3.c.1 booking_node", exc)
            raise

    async def _finalize_turn(self, state: GraphState) -> GraphState:
        try:
            step("3.9 finalize_turn", "RUN", "limpiando estado y refrescando resumen si hace falta")
            cleaned_state = self._cleanup_state(state)
            if cleaned_state.get("summary_refresh_requested") or self._needs_summary_refresh(cleaned_state):
                summary = await self._llm_service.build_state_summary(
                    current_summary=cleaned_state.get("conversation_summary", ""),
                    user_message=cleaned_state.get("last_user_message", ""),
                    assistant_message=cleaned_state.get("last_assistant_message", ""),
                    active_goal=cleaned_state.get("active_goal", ""),
                    stage=cleaned_state.get("stage", ""),
                )
                cleaned_state["conversation_summary"] = _shorten(summary, 700)
                cleaned_state["summary_refresh_requested"] = False
            cleaned_state["turn_count"] = int(cleaned_state.get("turn_count", 0))
            step("3.9 finalize_turn", "OK", "estado limpio")
            capture_trace_fragment(
                "state_finalized",
                {
                    "active_goal": cleaned_state.get("active_goal", ""),
                    "stage": cleaned_state.get("stage", ""),
                    "turn_count": cleaned_state.get("turn_count", 0),
                },
                label="finalize_turn",
            )
            return cleaned_state
        except Exception as exc:
            mark_error("3.9 finalize_turn", exc)
            raise

    async def _store_memory(self, state: GraphState) -> GraphState:
        response_text = state.get("response_text", "")
        user_message = state.get("last_user_message", "")
        contact_id = state.get("contact_id")
        route = state.get("next_node", "conversation")
        if response_text and user_message and contact_id:
            memories = should_store_memory(user_message, response_text, route, state)
            if memories:
                step("3.10 store_memory", "RUN", f"persistiendo {len(memories)} memorias utiles")
                try:
                    await self._memory_store.save_memories(contact_id, memories)
                    capture_trace_fragment(
                        "memory_persisted",
                        {"memories_found": len(memories)},
                        label="store_memory",
                    )
                    step("3.10 store_memory", "OK")
                except Exception as exc:
                    mark_error("3.10 store_memory", exc)
                    raise
            else:
                substep("3.10 store_memory", "OK", "sin hechos duraderos para guardar")
        else:
            substep("3.10 store_memory", "WARN", "faltan campos para persistir")
        return {}

    def _apply_state_update(self, state: GraphState, patch: dict[str, Any]) -> GraphState:
        merged: GraphState = deepcopy(state)
        for key, value in patch.items():
            if key == "booking_details" and isinstance(value, dict):
                existing = merged.get(key, {})
                merged[key] = _merge_booking_details(existing if isinstance(existing, dict) else {}, value)
            else:
                merged[key] = value
        return merged

    def _cleanup_state(self, state: GraphState) -> GraphState:
        cleaned: GraphState = deepcopy(state)
        if cleaned.get("next_node") != "booking":
            cleaned["pending_action"] = ""
            cleaned["pending_question"] = ""
            cleaned["booking_details"] = {}
            if cleaned.get("stage") in {"collecting_booking_details", "ready_for_handoff"}:
                cleaned["stage"] = "open"
            if cleaned.get("active_goal") == "booking" and not cleaned.get("handoff_required", False):
                cleaned["active_goal"] = "conversation"
        if cleaned.get("next_node") != "rag":
            cleaned["last_tool_result"] = ""
        return cleaned

    def _needs_summary_refresh(self, state: GraphState) -> bool:
        summary = state.get("conversation_summary", "")
        turn_count = int(state.get("turn_count", 0))
        if len(summary) >= self._settings.summary_refresh_char_threshold:
            return True
        if turn_count and turn_count % self._settings.summary_refresh_turn_threshold == 0:
            return True
        if state.get("next_node") == "booking" and state.get("stage") == "ready_for_handoff":
            return True
        return False

    def _build_trace_output(self, state: GraphState) -> dict[str, Any]:
        return {
            "response_text": state.get("response_text", ""),
            "next_node": state.get("next_node", "conversation"),
            "intent": state.get("intent", "conversation"),
            "confidence": state.get("confidence", 0.0),
            "needs_retrieval": state.get("needs_retrieval", False),
            "handoff_required": state.get("handoff_required", False),
            "booking_payload": state.get("booking_payload", {}),
            "routing_reason": state.get("routing_reason", ""),
        }


def _merge_booking_details(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key in ("client_name", "service", "preferred_date", "preferred_time"):
        value = incoming.get(key)
        if value:
            merged[key] = value
    if incoming.get("missing_fields") is not None:
        merged["missing_fields"] = list(incoming.get("missing_fields") or [])
    if "confidence" in incoming:
        merged["confidence"] = incoming["confidence"]
    if "should_handoff" in incoming:
        merged["should_handoff"] = incoming["should_handoff"]
    return merged


def _build_pending_question(missing_fields: list[str]) -> str:
    field_names = {
        "client_name": "el nombre del cliente",
        "service": "el servicio",
        "preferred_date": "la fecha preferida",
        "preferred_time": "la hora preferida",
    }
    readable = [field_names.get(field, field) for field in missing_fields]
    if not readable:
        return ""
    if len(readable) == 1:
        return f"Necesito {readable[0]} para continuar."
    if len(readable) == 2:
        return f"Necesito {readable[0]} y {readable[1]} para continuar."
    return "Necesito " + ", ".join(readable[:-1]) + f" y {readable[-1]} para continuar."


def _shorten(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
