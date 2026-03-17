from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.schemas import ChatwootWebhook
from app.observability.flow_logger import mark_error, step, substep
from app.services.clinic_config import ClinicConfigLoader
from app.services.llm import ClinicLLMService
from app.services.memory import MemoryStore
from app.services.router import ClinicIntentRouterService
from app.services.qdrant import QdrantRetrievalService

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    webhook: ChatwootWebhook
    conversation_id: str
    contact_id: str
    contact_name: str
    user_message: str
    clinic_context: str
    memories: list[str]
    rag_context: str
    intent: str
    routing_reason: str
    routing_confidence: float
    response_text: str
    appointment_payload: dict[str, Any]
    handoff_required: bool


class ClinicWorkflow:
    def __init__(
        self,
        router_service: ClinicIntentRouterService,
        llm_service: ClinicLLMService,
        memory_store: MemoryStore,
        clinic_config_loader: ClinicConfigLoader,
        qdrant_service: QdrantRetrievalService,
    ) -> None:
        self._router_service = router_service
        self._llm_service = llm_service
        self._memory_store = memory_store
        self._clinic_config_loader = clinic_config_loader
        self._qdrant_service = qdrant_service
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("route", self._route)
        graph.add_node("conversation", self._conversation)
        graph.add_node("rag", self._rag)
        graph.add_node("appointment", self._appointment)
        graph.add_node("store_memory", self._store_memory)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "route")
        graph.add_conditional_edges(
            "route",
            self._branch_after_route,
            {
                "conversation": "conversation",
                "rag": "rag",
                "appointment_intent": "appointment",
            },
        )
        graph.add_edge("conversation", "store_memory")
        graph.add_edge("rag", "store_memory")
        graph.add_edge("appointment", "store_memory")
        graph.add_edge("store_memory", END)
        return graph.compile()

    async def run(self, webhook: ChatwootWebhook) -> GraphState:
        initial_state: GraphState = {"webhook": webhook}
        return await self._graph.ainvoke(initial_state)

    async def _load_context(self, state: GraphState) -> GraphState:
        webhook = state["webhook"]
        try:
            step("2.1 build_context", "RUN", "cargando config clinica y memoria")
            clinic_context = self._clinic_config_loader.load().to_context_text()
            substep("clinic_config", "OK", "config estatica cargada")
            memories = await self._memory_store.search(
                webhook.contact_id,
                query=webhook.latest_message or "contexto del usuario",
                limit=5,
            )
            substep("mem0_lookup", "OK", f"memories={len(memories)}")
            rag_context = await self._qdrant_service.build_context(
                query=webhook.latest_message or "contexto del usuario",
                contact_id=webhook.contact_id,
                clinic_context=clinic_context,
                memories=memories,
            )
            substep("qdrant_lookup", "OK", "contexto vectorial simulado/preparado")
            step("2.1 build_context", "OK")
            return {
                "conversation_id": webhook.conversation_id,
                "contact_id": webhook.contact_id,
                "contact_name": webhook.contact_name,
                "user_message": webhook.latest_message,
                "clinic_context": clinic_context,
                "memories": memories,
                "rag_context": rag_context,
            }
        except Exception as exc:
            mark_error("2.1 build_context", exc)
            raise

    async def _route(self, state: GraphState) -> GraphState:
        try:
            step("2.2 intent_router_openai", "RUN", "clasificando intent")
            decision = await self._router_service.route_intent(
                user_message=state["user_message"],
                memories=state.get("memories", []),
                clinic_context=state["clinic_context"],
            )
            step(
                "2.2 intent_router_openai",
                "OK",
                f"intent={decision.intent} confidence={decision.confidence:.2f} reason={decision.reason}",
            )
            return {
                "intent": decision.intent,
                "routing_reason": decision.reason,
                "routing_confidence": decision.confidence,
            }
        except Exception as exc:
            mark_error("2.2 intent_router_openai", exc)
            raise

    def _branch_after_route(self, state: GraphState) -> str:
        branch = state.get("intent", "conversation")
        step("3. branch_selection", "OK", f"selected={branch}")
        if branch == "conversation":
            substep("3.a conversation", "OK", "usando nodo conversacional")
        elif branch == "rag":
            substep("3.b rag", "OK", "usando nodo RAG")
        elif branch == "appointment_intent":
            substep("3.c appointment", "OK", "usando nodo de agendado")
        else:
            substep("3.x unknown_branch", "WARN", f"branch={branch}; fallback a conversation")
            return "conversation"
        return branch

    async def _conversation(self, state: GraphState) -> GraphState:
        try:
            step("3.a.1 conversation_node", "RUN", "generando respuesta")
            response_text = await self._llm_service.build_conversation_reply(
                user_message=state["user_message"],
                memories=state.get("memories", []),
                clinic_context=state["clinic_context"],
            )
            step("3.a.1 conversation_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "handoff_required": False,
                "appointment_payload": {},
            }
        except Exception as exc:
            mark_error("3.a.1 conversation_node", exc)
            raise

    async def _rag(self, state: GraphState) -> GraphState:
        try:
            step("3.b.1 rag_node", "RUN", "preparando contexto qdrant simulado")
            response_text = await self._llm_service.build_rag_reply(
                user_message=state["user_message"],
                memories=state.get("memories", []),
                clinic_context=state.get("rag_context", state["clinic_context"]),
            )
            step("3.b.1 rag_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "handoff_required": False,
                "appointment_payload": {},
            }
        except Exception as exc:
            mark_error("3.b.1 rag_node", exc)
            raise

    async def _appointment(self, state: GraphState) -> GraphState:
        try:
            step("3.c.1 appointment_node", "RUN", "extrayendo datos de cita")
            appointment, response_text = await self._llm_service.extract_appointment_intent(
                user_message=state["user_message"],
                memories=state.get("memories", []),
                clinic_context=state["clinic_context"],
                contact_name=state["contact_name"],
            )
            substep(
                "appointment_payload",
                "OK",
                f"missing_fields={len(appointment.missing_fields)} handoff={appointment.should_handoff}",
            )
            step("3.c.1 appointment_node", "OK", f"chars={len(response_text)}")
            return {
                "response_text": response_text,
                "handoff_required": True,
                "appointment_payload": appointment.model_dump(),
            }
        except Exception as exc:
            mark_error("3.c.1 appointment_node", exc)
            raise

    async def _store_memory(self, state: GraphState) -> GraphState:
        response_text = state.get("response_text")
        user_message = state.get("user_message")
        contact_id = state.get("contact_id")
        if response_text and user_message and contact_id:
            step("3.9 store_memory", "RUN", "persistiendo exchange")
            try:
                await self._memory_store.save_exchange(contact_id, user_message, response_text)
                step("3.9 store_memory", "OK")
            except Exception as exc:
                mark_error("3.9 store_memory", exc)
                raise
        else:
            substep("3.9 store_memory", "WARN", "faltan campos para persistir")
        return {}
