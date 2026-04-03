from __future__ import annotations

import contextvars
import logging
import re
import time
import uuid

logger = logging.getLogger("barbershop.flow")

_flow_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("flow_id", default="-")
_conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("conversation_id", default="-")
_flow_started_at_var: contextvars.ContextVar[float] = contextvars.ContextVar("flow_started_at", default=0.0)

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_SEPARATOR = "-" * 72

_STEP_LABELS = {
    "webhook_received": "Webhook recibido",
    "webhook_payload_validated": "Validando payload",
    "webhook_background_dispatch": "Enviando a segundo plano",
    "semantic_routing_and_graph": "Procesando flujo principal",
    "build_context": "Preparando contexto",
    "barbershop_config": "Cargando configuracion de la barberia",
    "mem0_lookup": "Buscando memoria previa",
    "qdrant_lookup": "Consultando contexto RAG",
    "intent_router_openai": "Decidiendo intencion",
    "router_prompt_compose": "Preparando texto para clasificar",
    "router_match": "Ruta detectada",
    "router_fallback": "Usando ruta de respaldo",
    "branch_selection": "Eligiendo camino del flujo",
    "conversation": "Yendo a conversacion general",
    "rag": "Yendo a respuesta con contexto",
    "booking": "Yendo a gestion de cita",
    "conversation_node": "Generando respuesta general",
    "rag_node": "Generando respuesta con contexto",
    "booking_node": "Extrayendo datos de la cita",
    "booking_payload": "Revisando datos de la cita",
    "store_memory": "Guardando en memoria",
    "outbound_response": "Enviando respuesta a Chatwoot",
    "llm_chat_completion": "Consultando proveedor LLM",
    "llm_json_schema_retry": "Reintentando JSON compatible",
    "conversation_prompt_compose": "Preparando respuesta general",
    "conversation_fallback": "Usando respuesta general de respaldo",
    "rag_prompt_compose": "Preparando respuesta con contexto",
    "rag_fallback": "Usando respuesta RAG de respaldo",
    "booking_prompt_compose": "Preparando extraccion de cita",
    "booking_json_parse": "Leyendo datos estructurados",
    "booking_fallback": "Usando extraccion de respaldo",
    "flow_execution": "Ejecucion del flujo",
    "unknown_branch": "Ruta desconocida",
}


def configure_flow_logger(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def new_flow_id() -> str:
    return uuid.uuid4().hex[:10]


def bind_flow(flow_id: str, conversation_id: str) -> None:
    _flow_id_var.set(flow_id)
    _conversation_id_var.set(conversation_id)
    _flow_started_at_var.set(time.perf_counter())


def clear_flow() -> None:
    _flow_id_var.set("-")
    _conversation_id_var.set("-")
    _flow_started_at_var.set(0.0)


def start_flow(message_preview: str) -> None:
    logger.info("")
    logger.info(f"{_DIM}{_SEPARATOR}{_RESET}")
    logger.info(f"{_BOLD}{_CYAN}Flujo {_flow_id_var.get()}{_RESET}  {_DIM}Conversacion {_conversation_id_var.get()}{_RESET}")
    logger.info(f"{_BLUE}Mensaje{_RESET}  {_safe_preview(message_preview)}")


def end_flow(status: str, detail: str = "") -> None:
    started_at = _flow_started_at_var.get()
    elapsed_ms = 0 if started_at == 0.0 else int((time.perf_counter() - started_at) * 1000)
    suffix = f"{detail} | elapsed={elapsed_ms}ms" if detail else f"elapsed={elapsed_ms}ms"
    color = _status_color(status)
    logger.info(f"{color}{_BOLD}Resultado {status}{_RESET}  {suffix}")
    logger.info(f"{_DIM}{_SEPARATOR}{_RESET}")


def step(name: str, status: str = "RUN", detail: str = "") -> None:
    logger.info(_line(name=name, status=status, detail=detail, indent=0))


def substep(name: str, status: str = "RUN", detail: str = "") -> None:
    logger.info(_line(name=name, status=status, detail=detail, indent=2))


def mark_error(step_name: str, exc: Exception) -> None:
    step(step_name, "ERROR", f"{type(exc).__name__}: {exc}")


def _line(name: str, status: str, detail: str, indent: int) -> str:
    padding = " " * indent
    label = _status_label(status)
    color = _status_color(status)
    clean_name = _clean_name(name)
    if detail:
        return f"{padding}{color}{label:<11}{_RESET} {clean_name}  {_DIM}{detail}{_RESET}"
    return f"{padding}{color}{label:<11}{_RESET} {clean_name}"


def _status_label(status: str) -> str:
    labels = {
        "RUN": "En curso",
        "OK": "Listo",
        "WARN": "Aviso",
        "ERROR": "Error",
    }
    return labels.get(status.upper(), status.title())


def _status_color(status: str) -> str:
    colors = {
        "RUN": _BLUE,
        "OK": _GREEN,
        "WARN": _YELLOW,
        "ERROR": _RED,
    }
    return colors.get(status.upper(), _CYAN)


def _clean_name(name: str) -> str:
    base_name = re.sub(r"^[0-9]+(?:\.[0-9a-z]+)*(?:\s+)?", "", name, flags=re.IGNORECASE).strip()
    slug = re.sub(r"[^a-z0-9]+", "_", base_name.lower()).strip("_")
    if slug in _STEP_LABELS:
        return _STEP_LABELS[slug]
    return base_name.replace("_", " ")


def _safe_preview(message: str, max_len: int = 120) -> str:
    compact = " ".join(message.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
