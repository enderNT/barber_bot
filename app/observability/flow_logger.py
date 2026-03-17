from __future__ import annotations

import contextvars
import logging
import time
import uuid

logger = logging.getLogger("clinica.flow")

_flow_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("flow_id", default="-")
_conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("conversation_id", default="-")
_flow_started_at_var: contextvars.ContextVar[float] = contextvars.ContextVar("flow_started_at", default=0.0)


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
    logger.info("=" * 88)
    logger.info(_format("FLOW START", f"msg='{_safe_preview(message_preview)}'"))
    logger.info("=" * 88)


def end_flow(status: str, detail: str = "") -> None:
    started_at = _flow_started_at_var.get()
    elapsed_ms = 0 if started_at == 0.0 else int((time.perf_counter() - started_at) * 1000)
    suffix = f"{detail} | elapsed={elapsed_ms}ms" if detail else f"elapsed={elapsed_ms}ms"
    logger.info(_format(f"FLOW END {status}", suffix))
    logger.info("=" * 88)


def step(name: str, status: str = "RUN", detail: str = "") -> None:
    payload = detail if detail else "-"
    logger.info(_format(f"[{status}] {name}", payload))


def substep(name: str, status: str = "RUN", detail: str = "") -> None:
    payload = detail if detail else "-"
    logger.info(_format(f"  > [{status}] {name}", payload))


def mark_error(step_name: str, exc: Exception) -> None:
    step(step_name, "ERROR", f"{type(exc).__name__}: {exc}")


def _format(event: str, detail: str) -> str:
    return f"| flow={_flow_id_var.get()} | conv={_conversation_id_var.get()} | {event} | {detail}"


def _safe_preview(message: str, max_len: int = 120) -> str:
    compact = " ".join(message.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
