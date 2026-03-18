from __future__ import annotations

import logging
from typing import Any

from app.models.schemas import IntentDecision
from app.observability.flow_logger import substep
from app.settings import Settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency during local development
    from semantic_router import Route
    from semantic_router.encoders import OpenAIEncoder
    from semantic_router.index import LocalIndex
    from semantic_router.routers import SemanticRouter

    SEMANTIC_ROUTER_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for environments without the package
    Route = Any  # type: ignore[assignment]
    OpenAIEncoder = Any  # type: ignore[assignment]
    LocalIndex = Any  # type: ignore[assignment]
    SemanticRouter = Any  # type: ignore[assignment]
    SEMANTIC_ROUTER_AVAILABLE = False
ROUTE_DEFINITIONS = [
    {
        "name": "conversation",
        "utterances": [
            "hola",
            "buenos dias",
            "gracias",
            "perfecto",
            "ok",
            "me puedes ayudar",
            "solo queria saludar",
            "tengo una duda corta",
            "muchas gracias",
            "entendido",
            "de acuerdo",
            "hola buenos dias",
        ],
        "score_threshold": 0.42,
    },
    {
        "name": "rag",
        "utterances": [
            "cuales son sus horarios",
            "que servicios ofrecen",
            "cuanto cuesta la consulta",
            "que doctores tienen",
            "atienden sabados",
            "donde estan ubicados",
            "que especialidades manejan",
            "politicas de pago",
            "cuanto cuesta",
            "que precio tiene la consulta",
            "que doctores atienden",
            "a que hora abren",
        ],
        "score_threshold": 0.46,
    },
    {
        "name": "appointment_intent",
        "utterances": [
            "quiero agendar una cita",
            "necesito una consulta",
            "quiero reservar con un doctor",
            "me ayudas a sacar una cita",
            "busco cita para dermatologia",
            "quiero programar una visita",
            "tengo que agendar una cita para manana",
            "quiero una cita el viernes",
            "me gustaria una cita",
            "quiero reservar una consulta",
            "puedo agendar para hoy",
            "necesito cita con pediatria",
        ],
        "score_threshold": 0.48,
    },
]


class ClinicIntentRouterService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._debug = settings.semantic_router_debug
        self._router = self._build_router()

    def _build_router(self):
        if not SEMANTIC_ROUTER_AVAILABLE or not self._settings.openai_api_key:
            logger.warning(
                "semantic-router unavailable or OPENAI_API_KEY missing; using deterministic fallback router"
            )
            return None

        encoder_kwargs: dict[str, Any] = {
            "name": self._settings.openai_embedding_model,
            "openai_api_key": self._settings.openai_api_key,
        }
        if self._settings.openai_base_url:
            encoder_kwargs["openai_base_url"] = self._settings.openai_base_url.rstrip("/")
        encoder = OpenAIEncoder(**encoder_kwargs)
        routes = [Route(name=item["name"], utterances=item["utterances"], score_threshold=item["score_threshold"]) for item in ROUTE_DEFINITIONS]
        return SemanticRouter(
            encoder=encoder,
            routes=routes,
            index=LocalIndex(),
            auto_sync="local",
        )

    async def route_intent(self, user_message: str, memories: list[str]) -> IntentDecision:
        if self._router is None:
            return self._fallback_route(user_message)

        try:
            router_input = self._build_router_input(user_message, memories)
            substep(
                "router_prompt_compose",
                "OK",
                f"msg_chars={len(user_message)} memories={len(memories)} router_chars={len(router_input)}",
            )
            if self._debug:
                await self._log_route_diagnostics(router_input, user_message)
            route_choice = await self._router.acall(router_input)
            route_name = getattr(route_choice, "name", None)
            score = self._extract_score(route_choice)
            if not route_name:
                substep("router_match", "WARN", "no semantic match, using fallback")
                return self._fallback_route(user_message)

            confidence = self._normalize_score(score)
            score_text = f"{score:.3f}" if score is not None else "n/a"
            substep("router_match", "OK", f"route={route_name} score={score_text}")
            return IntentDecision(
                intent=route_name,
                confidence=confidence,
                reason=f"semantic-router:{route_name}:{score:.3f}" if score is not None else f"semantic-router:{route_name}",
            )
        except Exception as exc:
            logger.warning("semantic-router failed, using fallback: %s", exc)
            substep("router_fallback", "WARN", "semantic-router error")
            return self._fallback_route(user_message)

    async def _log_route_diagnostics(self, router_input: str, user_message: str) -> None:
        if self._router is None:
            return

        try:
            if not (await self._router.index.ais_ready()):
                await self._router._async_init_index_state()

            vector = await self._router._async_encode(text=[router_input], input_type="queries")
            scores, routes = await self._router.index.aquery(vector=vector[0], top_k=self._router.top_k)
            query_results = [{"route": route, "score": score.item()} for route, score in zip(routes, scores)]
            scored_routes = self._router._score_routes(query_results=query_results)

            substep("router_input_preview", "OK", _compact_text(user_message, max_len=140))
            if not scored_routes:
                substep("router_scores", "WARN", "sin resultados del indice")
                return

            for route_name, total_score, _ in scored_routes[:3]:
                route = self._router.check_for_matching_routes(top_class=route_name)
                threshold = getattr(route, "score_threshold", None) if route is not None else None
                threshold_text = f"{threshold:.3f}" if isinstance(threshold, (float, int)) else "n/a"
                passed_text = "pasa" if threshold is None or total_score >= threshold else "no pasa"
                substep(
                    "router_scores",
                    "OK",
                    f"{route_name}: score={total_score:.3f} threshold={threshold_text} {passed_text}",
                )
        except Exception as exc:
            substep("router_scores", "WARN", f"no pude calcular diagnostico: {type(exc).__name__}: {exc}")

    def _build_router_input(self, user_message: str, memories: list[str]) -> str:
        sections = [f"Mensaje actual:\n{_compact_text(user_message, max_len=400)}"]

        memory_lines = [_compact_text(memory, max_len=120) for memory in memories[:2] if memory.strip()]
        if memory_lines:
            sections.append("Memoria relevante del usuario:\n- " + "\n- ".join(memory_lines))

        return "\n\n".join(sections)

    def _extract_score(self, route_choice: Any) -> float | None:
        score = getattr(route_choice, "similarity_score", None)
        if score is None:
            score = getattr(route_choice, "score", None)
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def _normalize_score(self, score: float | None) -> float:
        if score is None:
            return 0.0
        return max(0.0, min(1.0, score))

    def _fallback_route(self, user_message: str) -> IntentDecision:
        lowered = user_message.lower()
        appointment_keywords = ("cita", "agendar", "agendo", "reservar", "consulta", "doctor", "doctora")
        rag_keywords = ("precio", "costo", "horario", "horarios", "doctor", "especialidad", "servicio", "direccion")
        if any(word in lowered for word in appointment_keywords):
            return IntentDecision(intent="appointment_intent", confidence=0.72, reason="heuristic-fallback")
        if any(word in lowered for word in rag_keywords):
            return IntentDecision(intent="rag", confidence=0.60, reason="heuristic-fallback")
        return IntentDecision(intent="conversation", confidence=0.55, reason="heuristic-fallback")


def _compact_text(value: str, max_len: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
