from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import httpx

from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QdrantSearchResult:
    id: str
    score: float
    payload: dict[str, Any]


class QdrantRetrievalService:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.qdrant_enabled
        self._simulate = settings.qdrant_simulate
        self._base_url = settings.qdrant_base_url.rstrip("/") if settings.qdrant_base_url else None
        self._api_key = settings.qdrant_api_key
        self._collection_name = settings.qdrant_collection_name
        self._timeout = settings.qdrant_timeout_seconds
        self._top_k = settings.qdrant_top_k
        self._vector_size = settings.qdrant_vector_size

    @property
    def ready(self) -> bool:
        return bool(self._enabled and self._base_url and self._collection_name)

    async def search(self, query: str, contact_id: str, limit: int | None = None) -> list[QdrantSearchResult]:
        top_k = limit or self._top_k
        if self._simulate or not self.ready:
            return self._simulate_search(query=query, contact_id=contact_id, limit=top_k)
        return await self._http_search(query=query, contact_id=contact_id, limit=top_k)

    async def build_context(self, query: str, contact_id: str, barbershop_context: str, memories: list[str]) -> str:
        results = await self.search(query=query, contact_id=contact_id)
        chunks = [
            "Contexto base de la barberia:",
            barbershop_context,
            "",
            "Memoria conversacional:",
            "\n".join(f"- {item}" for item in memories) if memories else "- Sin memorias",
            "",
            "Recuperacion vectorial Qdrant:",
        ]
        if results:
            for result in results:
                source = result.payload.get("source", "unknown")
                text = result.payload.get("text", "")
                chunks.append(f"- [{result.id}] score={result.score:.3f} source={source} text={text}")
        else:
            chunks.append("- Sin resultados")
        return "\n".join(chunks)

    async def _http_search(self, query: str, contact_id: str, limit: int) -> list[QdrantSearchResult]:
        payload = {
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
            "vector": self._fake_vector(query, contact_id),
            "filter": {
                "must": [
                    {
                        "key": "contact_id",
                        "match": {"value": contact_id},
                    }
                ]
            },
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key

        url = f"{self._base_url}/collections/{self._collection_name}/points/search"
        logger.info("Qdrant search request prepared for collection=%s limit=%s", self._collection_name, limit)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        results: list[QdrantSearchResult] = []
        for item in data.get("result", [])[:limit]:
            results.append(
                QdrantSearchResult(
                    id=str(item.get("id", "unknown")),
                    score=float(item.get("score", 0.0)),
                    payload=item.get("payload", {}) or {},
                )
            )
        return results

    def _simulate_search(self, query: str, contact_id: str, limit: int) -> list[QdrantSearchResult]:
        logger.info(
            "Qdrant simulated search collection=%s contact_id=%s limit=%s query=%s",
            self._collection_name,
            contact_id,
            limit,
            query,
        )
        base = sha256(f"{contact_id}:{query}".encode("utf-8")).hexdigest()
        results: list[QdrantSearchResult] = []
        for index in range(limit):
            token = base[index * 8 : (index + 1) * 8] or base[:8]
            score = max(0.25, 0.93 - index * 0.11)
            results.append(
                QdrantSearchResult(
                    id=f"sim-{token}",
                    score=score,
                    payload={
                        "text": f"Simulacion Qdrant para '{query}'",
                        "source": "simulated-vector-store",
                        "contact_id": contact_id,
                        "collection": self._collection_name,
                        "rank": index + 1,
                    },
                )
            )
        return results

    def _fake_vector(self, query: str, contact_id: str) -> list[float]:
        digest = sha256(f"{contact_id}:{query}".encode("utf-8")).digest()
        vector: list[float] = []
        for index in range(self._vector_size):
            byte = digest[index % len(digest)]
            vector.append(round(byte / 255.0, 4))
        return vector
