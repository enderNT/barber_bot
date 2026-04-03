from __future__ import annotations

import argparse
import asyncio
from urllib.parse import urlsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.services.memory import build_memory_index_config
from app.settings import Settings
from app.tracing.postgres import AsyncPostgresTraceRepository


def _normalize_dsn(raw_dsn: str) -> str:
    dsn = raw_dsn.strip()
    marker = "postgres://"
    alt_marker = "postgresql://"
    repeated_index = dsn.rfind(marker)
    if repeated_index > 0:
        return dsn[repeated_index:]
    repeated_index = dsn.rfind(alt_marker)
    if repeated_index > 0:
        return dsn[repeated_index:]
    return dsn


def _redact_dsn(dsn: str) -> str:
    parsed = urlsplit(dsn)
    if not parsed.scheme or not parsed.hostname:
        return "<invalid-dsn>"
    username = parsed.username or "<user>"
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    return f"{parsed.scheme}://{username}:***@{parsed.hostname}{port}{path}"


async def _setup_schema(dsn: str, dims: int) -> None:
    settings = Settings(
        _env_file=None,
        memory_backend="postgres",
        checkpoint_backend="postgres",
        postgres_dsn=dsn,
        memory_embedding_dims=dims,
    )

    async with AsyncPostgresStore.from_conn_string(
        dsn,
        index=build_memory_index_config(settings),
    ) as store:
        await store.setup()

    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()

    trace_repository = AsyncPostgresTraceRepository(dsn)
    try:
        await trace_repository.setup()
    finally:
        await trace_repository.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crea las tablas de memoria, checkpoints y tracing esperadas por la app."
    )
    parser.add_argument("--dsn", required=True, help="Cadena de conexion Postgres.")
    parser.add_argument("--dims", type=int, default=1536, help="Dimension de embeddings para pgvector.")
    args = parser.parse_args()

    normalized_dsn = _normalize_dsn(args.dsn)
    print(f"Usando DSN: {_redact_dsn(normalized_dsn)}")
    if normalized_dsn != args.dsn.strip():
        print("Se detecto un DSN duplicado o mal formado y se normalizo automaticamente.")

    asyncio.run(_setup_schema(normalized_dsn, args.dims))
    print("Esquema de Postgres preparado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
