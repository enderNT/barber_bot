from __future__ import annotations

import asyncio
from contextlib import suppress

from app.tracing.types import TraceRecord, TraceRepository, TraceSink


class NoopTraceSink(TraceSink):
    async def start(self) -> None:
        return None

    async def enqueue(self, trace_record: TraceRecord) -> None:
        del trace_record

    async def close(self) -> None:
        return None


class AsyncBatchTraceSink(TraceSink):
    def __init__(
        self,
        repository: TraceRepository,
        *,
        batch_size: int = 25,
        flush_interval_seconds: float = 2.0,
        queue_maxsize: int = 1000,
    ) -> None:
        self._repository = repository
        self._batch_size = max(1, batch_size)
        self._flush_interval_seconds = max(0.05, flush_interval_seconds)
        self._queue: asyncio.Queue[TraceRecord | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        await self._repository.setup()
        self._worker_task = asyncio.create_task(self._worker(), name="trace-batch-worker")

    async def enqueue(self, trace_record: TraceRecord) -> None:
        if self._worker_task is None or self._worker_task.done():
            await self.start()
        await self._queue.put(trace_record)

    async def close(self) -> None:
        if self._worker_task is None:
            await self._repository.close()
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None
        await self._repository.close()

    async def _worker(self) -> None:
        pending: list[TraceRecord] = []
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval_seconds)
                except asyncio.TimeoutError:
                    item = None

                if item is None:
                    if pending:
                        await self._repository.persist_batch(pending)
                        pending = []
                    if self._queue.empty():
                        break
                    continue

                pending.append(item)
                if len(pending) >= self._batch_size:
                    await self._repository.persist_batch(pending)
                    pending = []
        finally:
            with suppress(Exception):
                if pending:
                    await self._repository.persist_batch(pending)
