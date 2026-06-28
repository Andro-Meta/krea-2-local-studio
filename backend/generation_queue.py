from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any


JobHandler = Callable[[str, Any], Awaitable[None]]


class GenerationQueue:
    """Single-worker FIFO queue for GPU-bound generation work."""

    def __init__(self, handler: JobHandler):
        self._handler = handler
        self._pending: deque[str] = deque()
        self._payloads: dict[str, Any] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None
        self._event = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()

    def enqueue(self, job_id: str, payload: Any, *, username: str | None, role: str) -> dict[str, Any]:
        if job_id in self._records:
            raise ValueError(f"duplicate job id: {job_id}")
        record = {
            "status": "queued",
            "queue_position": len(self._pending) + 1,
            "queue_length": len(self._pending) + 1,
            "active_job_id": self._active_job_id,
            "username": username,
            "role": role,
            "queued_at": time.time(),
            "started_at": None,
        }
        self._records[job_id] = record
        self._payloads[job_id] = payload
        self._pending.append(job_id)
        self._idle.clear()
        self._event.set()
        self._recompute_positions()
        return dict(record)

    def cancel(self, job_id: str) -> bool:
        record = self._records.get(job_id)
        if not record or record.get("status") != "queued":
            return False
        try:
            self._pending.remove(job_id)
        except ValueError:
            return False
        record["status"] = "cancelled"
        record["queue_position"] = None
        record["queue_length"] = len(self._pending)
        self._payloads.pop(job_id, None)
        self._recompute_positions()
        self._maybe_idle()
        return True

    def status(self, job_id: str) -> dict[str, Any]:
        return dict(self._records[job_id])

    def all_statuses(self) -> dict[str, dict[str, Any]]:
        return {job_id: dict(record) for job_id, record in self._records.items()}

    def has_active_or_pending(self) -> bool:
        return self._active_job_id is not None or bool(self._pending)

    async def run(self) -> None:
        while True:
            if not self._pending:
                self._maybe_idle()
                self._event.clear()
                await self._event.wait()
                continue
            job_id = self._pending.popleft()
            record = self._records[job_id]
            if record.get("status") == "cancelled":
                self._payloads.pop(job_id, None)
                self._recompute_positions()
                self._maybe_idle()
                continue
            self._active_job_id = job_id
            record["status"] = "running"
            record["started_at"] = time.time()
            record["queue_position"] = None
            self._recompute_positions()
            try:
                await self._handler(job_id, self._payloads[job_id])
            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)
            else:
                if record.get("status") == "running":
                    record["status"] = "done"
            finally:
                self._payloads.pop(job_id, None)
                self._active_job_id = None
                self._recompute_positions()
                self._maybe_idle()

    async def join(self) -> None:
        await self._idle.wait()

    def _maybe_idle(self) -> None:
        if self._active_job_id is None and not self._pending:
            self._idle.set()

    def _recompute_positions(self) -> None:
        queue_length = len(self._pending)
        for index, job_id in enumerate(self._pending, start=1):
            record = self._records[job_id]
            record["queue_position"] = index
            record["queue_length"] = queue_length
            record["active_job_id"] = self._active_job_id
        if self._active_job_id and self._active_job_id in self._records:
            self._records[self._active_job_id]["queue_length"] = queue_length
