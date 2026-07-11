from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

JobHandler = Callable[[str, Any], Awaitable[None]]

_LOCAL_USER = "__local__"
_MAX_TERMINAL_RECORDS = 200


class GenerationQueue:
    """Single-worker queue for GPU-bound generation work.

    Scheduling is round-robin across users: each turn serves one job from the
    next user who has pending work, so one user's large batch cannot monopolize
    the GPU while others wait. Within a single user, jobs stay FIFO. With one
    user active, behavior is plain FIFO.
    """

    def __init__(self, handler: JobHandler):
        self._handler = handler
        self._pending: deque[str] = deque()  # arrival order (FIFO within a user)
        self._payloads: dict[str, Any] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None
        self._cancel_requested: set[str] = set()
        self._last_served_user: str | None = None
        self._event = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active_job_id(self) -> str | None:
        return self._active_job_id

    @staticmethod
    def _user_key(username: str | None) -> str:
        return str(username) if username else _LOCAL_USER

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

    def request_cancel(self, job_id: str) -> str:
        """Cancel a queued job or flag the running one for interruption.

        Returns 'dequeued' (was pending, removed), 'interrupt' (is the active job,
        now flagged so the caller can interrupt ComfyUI), or 'none'.
        """
        record = self._records.get(job_id)
        if not record:
            return "none"
        if record.get("status") == "queued":
            return "dequeued" if self.cancel(job_id) else "none"
        if job_id == self._active_job_id and record.get("status") == "running":
            self._cancel_requested.add(job_id)
            return "interrupt"
        return "none"

    def cancel_requested(self, job_id: str) -> bool:
        return job_id in self._cancel_requested

    def status(self, job_id: str) -> dict[str, Any]:
        record = self._records.get(job_id)
        if record is None:
            # Pruned or never enqueued — report a terminal-ish shape instead
            # of raising KeyError at callers.
            return {"status": "unknown", "queue_position": None, "queue_length": len(self._pending)}
        return dict(record)

    def all_statuses(self) -> dict[str, dict[str, Any]]:
        return {job_id: dict(record) for job_id, record in self._records.items()}

    def has_active_or_pending(self) -> bool:
        return self._active_job_id is not None or bool(self._pending)

    def _scheduled_order(self) -> list[str]:
        """Pending jobs in projected execution order (round-robin across users).

        Users are ordered by their earliest pending arrival; scheduling resumes
        after the last-served user. Each round takes one job per user.
        """
        per_user: dict[str, deque[str]] = {}
        for job_id in self._pending:
            user = self._user_key(self._records[job_id].get("username"))
            per_user.setdefault(user, deque()).append(job_id)
        users = list(per_user)
        if not users:
            return []
        # Rotate so the first turn goes to the user after the last one served.
        if self._last_served_user in users:
            start = (users.index(self._last_served_user) + 1) % len(users)
            users = users[start:] + users[:start]
        order: list[str] = []
        while any(per_user[user] for user in users):
            for user in users:
                if per_user[user]:
                    order.append(per_user[user].popleft())
        return order

    def _pop_next(self) -> str | None:
        order = self._scheduled_order()
        if not order:
            return None
        job_id = order[0]
        self._pending.remove(job_id)
        self._last_served_user = self._user_key(self._records[job_id].get("username"))
        return job_id

    async def run(self) -> None:
        while True:
            if not self._pending:
                self._maybe_idle()
                self._event.clear()
                await self._event.wait()
                continue
            job_id = self._pop_next()
            if job_id is None:
                continue
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
                if job_id in self._cancel_requested:
                    record["status"] = "cancelled"
                else:
                    record["status"] = "error"
                    record["error"] = str(exc)
            else:
                if record.get("status") == "running":
                    record["status"] = "cancelled" if job_id in self._cancel_requested else "done"
            finally:
                self._cancel_requested.discard(job_id)
                self._payloads.pop(job_id, None)
                self._active_job_id = None
                self._prune_terminal_records()
                self._recompute_positions()
                self._maybe_idle()

    async def join(self) -> None:
        await self._idle.wait()

    def _maybe_idle(self) -> None:
        if self._active_job_id is None and not self._pending:
            self._idle.set()

    def _prune_terminal_records(self) -> None:
        """Drop the oldest finished/cancelled/errored records so a long-lived
        multi-user server doesn't grow this dict forever (dicts keep insertion
        order, so the oldest terminal records go first)."""
        terminal = [
            job_id for job_id, record in self._records.items()
            if record.get("status") in {"done", "cancelled", "error"}
        ]
        for job_id in terminal[: max(0, len(terminal) - _MAX_TERMINAL_RECORDS)]:
            self._records.pop(job_id, None)

    def _recompute_positions(self) -> None:
        # Positions reflect the PROJECTED round-robin execution order, so the
        # number a user sees is their real place in line, not raw arrival order.
        order = self._scheduled_order()
        queue_length = len(order)
        for index, job_id in enumerate(order, start=1):
            record = self._records[job_id]
            record["queue_position"] = index
            record["queue_length"] = queue_length
            record["active_job_id"] = self._active_job_id
        if self._active_job_id and self._active_job_id in self._records:
            self._records[self._active_job_id]["queue_length"] = queue_length
