from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

INTERACTIVE = "interactive"
BACKGROUND = "background"

_LOCAL_USER = "__local__"
_MAX_INTERACTIVE_PER_USER = 8
_MAX_INTERACTIVE_GLOBAL = 64
_MAX_BACKGROUND_GLOBAL = 4
_MAX_TERMINAL_RECORDS = 200
_TERMINAL_STATUSES = {"done", "cancelled", "error"}

TaskHandler = Callable[[str, Any], Awaitable[None]]


@dataclass(frozen=True)
class EnqueueResult(Mapping[str, Any]):
    accepted: bool
    record: dict[str, Any] | None
    reason: str | None
    limit: int | None
    active_count: int

    def __getitem__(self, key: str) -> Any:
        if self.record is None:
            raise KeyError(key)
        return self.record[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.record or {})

    def __len__(self) -> int:
        return len(self.record or {})


class GpuTaskQueue:
    """Single-worker GPU queue with fair interactive and idle background work."""

    def __init__(self, handler: TaskHandler, *, enforce_limits: bool = True):
        self._handler = handler
        self._enforce_limits = enforce_limits
        self._interactive_lanes: dict[str, deque[str]] = {}
        self._user_ring: deque[str] = deque()
        self._background_pending: deque[str] = deque()
        self._payloads: dict[str, Any] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._active_task_id: str | None = None
        self._cancel_requested: set[str] = set()
        self._worker_running = False
        self._event = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active_task_id(self) -> str | None:
        return self._active_task_id

    @property
    def active_job_id(self) -> str | None:
        return self._active_task_id

    @staticmethod
    def _user_key(username: str | None) -> str:
        return str(username) if username else _LOCAL_USER

    def check_capacity(
        self,
        username: str | None,
        priority_class: str,
        count: int = 1,
    ) -> EnqueueResult:
        if priority_class not in {INTERACTIVE, BACKGROUND}:
            raise ValueError(f"unknown priority class: {priority_class}")
        if count < 0:
            raise ValueError("count must be non-negative")

        if priority_class == INTERACTIVE:
            user_count = self._unfinished_count(
                priority_class=INTERACTIVE,
                user_key=self._user_key(username),
            )
            if (
                self._enforce_limits
                and user_count + count > _MAX_INTERACTIVE_PER_USER
            ):
                return EnqueueResult(
                    False,
                    None,
                    "per_user_limit",
                    _MAX_INTERACTIVE_PER_USER,
                    user_count,
                )
            global_count = self._unfinished_count(priority_class=INTERACTIVE)
            if (
                self._enforce_limits
                and global_count + count > _MAX_INTERACTIVE_GLOBAL
            ):
                return EnqueueResult(
                    False,
                    None,
                    "global_limit",
                    _MAX_INTERACTIVE_GLOBAL,
                    global_count,
                )
            return EnqueueResult(
                True, None, None, None, global_count + count
            )

        global_count = self._unfinished_count(priority_class=BACKGROUND)
        if (
            self._enforce_limits
            and global_count + count > _MAX_BACKGROUND_GLOBAL
        ):
            return EnqueueResult(
                False,
                None,
                "global_limit",
                _MAX_BACKGROUND_GLOBAL,
                global_count,
            )
        return EnqueueResult(True, None, None, None, global_count + count)

    def admission(self, username: str | None) -> dict[str, int]:
        return {
            "per_user_active": self._unfinished_count(
                priority_class=INTERACTIVE,
                user_key=self._user_key(username),
            ),
            "per_user_limit": _MAX_INTERACTIVE_PER_USER,
            "global_interactive_active": self._unfinished_count(
                priority_class=INTERACTIVE
            ),
            "global_interactive_limit": _MAX_INTERACTIVE_GLOBAL,
            "global_background_active": self._unfinished_count(
                priority_class=BACKGROUND
            ),
            "global_background_limit": _MAX_BACKGROUND_GLOBAL,
        }

    def enqueue(
        self,
        task_id: str,
        payload: Any,
        *,
        username: str | None,
        role: str | None,
        task_kind: str = "generation",
        priority_class: str = INTERACTIVE,
    ) -> EnqueueResult:
        if task_id in self._records:
            raise ValueError(f"duplicate task id: {task_id}")
        capacity = self.check_capacity(username, priority_class, 1)
        if not capacity.accepted:
            return capacity
        active_count = capacity.active_count

        record = {
            "status": "queued",
            "queue_position": None,
            "queue_length": self._pending_count() + 1,
            "active_job_id": self._active_task_id,
            "active_task_id": self._active_task_id,
            "username": username,
            "role": role,
            "task_kind": task_kind,
            "priority_class": priority_class,
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }
        self._records[task_id] = record
        self._payloads[task_id] = payload
        if priority_class == INTERACTIVE:
            user = self._user_key(username)
            lane = self._interactive_lanes.get(user)
            if lane is None:
                lane = deque()
                self._interactive_lanes[user] = lane
                self._user_ring.append(user)
            lane.append(task_id)
        else:
            self._background_pending.append(task_id)

        self._idle.clear()
        self._event.set()
        self._recompute_positions()
        return EnqueueResult(True, dict(record), None, None, active_count)

    def cancel(self, task_id: str) -> bool:
        record = self._records.get(task_id)
        if not record or record.get("status") != "queued":
            return False

        if record.get("priority_class") == INTERACTIVE:
            user = self._user_key(record.get("username"))
            lane = self._interactive_lanes.get(user)
            if lane is None:
                return False
            try:
                lane.remove(task_id)
            except ValueError:
                return False
            if not lane:
                self._interactive_lanes.pop(user, None)
                with suppress(ValueError):
                    self._user_ring.remove(user)
        else:
            try:
                self._background_pending.remove(task_id)
            except ValueError:
                return False

        record["status"] = "cancelled"
        record["queue_position"] = None
        record["queue_length"] = self._pending_count()
        record["finished_at"] = time.time()
        self._payloads.pop(task_id, None)
        self._recompute_positions()
        self._prune_terminal_records()
        self._maybe_idle()
        return True

    def request_cancel(self, task_id: str) -> str:
        record = self._records.get(task_id)
        if not record:
            return "none"
        if record.get("status") == "queued":
            return "dequeued" if self.cancel(task_id) else "none"
        if task_id == self._active_task_id and record.get("status") == "running":
            self._cancel_requested.add(task_id)
            return "interrupt"
        return "none"

    def begin_finalizing(self, task_id: str) -> bool:
        """Atomically close cancellation before result persistence."""
        record = self._records.get(task_id)
        if (
            task_id != self._active_task_id
            or not record
            or record.get("status") != "running"
            or task_id in self._cancel_requested
        ):
            return False
        record["status"] = "finalizing"
        return True

    def cancel_requested(self, task_id: str) -> bool:
        return task_id in self._cancel_requested

    def status(self, task_id: str) -> dict[str, Any]:
        record = self._records.get(task_id)
        if record is None:
            return {
                "status": "unknown",
                "queue_position": None,
                "queue_length": self._pending_count(),
            }
        return dict(record)

    def all_statuses(self) -> dict[str, dict[str, Any]]:
        return {
            task_id: dict(record) for task_id, record in self._records.items()
        }

    def has_active_or_pending(self) -> bool:
        return self._active_task_id is not None or self._pending_count() > 0

    async def run(self) -> None:
        if self._worker_running:
            raise RuntimeError("GpuTaskQueue worker is already running")
        self._worker_running = True
        try:
            while True:
                task_id = self._pop_next()
                if task_id is None:
                    self._maybe_idle()
                    self._event.clear()
                    await self._event.wait()
                    continue

                record = self._records[task_id]
                self._active_task_id = task_id
                record["status"] = "running"
                record["started_at"] = time.time()
                record["queue_position"] = None
                self._recompute_positions()
                try:
                    await self._handler(task_id, self._payloads[task_id])
                except asyncio.CancelledError:
                    record["status"] = "cancelled"
                    raise
                except Exception as exc:
                    if task_id in self._cancel_requested:
                        record["status"] = "cancelled"
                    else:
                        record["status"] = "error"
                        record["error"] = str(exc)
                else:
                    if record.get("status") in {"running", "finalizing"}:
                        record["status"] = (
                            "cancelled"
                            if task_id in self._cancel_requested
                            else "done"
                        )
                finally:
                    record["finished_at"] = time.time()
                    self._cancel_requested.discard(task_id)
                    self._payloads.pop(task_id, None)
                    self._active_task_id = None
                    record["active_job_id"] = None
                    record["active_task_id"] = None
                    self._prune_terminal_records()
                    self._recompute_positions()
                    self._maybe_idle()
        finally:
            self._worker_running = False

    async def join(self) -> None:
        await self._idle.wait()

    def _unfinished_count(
        self,
        *,
        priority_class: str,
        user_key: str | None = None,
    ) -> int:
        return sum(
            1
            for record in self._records.values()
            if record.get("status") not in _TERMINAL_STATUSES
            and record.get("priority_class") == priority_class
            and (
                user_key is None
                or self._user_key(record.get("username")) == user_key
            )
        )

    def _pending_count(self) -> int:
        return sum(map(len, self._interactive_lanes.values())) + len(
            self._background_pending
        )

    def _scheduled_order(self) -> list[str]:
        lanes = {
            user: deque(task_ids)
            for user, task_ids in self._interactive_lanes.items()
        }
        ring = deque(self._user_ring)
        order: list[str] = []
        while ring:
            user = ring.popleft()
            lane = lanes[user]
            order.append(lane.popleft())
            if lane:
                ring.append(user)
        order.extend(self._background_pending)
        return order

    def _pop_next(self) -> str | None:
        if self._user_ring:
            user = self._user_ring.popleft()
            lane = self._interactive_lanes[user]
            task_id = lane.popleft()
            if lane:
                self._user_ring.append(user)
            else:
                self._interactive_lanes.pop(user, None)
            return task_id
        if self._background_pending:
            return self._background_pending.popleft()
        return None

    def _maybe_idle(self) -> None:
        if self._active_task_id is None and self._pending_count() == 0:
            self._idle.set()

    def _prune_terminal_records(self) -> None:
        terminal = [
            task_id
            for task_id, record in self._records.items()
            if record.get("status") in _TERMINAL_STATUSES
        ]
        excess = len(terminal) - _MAX_TERMINAL_RECORDS
        for task_id in terminal[: max(0, excess)]:
            self._records.pop(task_id, None)

    def _recompute_positions(self) -> None:
        order = self._scheduled_order()
        queue_length = len(order)
        for index, task_id in enumerate(order, start=1):
            record = self._records[task_id]
            record["queue_position"] = index
            record["queue_length"] = queue_length
            record["active_job_id"] = self._active_task_id
            record["active_task_id"] = self._active_task_id
        if self._active_task_id and self._active_task_id in self._records:
            active = self._records[self._active_task_id]
            active["queue_length"] = queue_length
            active["active_job_id"] = self._active_task_id
            active["active_task_id"] = self._active_task_id
