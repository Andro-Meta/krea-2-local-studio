from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any


class RealtimePreviewRegistry:
    """Small in-memory registry for debounced preview jobs.

    The latest session revision wins. Older jobs are allowed to finish in the
    background, but their results are marked stale so the frontend never shows
    old canvas states over newer edits.
    """

    def __init__(self, max_jobs: int = 100) -> None:
        self.max_jobs = max_jobs
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._latest_revision: dict[str, int] = {}

    def create(self, session_id: str) -> dict[str, Any]:
        revision = self._latest_revision.get(session_id, 0) + 1
        self._latest_revision[session_id] = revision
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "session_id": session_id,
            "revision": revision,
            "status": "queued",
            "progress": 0,
            "image_b64": None,
            "seed": None,
            "metadata": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._jobs[job_id] = job
        self._evict()
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return None if job is None else dict(job)

    def update(self, job_id: str, **patch: Any) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.update(patch)
        job["updated_at"] = time.time()
        return True

    def cancel(self, job_id: str) -> bool:
        return self.update(job_id, status="cancelled", progress=100)

    def busy(self) -> bool:
        return any(job.get("status") in {"queued", "running"} for job in self._jobs.values())

    def is_current(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        return self._latest_revision.get(job["session_id"]) == job["revision"]

    def complete(self, job_id: str, *, image_b64: str, seed: int | None, metadata: dict | None = None) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.get("status") == "cancelled":
            return False
        if not self.is_current(job_id):
            self.update(job_id, status="stale", progress=100)
            return False
        return self.update(job_id, status="done", progress=100, image_b64=image_b64, seed=seed, metadata=metadata or {})

    def fail(self, job_id: str, error: str) -> bool:
        return self.update(job_id, status="error", progress=100, error=error)

    def _evict(self) -> None:
        while len(self._jobs) > self.max_jobs:
            self._jobs.popitem(last=False)
