from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FunnelHealthState:
    """State for interval-level Funnel health and bounded repair backoff."""

    failure_threshold: int = 3
    backoff_seconds: tuple[int, ...] = (300, 900, 1800)
    failed_intervals: int = 0
    repair_count: int = 0
    next_repair_at: float = 0.0

    def record_probe(self, ok: bool, *, now: float) -> bool:
        if ok:
            self.failed_intervals = 0
            self.repair_count = 0
            self.next_repair_at = 0.0
            return False
        self.failed_intervals += 1
        return self.repair_due(now=now)

    def repair_due(self, *, now: float) -> bool:
        return self.failed_intervals >= self.failure_threshold and now >= self.next_repair_at

    def record_repair(self, *, now: float) -> int:
        index = min(self.repair_count, len(self.backoff_seconds) - 1)
        delay = self.backoff_seconds[index]
        self.repair_count += 1
        self.failed_intervals = 0
        self.next_repair_at = now + delay
        return delay


@dataclass
class FunnelHealthMonitor:
    """Explicit in-process switch plus interval-level health state."""

    enabled: bool = False
    state: FunnelHealthState = field(default_factory=FunnelHealthState)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False
        self.state.record_probe(True, now=0)

    def observe(self, healthy: bool, *, now: float) -> bool:
        if not self.enabled:
            self.state.record_probe(True, now=now)
            return False
        return self.state.record_probe(healthy, now=now)

    def record_repair(self, *, now: float) -> int:
        return self.state.record_repair(now=now)


def funnel_interval_healthy(funnel: dict, probe_ok: bool | None) -> bool:
    return bool(funnel.get("running") and funnel.get("url") and probe_ok is True)


def auto_repair_configured(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
