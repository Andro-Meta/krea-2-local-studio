"""Timestep-density schedulers for the Krea flow sampler (torch-free).

The Krea sampler integrates a flow-matching ODE on t: 1 -> 0 and applies an
exp(mu) time-shift (ComfyUI ``flux_time_shift``). A *scheduler* reshapes the
*base* 1->0 grid BEFORE that shift, changing where steps are spent.

We port the flow-appropriate ComfyUI schedulers (``comfy/samplers.py``):
  - ``simple`` / ``normal``: uniform in t (the existing Krea default).
  - ``sgm_uniform``: uniform but drops the sigma_min endpoint (denser near 0).
  - ``beta``: Beta(0.6, 0.6) U-shaped spacing (arXiv:2407.12173) — the most
    cited "crisper detail" scheduler for Flux/flow models.
  - ``karras`` / ``exponential``: EDM-shaped; offered for experimentation but
    flagged not-recommended for flow models (they assume EDM sigma ranges).

Everything here is pure ``math`` so it stays unit-testable without torch.
"""
from __future__ import annotations

import math

FLOW_SCHEDULERS: tuple[str, ...] = ("simple", "normal", "beta", "sgm_uniform")
ALL_SCHEDULERS: tuple[str, ...] = FLOW_SCHEDULERS + ("karras", "exponential")

# Default Beta(alpha, beta) shape parameters (ComfyUI beta_scheduler defaults).
BETA_ALPHA = 0.6
BETA_BETA = 0.6


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Numerical Recipes)."""
    MAXIT = 300
    EPS = 3.0e-12
    FPMIN = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, alpha: float = BETA_ALPHA, beta: float = BETA_BETA) -> float:
    """Inverse CDF (quantile) of Beta(alpha, beta) via bisection.

    Mirrors ``scipy.stats.beta.ppf`` closely enough for schedule construction
    without the scipy dependency. Monotonic, endpoints exact.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(alpha, beta, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _karras_grid(steps: int, *, rho: float = 7.0, sigma_min: float = 1.0e-3) -> list[float]:
    # Karras rho schedule remapped onto a normalized [sigma_min, 1] range.
    n = steps
    if n <= 1:
        return [1.0, 0.0]
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = 1.0 ** (1.0 / rho)
    grid = [(max_inv + (k / (n - 1)) * (min_inv - max_inv)) ** rho for k in range(n)]
    grid.append(0.0)
    return grid


def _exponential_grid(steps: int, *, sigma_min: float = 1.0e-3) -> list[float]:
    n = steps
    if n <= 1:
        return [1.0, 0.0]
    log_max = math.log(1.0)
    log_min = math.log(sigma_min)
    grid = [math.exp(log_max + (k / (n - 1)) * (log_min - log_max)) for k in range(n)]
    grid.append(0.0)
    return grid


def base_grid(steps: int, scheduler: str = "simple") -> list[float]:
    """Return the base flow-time grid (len steps+1, descending 1 -> 0).

    This grid is fed through the exp(mu) time-shift by the sampler. ``simple``
    and ``normal`` reproduce the existing uniform default exactly.
    """
    n = max(1, int(steps))
    name = str(scheduler or "simple").strip().lower()

    if name in ("simple", "normal", ""):
        return [1.0 - k / n for k in range(n + 1)]

    if name == "sgm_uniform":
        grid = [1.0 - k / n for k in range(n)]
        grid.append(0.0)
        return grid

    if name == "beta":
        # ComfyUI: ts = 1 - linspace(0,1,steps, endpoint=False); beta.ppf(ts);
        # we keep it in normalized time and let the mu-shift apply afterwards.
        grid = [beta_ppf(1.0 - k / n) for k in range(n)]
        grid.append(0.0)
        return grid

    if name == "karras":
        return _karras_grid(n)

    if name == "exponential":
        return _exponential_grid(n)

    raise ValueError(f"Unknown scheduler: {scheduler}")
