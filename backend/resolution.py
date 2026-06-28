"""Resolution tiers and aspect-ratio -> dimension computation.

"1K" / "2K" are defined by the **long side** (1024 / 2048), which keeps results
predictable and within the 2048 hard cap for every aspect ratio. All dimensions
are aligned to the model's patch grid (multiple of 16 = COMPRESSION*PATCH).

Torch-free so it can be unit-tested in lightweight CI and mirrored on the
frontend. The backend also uses `normalize_dimensions` to guarantee any inbound
width/height (including custom values) is aligned and within range before
sampling.
"""
from __future__ import annotations

ALIGN = 16            # COMPRESSION (8) * PATCH (2)
MIN_EDGE = 256
MAX_EDGE = 2048

RESOLUTION_TIERS: dict[str, int] = {"1k": 1024, "2k": 2048}

# width:height ratios
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}


def _round_align(value: float, align: int = ALIGN) -> int:
    return max(align, int(round(value / align)) * align)


def normalize_dimensions(width: int, height: int, *, max_edge: int = MAX_EDGE, min_edge: int = MIN_EDGE) -> tuple[int, int]:
    """Clamp to [min_edge, max_edge] and align both dims to the patch grid."""
    w = _round_align(max(min_edge, min(int(width or min_edge), max_edge)))
    h = _round_align(max(min_edge, min(int(height or min_edge), max_edge)))
    return w, h


def compute_dimensions(aspect: str, tier: str, *, max_edge: int = MAX_EDGE) -> tuple[int, int]:
    """Resolve (width, height) for an aspect ratio at a 1k/2k tier.

    The tier sets the long side; the short side is derived from the ratio and
    aligned to the patch grid. Unknown aspect/tier falls back to a square tile.
    """
    long_side = RESOLUTION_TIERS.get(str(tier), RESOLUTION_TIERS["1k"])
    long_side = min(long_side, max_edge)
    rw, rh = ASPECT_RATIOS.get(str(aspect), (1, 1))
    if rw >= rh:
        width = long_side
        height = _round_align(long_side * rh / rw)
    else:
        height = long_side
        width = _round_align(long_side * rw / rh)
    return min(width, max_edge), min(height, max_edge)


def resolution_options() -> dict:
    """Payload for the UI: tiers, aspect ratios, and the resolved grid."""
    return {
        "tiers": list(RESOLUTION_TIERS.keys()),
        "aspects": list(ASPECT_RATIOS.keys()),
        "dimensions": {
            tier: {aspect: list(compute_dimensions(aspect, tier)) for aspect in ASPECT_RATIOS}
            for tier in RESOLUTION_TIERS
        },
    }
