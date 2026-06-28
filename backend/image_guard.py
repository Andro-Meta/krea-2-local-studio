"""Post-render quality guard: detect blank / NaN / degenerate frames.

Cheap sanity checks so the pipeline never silently returns garbage (e.g. a
black frame from an OOM-partial, or NaNs from an unstable fp8/attention path).
Operates on a decoded array or PIL image; returns a small report.
"""
from __future__ import annotations

from typing import Any

# A near-uniform frame has std below this (0-255 scale); near-black also has low mean.
_STD_EPS = 1.0
_BLACK_MEAN = 8.0


def assess_image_array(arr: Any) -> dict:
    import numpy as np

    a = np.asarray(arr, dtype="float32")
    if a.size == 0:
        return {"ok": False, "issue": "empty", "mean": 0.0, "std": 0.0}
    if bool(np.isnan(a).any()) or bool(np.isinf(a).any()):
        return {"ok": False, "issue": "nan", "mean": 0.0, "std": 0.0}
    # Normalize float [0,1] images to 0-255 scale for the thresholds.
    if a.max() <= 1.0:
        a = a * 255.0
    mean = float(a.mean())
    std = float(a.std())
    issue = None
    if std < _STD_EPS and mean < _BLACK_MEAN:
        issue = "black"
    elif std < _STD_EPS:
        issue = "uniform"
    return {"ok": issue is None, "issue": issue, "mean": round(mean, 3), "std": round(std, 3)}


def assess_image(image: Any) -> dict:
    """Accept a PIL image (or array) and return a quality report."""
    import numpy as np

    if hasattr(image, "convert"):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    return assess_image_array(arr)


def summarize_quality(reports: list[dict]) -> dict:
    bad = [r for r in reports if not r.get("ok", True)]
    return {
        "bad_count": len(bad),
        "total": len(reports),
        "all_bad": len(reports) > 0 and len(bad) == len(reports),
        "issues": [r.get("issue") for r in bad if r.get("issue")],
    }
