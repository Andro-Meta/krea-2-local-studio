"""Torch-free text-prompt template helpers for the Krea 2 encoder.

Kept dependency-light (no torch) so the template logic can be unit-tested in the
lightweight CI environment and reused by the encoder.

`<think>` expression steering (Banodoco/Tests-and-Findings 6/27): Krea 2's text
encoder slices conditioning from the second `<|im_start|>` (the user turn)
onward, discarding the system turn. The surviving, directly-steerable write
points are the user turn and the assistant turn. Appending a short
`<think>...</think>` reasoning span to the assistant turn behaves like a steering
vector that restores expression/intensity Turbo's distillation flattens, while
staying in-distribution (unlike heavy conditioning rebalance, which over-
saturates and harms text). The span must ride on the assistant suffix so it
lands after the slice point and is kept as conditioning.
"""
from __future__ import annotations

# A safe, generic default that nudges expression/intensity without changing the
# requested subject. Users can override with their own think text.
DEFAULT_EXPRESSION_THINK = (
    "Render the described expression, emotion, and intensity fully and honestly, "
    "including tension, strong feelings, and dramatic lighting where implied, while "
    "staying photorealistic and faithful to the user's subject and composition."
)

MAX_THINK_CHARS = 600


def assistant_suffix(base_suffix: str, think: str | None) -> str:
    """Return the assistant-turn suffix, optionally carrying a `<think>` span.

    With no think text the base suffix is returned unchanged (bitwise-identical
    behavior), so this is safe to call unconditionally.
    """
    text = (think or "").strip()
    if not text:
        return base_suffix
    text = text[:MAX_THINK_CHARS]
    return f"{base_suffix}<think>{text}</think>"
