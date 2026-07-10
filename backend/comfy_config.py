"""ComfyUI backend routing.

ComfyUI is the generation engine: all image work is routed to the ComfyUI
server (local by default, or wherever KREA_COMFY_URL points) via
comfy_workflows.comfy_generate. The legacy in-process PyTorch pipeline was
removed, so this always returns True; it remains a function because call
sites read like a capability check.
"""
from __future__ import annotations


def use_comfy_backend() -> bool:
    return True
