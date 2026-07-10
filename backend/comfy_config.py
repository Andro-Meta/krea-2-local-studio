"""Feature toggle for the ComfyUI backend adapter.

When enabled (the default), image generation is routed to a local ComfyUI
server via comfy_workflows.comfy_generate instead of the in-process native
PyTorch pipeline. Set KREA_USE_COMFY=0 to fall back to the legacy native
engine.
"""
from __future__ import annotations

import os

_FALSEY = {"0", "false", "no", "off", ""}


def use_comfy_backend() -> bool:
    # Phase 1: native Studio DiT is deprecated. Always route image work to ComfyUI.
    # KREA_USE_COMFY=0 is ignored (kept only so old .env files don't surprise anyone).
    return True
