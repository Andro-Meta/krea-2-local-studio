"""Comfy-only stand-in for the legacy in-process Krea pipeline.

Phase 1 deprecation: Studio no longer loads DiT weights into its own venv.
Image generation / upscale / depth run in ComfyUI. This stub keeps call sites
in main.py (memory helpers, system report, load/unload endpoints) working
without importing torch or backend.inference / backend.krea2.
"""
from __future__ import annotations

from typing import Any


class ComfyPipelineStub:
    """Minimal surface matching what main.py still expects from `pipeline`."""

    _loading = False
    _loaded_checkpoint: str | None = None
    _loaded_quant: str | None = None
    _last_load_error: str | None = None
    _text_encoder_source: str | None = None
    _device = "comfyui"
    _dtype = None
    ae = None

    def is_loaded(self) -> bool:
        return False

    def memory_status(self) -> str:
        try:
            from system_check import mem_snapshot

            return mem_snapshot()
        except Exception:
            return "ComfyUI backend (native Studio pipeline deprecated)"

    def release_transient_memory(self, *, clear_conditioning_cache: bool = True) -> dict[str, Any]:
        return {
            "released": True,
            "encoder_loaded": False,
            "cleared_conditioning_entries": 0,
            "memory": self.memory_status(),
            "native_pipeline": False,
        }

    def unload(self) -> None:
        return None

    def load(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Native Studio model loading is deprecated. Image generation runs in ComfyUI "
            "(KREA_USE_COMFY=1). Start ComfyUI and generate — no in-process load is needed."
        )

    def generate(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Native Studio generation is deprecated. Set KREA_USE_COMFY=1 (default) and use ComfyUI."
        )


pipeline = ComfyPipelineStub()
