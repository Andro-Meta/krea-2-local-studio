from __future__ import annotations

def mock_comfy_task_capabilities(main_module) -> None:
    """Keep mocked GPU-worker tests independent from a real ComfyUI server."""
    main_module.comfy_available = lambda: True
    main_module.comfy_atomic_cancel_available = lambda: True


mock_atomic_cancel_capability = mock_comfy_task_capabilities
