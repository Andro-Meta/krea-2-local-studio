from __future__ import annotations

import re


def is_cuda_oom(exc: BaseException) -> bool:
    """Return true only for CUDA-specific out-of-memory failures."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = " ".join(str(exc).lower().split())
    return bool(
        re.search(
            r"(?:cuda out of memory|cuda error:\s*out of memory|out of memory on cuda)",
            message,
        )
    )
