try:
    from .gpu_task_queue import (
        BACKGROUND,
        INTERACTIVE,
        EnqueueResult,
        GpuTaskQueue,
    )
except ImportError:
    from gpu_task_queue import (  # type: ignore[no-redef]
        BACKGROUND,
        INTERACTIVE,
        EnqueueResult,
        GpuTaskQueue,
    )


class GenerationQueue(GpuTaskQueue):
    def __init__(self, handler):
        super().__init__(handler, enforce_limits=False)

    def enqueue(
        self,
        task_id,
        payload,
        *,
        username,
        role,
        task_kind="generation",
        priority_class=INTERACTIVE,
    ):
        result = super().enqueue(
            task_id,
            payload,
            username=username,
            role=role,
            task_kind=task_kind,
            priority_class=priority_class,
        )
        assert result.record is not None
        return dict(result.record)

__all__ = [
    "BACKGROUND",
    "INTERACTIVE",
    "EnqueueResult",
    "GenerationQueue",
    "GpuTaskQueue",
]
