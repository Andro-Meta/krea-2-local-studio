"""DiT block swapping for low-VRAM inference.

Keeps the first `resident_count` transformer blocks on the compute device and
streams the remaining (last N) blocks between system RAM and the GPU around the
forward pass. This is the same idea ComfyUI / kijai's WanVideoWrapper use: only
the resident blocks plus a small prefetch window are ever on the GPU at once, so
a model that would not otherwise fit (e.g. Krea 2 RAW) can run on 24 GB.

Design notes:
- We attach forward pre/post hooks to each swapped block instead of rewriting
  `SingleStreamDiT.forward`, so this is transparent to the model definition and
  composes with the fp8 dequant closures and LoRA forward wrappers (those run
  inside the block's own forward, by which point the block is on the device).
- An optional side CUDA stream prefetches upcoming swapped blocks so the H2D
  copy overlaps with the current block's compute. With `prefetch=0` the transfer
  is synchronous (simpler, slightly slower) which is also what CPU-only test
  runs exercise.
- CPU weights can be pinned once so host->device copies use DMA.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SwapPlan:
    total_blocks: int
    swapped_indices: list[int] = field(default_factory=list)

    @property
    def resident_count(self) -> int:
        return self.total_blocks - len(self.swapped_indices)


def resolve_swap_plan(*, total_blocks: int, blocks_to_swap: int) -> SwapPlan:
    """Pick which block indices to stream from CPU (always the last N)."""
    total_blocks = max(0, int(total_blocks))
    n = max(0, min(int(blocks_to_swap), total_blocks))
    swap_start = total_blocks - n
    return SwapPlan(total_blocks=total_blocks, swapped_indices=list(range(swap_start, total_blocks)))


def _pin_module(module: torch.nn.Module) -> None:
    for param in module.parameters(recurse=True):
        if param.device.type == "cpu" and not param.is_pinned():
            try:
                param.data = param.data.pin_memory()
            except Exception:
                # Pinning is a best-effort DMA speedup; if the allocator can't
                # pin (e.g. low pinned-pool), fall back to pageable memory.
                pass
    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is not None and buf.device.type == "cpu" and not buf.is_pinned():
            try:
                _assign_buffer(module, name, buf.pin_memory())
            except Exception:
                # Best-effort buffer pinning; safe to leave pageable on failure.
                pass


def _assign_buffer(root: torch.nn.Module, dotted_name: str, value: torch.Tensor) -> None:
    *path, leaf = dotted_name.split(".")
    target = root
    for part in path:
        target = getattr(target, part)
    target._buffers[leaf] = value


class BlockSwapController:
    """Streams the last N blocks of `model.<blocks_attr>` between CPU and GPU."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        blocks_to_swap: int,
        device: str = "cuda",
        offload_device: str = "cpu",
        prefetch: int = 1,
        pin_memory: bool = True,
        non_blocking: bool = True,
        blocks_attr: str = "blocks",
    ) -> None:
        self.model = model
        self.blocks = getattr(model, blocks_attr)
        self.device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        self.prefetch = max(0, int(prefetch))
        self.pin_memory = bool(pin_memory)
        self.non_blocking = bool(non_blocking)
        self.plan = resolve_swap_plan(total_blocks=len(self.blocks), blocks_to_swap=blocks_to_swap)
        self._swapped = set(self.plan.swapped_indices)
        self._index_of: dict[int, int] = {id(block): i for i, block in enumerate(self.blocks)}
        self._hooks: list = []
        self._stream = None
        self._events: dict[int, object] = {}
        self.stats = {"swaps_in": 0, "swaps_out": 0, "prefetches": 0}

    @property
    def active(self) -> bool:
        return len(self._swapped) > 0

    def install(self) -> "BlockSwapController":
        if not self.active:
            return self
        use_cuda_stream = self.prefetch > 0 and self.device.type == "cuda" and torch.cuda.is_available()
        if use_cuda_stream:
            self._stream = torch.cuda.Stream(device=self.device)

        for idx, block in enumerate(self.blocks):
            if idx in self._swapped:
                block.to(self.offload_device)
                if self.pin_memory and self.offload_device.type == "cpu":
                    _pin_module(block)
            else:
                block.to(self.device)
            self._hooks.append(block.register_forward_pre_hook(self._pre_hook))
            self._hooks.append(block.register_forward_hook(self._post_hook))
        return self

    def remove(self, *, restore_device: bool = True) -> None:
        """Detach hooks. With restore_device, move swapped blocks back to the
        compute device (use when keeping the model); skip it during unload so we
        don't pull offloaded blocks back into VRAM just before discarding them."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if restore_device:
            for block in self.blocks:
                block.to(self.device)
        self._events.clear()
        self._stream = None

    def _move(self, idx: int, target: torch.device) -> None:
        self.blocks[idx].to(target, non_blocking=self.non_blocking)

    def _prefetch_from(self, current_idx: int) -> None:
        if self.prefetch <= 0:
            return
        for offset in range(1, self.prefetch + 1):
            nxt = current_idx + offset
            if nxt in self._swapped and nxt not in self._events:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        self._move(nxt, self.device)
                    event = torch.cuda.Event()
                    event.record(self._stream)
                    self._events[nxt] = event
                else:
                    self._move(nxt, self.device)
                    self._events[nxt] = True
                self.stats["prefetches"] += 1

    def _pre_hook(self, module: torch.nn.Module, args):
        idx = self._index_of.get(id(module))
        if idx is None or idx not in self._swapped:
            return None
        event = self._events.pop(idx, None)
        if event is None:
            self._move(idx, self.device)
        elif self._stream is not None:
            event.synchronize()
        self.stats["swaps_in"] += 1
        self._prefetch_from(idx)
        return None

    def _post_hook(self, module: torch.nn.Module, args, output):
        idx = self._index_of.get(id(module))
        if idx is None or idx not in self._swapped:
            return None
        self._move(idx, self.offload_device)
        self.stats["swaps_out"] += 1
        return None
