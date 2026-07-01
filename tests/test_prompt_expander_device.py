from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class _FakeCuda:
    def __init__(self, free_gb: float, total_gb: float = 24.0, available: bool = True, allocated_gb: float = 0.0) -> None:
        self.free_gb = free_gb
        self.total_gb = total_gb
        self.available = available
        self.allocated_gb = allocated_gb

    def is_available(self) -> bool:
        return self.available

    def mem_get_info(self):
        return int(self.free_gb * 1024**3), int(self.total_gb * 1024**3)

    def memory_allocated(self) -> int:
        return int(self.allocated_gb * 1024**3)


class _FakeTorch:
    def __init__(self, free_gb: float, available: bool = True, allocated_gb: float = 0.0) -> None:
        self.cuda = _FakeCuda(free_gb=free_gb, available=available, allocated_gb=allocated_gb)


class PromptExpanderDeviceTests(unittest.TestCase):
    def test_auto_uses_cpu_when_vram_is_tight(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            self.assertEqual(prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=8.0)), "cpu")

    def test_auto_uses_cuda_when_vram_is_plentiful(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            self.assertEqual(prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=18.0)), "cuda")

    def test_auto_uses_cpu_when_krea_pipeline_already_allocated_cuda(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            self.assertEqual(prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=13.2, allocated_gb=10.0)), "cpu")

    def test_explicit_device_overrides_auto_policy(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "cpu"):
            self.assertEqual(prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=24.0)), "cpu")
        with patch.object(settings, "local_qwen_device", "cuda"):
            self.assertEqual(prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=1.0)), "cuda")

    def test_unload_local_qwen_clears_loader_cache(self) -> None:
        import prompt_expander

        prompt_expander._load_local_qwen.cache_clear()
        with patch("prompt_expander._load_local_qwen", side_effect=RuntimeError("unused")) as loader:
            # The real function is replaced here, so this asserts the public hook is callable.
            prompt_expander.unload_local_qwen()
            loader.cache_clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
