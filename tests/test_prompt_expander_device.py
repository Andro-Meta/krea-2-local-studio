from __future__ import annotations

import sys
import asyncio
import tempfile
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
    def test_auto_fails_fast_when_vram_is_tight(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            with self.assertRaisesRegex(RuntimeError, "Auto mode no longer falls back to CPU"):
                prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=8.0))

    def test_auto_uses_cuda_when_vram_is_plentiful(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            self.assertEqual(prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=18.0)), "cuda")

    def test_auto_fails_fast_when_krea_pipeline_already_allocated_cuda(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            with self.assertRaisesRegex(RuntimeError, "already has 10.0GB CUDA allocated"):
                prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=13.2, allocated_gb=10.0))

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

    def test_unload_local_qwen_uses_helper_lock(self) -> None:
        import prompt_expander

        calls: list[str] = []

        class FakeLock:
            def __enter__(self):
                calls.append("enter")

            def __exit__(self, *args):
                calls.append("exit")

        with patch.object(prompt_expander, "_LOCAL_QWEN_LOCK", FakeLock()):
            prompt_expander.unload_local_qwen()

        self.assertEqual(calls, ["enter", "exit"])

    def test_expand_prompt_local_unloads_after_use(self) -> None:
        import prompt_expander

        class FakeTokenizer:
            eos_token_id = 1

            def apply_chat_template(self, *_args, **_kwargs):
                class Inputs:
                    shape = (1, 2)

                    def to(self, _device):
                        return self

                return Inputs()

            def decode(self, *_args, **_kwargs):
                return "expanded prompt"

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[1, 2, 3]]

        with (
            patch.object(prompt_expander, "_load_local_qwen", return_value=(FakeTokenizer(), None, FakeModel())),
            patch.object(prompt_expander, "unload_local_qwen_after_use") as unload,
        ):
            result = prompt_expander.expand_prompt_local("a fox")

        self.assertTrue(result.changed)
        unload.assert_called_once()

    def test_local_helper_preempts_and_reloads_loaded_krea_model(self) -> None:
        import main
        from settings import settings

        class FakePipeline:
            _loaded_quant = "int8"
            _loading = False

            def __init__(self, checkpoint: str):
                self._loaded_checkpoint = checkpoint
                self.calls: list[str] = []

            def is_loaded(self):
                return True

            def unload(self):
                self.calls.append("unload")

            def load(self, checkpoint, quantization, **_kwargs):
                self.calls.append(f"load:{Path(checkpoint).name}:{quantization}")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.safetensors"
            checkpoint.write_bytes(b"x")
            fake = FakePipeline(str(checkpoint))

            async def run():
                with (
                    patch.object(main, "pipeline", fake),
                    patch.object(main, "generation_queue", None),
                    patch.object(main, "clear_cuda_cache") as clear,
                    patch.object(settings, "local_llm_backend", "transformers"),
                    patch.object(settings, "local_qwen_device", "auto"),
                ):
                    result = await main._run_helper_with_optional_krea_preempt("local", lambda: "ok")
                return result, clear.called

            result, cleared = asyncio.run(run())

        self.assertEqual(result, "ok")
        self.assertTrue(cleared)
        self.assertEqual(fake.calls, ["unload", "load:model.safetensors:int8"])


if __name__ == "__main__":
    unittest.main()
