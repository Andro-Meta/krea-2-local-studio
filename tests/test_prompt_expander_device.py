from __future__ import annotations

import sys
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

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


def _fake_transformers_module() -> ModuleType:
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(from_pretrained=Mock())
    module.Qwen3VLProcessor = SimpleNamespace(from_pretrained=Mock())
    module.Qwen3VLForConditionalGeneration = SimpleNamespace(from_pretrained=Mock())
    return module


class PromptExpanderDeviceTests(unittest.TestCase):
    def test_recreate_prompt_requires_literal_unbeautified_fidelity(self) -> None:
        import prompt_expander

        prompt = prompt_expander.DESCRIBE_PROMPTS["recreate"].lower()
        for required in (
            "literal",
            "distress",
            "injury",
            "horror",
            "imperfections",
            "do not beautify",
            "do not sanitize",
        ):
            with self.subTest(required=required):
                self.assertIn(required, prompt)

    def test_auto_fails_fast_when_vram_is_tight(self) -> None:
        import prompt_expander
        from settings import settings

        with patch.object(settings, "local_qwen_device", "auto"):
            with self.assertRaisesRegex(RuntimeError, "Auto mode no longer falls back to CPU"):
                prompt_expander._resolve_local_qwen_device(_FakeTorch(free_gb=8.0))
    def test_shared_classifier_is_precise(self) -> None:
        from gpu_recovery import is_cuda_oom

        class FakeCudaOutOfMemoryError(RuntimeError):
            pass

        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(OutOfMemoryError=FakeCudaOutOfMemoryError)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertTrue(is_cuda_oom(FakeCudaOutOfMemoryError("allocation")))
            self.assertTrue(is_cuda_oom(RuntimeError("CUDA out of memory")))
            self.assertFalse(is_cuda_oom(RuntimeError("CPU out of memory")))

    def test_prompt_expander_reraises_cuda_oom_but_soft_fails_other_errors(self):
        import prompt_expander

        with patch(
            "comfy_qwen_vl.expand_prompt_comfy",
            side_effect=RuntimeError("CUDA out of memory"),
        ), self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            prompt_expander.expand_prompt_comfy("fox")

        with patch(
            "comfy_qwen_vl.expand_prompt_comfy",
            side_effect=RuntimeError("ordinary failure"),
        ):
            result = prompt_expander.expand_prompt_comfy("fox")
        self.assertIn("ordinary failure", result.error)

    def test_prompt_planner_reraises_cuda_oom_but_keeps_heuristic_fallback(self):
        import prompt_planner

        with patch(
            "comfy_qwen_vl.expand_prompt_comfy",
            side_effect=RuntimeError("CUDA error: out of memory"),
        ), self.assertRaisesRegex(RuntimeError, "out of memory"):
            prompt_planner.plan_prompt_comfy("fox")

        with patch(
            "comfy_qwen_vl.expand_prompt_comfy",
            side_effect=RuntimeError("ordinary failure"),
        ):
            result = prompt_planner.plan_prompt_comfy("fox")
        self.assertEqual(result.backend, "heuristic")
        self.assertIn("ordinary failure", result.error)

    def test_moodboard_reraises_cuda_oom_without_transformers_fallback(self):
        import moodboard_enrichment
        from settings import settings

        with (
            patch.object(settings, "local_llm_backend", "comfy"),
            patch.object(
                moodboard_enrichment,
                "_comfy_qwen_generate",
                side_effect=RuntimeError("out of memory on CUDA"),
            ),
            patch.object(moodboard_enrichment, "_local_qwen_generate") as local,
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            moodboard_enrichment._qwen_generate("prompt", [])
        local.assert_not_called()

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
        from settings import settings

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
            patch.object(settings, "local_llm_backend", "transformers"),
            patch.object(prompt_expander, "_load_local_qwen", return_value=(FakeTokenizer(), None, FakeModel())),
            patch.object(prompt_expander, "unload_local_qwen_after_use") as unload,
        ):
            result = prompt_expander.expand_prompt_local("a fox")

        self.assertTrue(result.changed)
        unload.assert_called_once()

    def test_explicit_transformers_cuda_frees_comfy_before_model_load(self) -> None:
        import prompt_expander

        fake_torch = ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        fake_transformers = _fake_transformers_module()
        with (
            patch.dict(
                sys.modules,
                {"torch": fake_torch, "transformers": fake_transformers},
            ),
            patch.object(prompt_expander, "_resolve_local_qwen_device", return_value="cuda"),
            patch.object(prompt_expander, "free_comfy_vram") as free,
            patch("settings.settings.local_llm_backend", "transformers"),
            patch.object(
                fake_transformers.AutoTokenizer,
                "from_pretrained",
                side_effect=RuntimeError("stop after free"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after free"):
                prompt_expander._load_local_qwen.__wrapped__("fake-model")

        free.assert_called_once_with(unload_models=True, free_memory=True)

    def test_transformers_cuda_aborts_when_comfy_vram_release_fails(self) -> None:
        import prompt_expander

        fake_torch = ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        fake_transformers = _fake_transformers_module()
        with (
            patch.dict(
                sys.modules,
                {"torch": fake_torch, "transformers": fake_transformers},
            ),
            patch.object(prompt_expander, "_resolve_local_qwen_device", return_value="cuda"),
            patch.object(prompt_expander, "free_comfy_vram", return_value=False),
            patch("settings.settings.local_llm_backend", "transformers"),
            patch.object(fake_transformers.AutoTokenizer, "from_pretrained") as tokenizer_loader,
            patch.object(fake_transformers.Qwen3VLProcessor, "from_pretrained") as processor_loader,
            patch.object(fake_transformers.Qwen3VLForConditionalGeneration, "from_pretrained") as model_loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI VRAM"):
                prompt_expander._load_local_qwen.__wrapped__("fake-model")

        tokenizer_loader.assert_not_called()
        processor_loader.assert_not_called()
        model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
