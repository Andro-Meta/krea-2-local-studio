from __future__ import annotations

import sys
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@unittest.skipIf(torch is None, "torch is required for loader parity tests")
class ComfyLoaderParityTests(unittest.TestCase):
    def test_nvidia_smi_memory_csv_is_parsed_in_gib(self) -> None:
        from system_check import parse_nvidia_smi_memory_csv

        parsed = parse_nvidia_smi_memory_csv("NVIDIA GeForce RTX 4090, 24564 MiB, 9353 MiB, 14786 MiB\n")

        self.assertEqual(parsed["name"], "NVIDIA GeForce RTX 4090")
        self.assertAlmostEqual(parsed["total_gb"], 24564 / 1024, places=2)
        self.assertAlmostEqual(parsed["free_gb"], 9353 / 1024, places=2)
        self.assertAlmostEqual(parsed["used_gb"], 14786 / 1024, places=2)

    def test_gpu_process_csv_keeps_pid_and_names(self) -> None:
        from system_check import parse_nvidia_smi_process_csv

        parsed = parse_nvidia_smi_process_csv(
            "35916, C:\\Python314\\python.exe, 8000 MiB\n"
            "47808, C:\\Python314\\python.exe, 6400 MiB\n",
            current_pid=35916,
        )

        self.assertEqual(parsed, [{"pid": 47808, "name": "python.exe", "used_memory_gb": 6.25}])

    def test_system_report_includes_gpu_process_details(self) -> None:
        system_check = importlib.import_module("system_check")

        with patch.object(system_check, "get_gpu_info", return_value=("RTX 4090", 24.0, 9.0)), \
             patch.object(system_check, "get_ram_gb", return_value=(32.0, 16.0)), \
             patch.object(system_check, "get_disk_free_gb", return_value=100.0), \
             patch.object(system_check, "get_gpu_process_details", return_value=[{"pid": 123, "name": "python.exe"}]):
            report = system_check.get_system_report()

        self.assertEqual(report["gpu_process_details"], [{"pid": 123, "name": "python.exe"}])

    def test_loader_preflight_fails_before_load_when_vram_is_low(self) -> None:
        from inference import preflight_model_load

        with patch("inference.get_ram_gb", return_value=(32.0, 20.0)), \
             patch("inference.get_gpu_info", return_value=("RTX 4090", 24.0, 9.0)), \
             patch("inference.get_gpu_process_details", return_value=[{"pid": 123, "name": "python.exe"}]):
            with self.assertRaisesRegex(RuntimeError, "Only 9.0GB VRAM free"):
                preflight_model_load("krea2_turbo_fp8_scaled.safetensors", "fp8")

    def test_loader_preflight_fails_before_load_when_ram_is_low(self) -> None:
        from inference import preflight_model_load

        with patch("inference.get_ram_gb", return_value=(32.0, 6.0)), \
             patch("inference.get_gpu_info", return_value=("RTX 4090", 24.0, 18.0)), \
             patch("inference.get_gpu_process_details", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "Only 6.0GB system RAM free"):
                preflight_model_load("krea2_turbo_fp8_scaled.safetensors", "fp8")

    def test_raw_bf16_preflight_fails_on_32gb_ram_systems(self) -> None:
        from inference import preflight_model_load

        with patch("inference.get_ram_gb", return_value=(32.0, 24.0)), \
             patch("inference.get_gpu_info", return_value=("RTX 4090", 24.0, 22.0)), \
             patch("inference.get_gpu_process_details", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "needs ~48GB system RAM"):
                preflight_model_load("krea2_raw_bf16.safetensors", "bf16")

    def test_raw_as_dynamic_fp8_is_allowed_on_24gb_card(self) -> None:
        from inference import preflight_model_load

        # RAW file requested as fp8 → dynamic fp8 path; must NOT hit the bf16
        # 48GB-RAM gate just because the filename contains "raw".
        with patch("inference.get_ram_gb", return_value=(32.0, 24.0)), \
             patch("inference.get_gpu_info", return_value=("RTX 4090", 24.0, 22.0)), \
             patch("inference.get_gpu_process_details", return_value=[]):
            preflight_model_load("krea2_raw_bf16.safetensors", "fp8")

    def test_block_swap_relaxes_fp8_vram_requirement(self) -> None:
        from inference import preflight_model_load

        # 10GB free VRAM would fail the ~13GB fp8 gate, but streaming 8 blocks
        # lowers the resident requirement enough to pass.
        with patch("inference.get_ram_gb", return_value=(32.0, 24.0)), \
             patch("inference.get_gpu_info", return_value=("RTX 4090", 24.0, 10.0)), \
             patch("inference.get_gpu_process_details", return_value=[]):
            preflight_model_load("krea2_turbo_fp8_scaled.safetensors", "fp8", blocks_to_swap=8)

    def test_raw_bf16_preflight_fails_when_vram_is_low(self) -> None:
        from inference import preflight_model_load

        with patch("inference.get_ram_gb", return_value=(64.0, 48.0)), \
             patch("inference.get_gpu_info", return_value=("RTX 4090", 24.0, 12.0)), \
             patch("inference.get_gpu_process_details", return_value=[{"pid": 123, "name": "python.exe"}]):
            with self.assertRaisesRegex(RuntimeError, "Only 12.0GB VRAM free"):
                preflight_model_load("krea2_raw_bf16.safetensors", "bf16")

    def test_load_model_error_detail_adds_system_hint(self) -> None:
        import main

        detail = main._load_model_error_detail(RuntimeError("Only 9.0GB VRAM free"))

        self.assertIn("Only 9.0GB VRAM free", detail)
        self.assertIn("System tab", detail)

    def test_fp8_state_dict_loader_extracts_scales_without_extra_keys(self) -> None:
        from inference import load_fp8_scaled_state_dict
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny_fp8.safetensors"
            save_file({
                "layer.weight": torch.ones(2, 2, dtype=torch.float8_e4m3fn),
                "layer.weight_scale": torch.tensor(0.5, dtype=torch.float32),
            }, str(path))

            sd, scales = load_fp8_scaled_state_dict(path)

        self.assertEqual(list(sd.keys()), ["layer.weight"])
        self.assertEqual(scales, {"layer": 0.5})

    def test_support_registry_includes_comfy_fp8_text_encoder(self) -> None:
        from support_models import _model_by_id

        model = _model_by_id("qwen3_vl_fp8")

        self.assertEqual(model["repo_id"], "Comfy-Org/Krea-2")
        self.assertIn("text_encoders/qwen3vl_4b_fp8_scaled.safetensors", model["allow_patterns"])

    def test_conditioner_source_prefers_installed_comfy_fp8_asset(self) -> None:
        from krea2.encoder import resolve_conditioner_source

        with tempfile.TemporaryDirectory() as td:
            fp8_path = Path(td) / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors"
            fp8_path.parent.mkdir(parents=True)
            fp8_path.write_bytes(b"tiny placeholder")
            with patch("krea2.encoder.support_model_path", return_value=Path(td)):
                source = resolve_conditioner_source()

        self.assertEqual(source["kind"], "comfy_fp8")
        self.assertEqual(source["path"], str(fp8_path))
        self.assertEqual(source["runtime"], "hf_bf16_fallback")
        self.assertEqual(source["status"], "FP8 asset installed; runtime unsupported, using HF BF16 fallback")

    def test_conditioner_source_falls_back_to_hf_model(self) -> None:
        from krea2.encoder import resolve_conditioner_source

        with patch("krea2.encoder.support_model_path", side_effect=FileNotFoundError("missing")):
            source = resolve_conditioner_source()

        self.assertEqual(source["kind"], "hf_bf16")
        self.assertEqual(source["runtime"], "hf_bf16")
        self.assertIn("Qwen3-VL-4B", source["path"])

    def test_conditioner_source_env_forces_bf16_fallback(self) -> None:
        from krea2.encoder import resolve_conditioner_source

        with tempfile.TemporaryDirectory() as td:
            fp8_path = Path(td) / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors"
            fp8_path.parent.mkdir(parents=True)
            fp8_path.write_bytes(b"tiny placeholder")
            with patch("krea2.encoder.support_model_path", return_value=Path(td)), \
                 patch.dict("os.environ", {"KREA2_TEXT_ENCODER": "bf16"}):
                source = resolve_conditioner_source()

        self.assertEqual(source["kind"], "hf_bf16")
        self.assertEqual(source["runtime"], "hf_bf16")
