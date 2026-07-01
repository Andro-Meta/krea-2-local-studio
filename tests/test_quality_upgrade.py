from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class QualityUpgradeTests(unittest.TestCase):
    def test_negative_lora_strength_does_not_inject_positive_trigger_words(self) -> None:
        import lora_manager

        prompt = lora_manager.build_trigger_prompt(
            "a red fox",
            [{"name": "krea2_darkbrush", "enabled": True, "strength": -0.7}],
        )

        self.assertEqual(prompt, "a red fox")

    def test_lokr_lora_inspection_and_application(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is required for LoKr application tests")
        from safetensors.torch import save_file
        import lora_manager

        class Attention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wq = torch.nn.Linear(4, 4, bias=False)

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = Attention()

        class TinyKrea(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block()])

        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "tiny_lokr.safetensors"
            w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            w2 = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
            save_file({
                "diffusion_model.blocks.0.attn.wq.lokr_w1": w1,
                "diffusion_model.blocks.0.attn.wq.lokr_w2": w2,
                "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(1.0),
            }, str(lora_path))

            model = TinyKrea()
            model.blocks[0].attn.wq.weight.data.zero_()
            verdict = lora_manager.inspect_lora(lora_path, model=model)
            self.assertTrue(verdict["compatible"], verdict)
            self.assertEqual(verdict["format"], "lokr")

            with patch.object(lora_manager, "LORAS_DIR", Path(td)):
                reports = lora_manager.apply_loras(
                    model,
                    [{"name": "tiny_lokr", "filename": "tiny_lokr.safetensors", "strength": 0.5}],
                    device="cpu",
                )

            x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            expected = torch.nn.functional.linear(x, torch.kron(w1, w2) * 0.5)
            self.assertTrue(reports[0]["applied"], reports)
            self.assertTrue(torch.allclose(model.blocks[0].attn.wq(x), expected))

    def test_direct_diff_adapter_inspection_and_application(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is required for direct diff adapter tests")
        from safetensors.torch import save_file
        import lora_manager

        class TinyFusion(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projector = torch.nn.Linear(4, 1, bias=False)

        class TinyKrea(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.txtfusion = TinyFusion()

        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "tiny_diff.safetensors"
            diff = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
            save_file({"diffusion_model.txtfusion.projector.diff": diff}, str(lora_path))

            model = TinyKrea()
            model.txtfusion.projector.weight.data.zero_()
            verdict = lora_manager.inspect_lora(lora_path, model=model)
            self.assertTrue(verdict["compatible"], verdict)
            self.assertEqual(verdict["format"], "diff")

            with patch.object(lora_manager, "LORAS_DIR", Path(td)):
                reports = lora_manager.apply_loras(
                    model,
                    [{"name": "tiny_diff", "filename": "tiny_diff.safetensors", "strength": 0.25}],
                    device="cpu",
                )

            x = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
            expected = torch.nn.functional.linear(x, diff * 0.25)
            self.assertTrue(reports[0]["applied"], reports)
            self.assertTrue(torch.allclose(model.txtfusion.projector(x), expected))

    def test_realtime_preview_request_accepts_shared_moodboards(self) -> None:
        from schemas import RealtimePreviewRequest

        req = RealtimePreviewRequest(
            session_id="rt-test",
            prompt="a castle",
            canvas_image_b64="abc",
            mood="cinematic",
            moodboard_ids=[12, 34],
            moodboard_uuids=["one", "two"],
            moodboard_images=["style-ref"],
            moodboard_strength=0.4,
            loras=[{"name": "krea2_darkbrush", "strength": -0.5, "enabled": True}],
        )

        self.assertEqual(req.mood, "cinematic")
        self.assertEqual(req.moodboard_ids, [12, 34])
        self.assertEqual(req.moodboard_uuids, ["one", "two"])
        self.assertEqual(req.moodboard_images, ["style-ref"])
        self.assertEqual(req.moodboard_strength, 0.4)
        self.assertEqual(req.loras[0]["strength"], -0.5)

    def test_provider_auto_uses_krea_when_flux_missing(self) -> None:
        import edit_providers

        provider = edit_providers.resolve_edit_provider("auto", "inpaint", flux_fill_installed=False)
        self.assertEqual(provider.name, "krea_native")
        self.assertIn("FLUX Fill", provider.reason)

    def test_provider_auto_uses_flux_for_strict_edits_when_available(self) -> None:
        import edit_providers

        provider = edit_providers.resolve_edit_provider("auto", "outpaint", flux_fill_installed=True)
        self.assertEqual(provider.name, "flux_fill")

    def test_mask_crop_expands_and_composites_back(self) -> None:
        import mask_editing

        image = Image.new("RGB", (128, 128), "navy")
        mask = Image.new("L", (128, 128), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((54, 50, 74, 72), fill=255)

        crop = mask_editing.crop_for_mask(image, mask, padding=16, align=16)
        self.assertEqual(crop.box, (32, 32, 96, 96))
        self.assertEqual(crop.image.size, (64, 64))
        self.assertEqual(crop.mask.size, (64, 64))

        generated = Image.new("RGB", crop.image.size, "orange")
        composited = mask_editing.composite_crop(image, generated, crop.feathered_mask, crop.box)
        self.assertEqual(composited.size, image.size)
        self.assertNotEqual(composited.getpixel((64, 60)), image.getpixel((64, 60)))
        self.assertEqual(composited.getpixel((5, 5)), image.getpixel((5, 5)))

    def test_output_encoder_can_skip_disk_writes_for_previews(self) -> None:
        import output_saver

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            image = Image.new("RGB", (16, 16), "purple")
            encoded, filenames = output_saver.encode_images([image], out_dir, save_outputs=False)
            saved_files = list(out_dir.glob("*.png"))

        self.assertEqual(len(encoded), 1)
        self.assertEqual(filenames, [])
        self.assertEqual(saved_files, [])

    def test_output_encoder_embeds_generation_metadata(self) -> None:
        import output_saver

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            image = Image.new("RGB", (16, 16), "purple")
            metadata = {"prompt": "a purple cube", "seed": 123, "steps": 8}

            encoded, filenames = output_saver.encode_images([image], out_dir, metadata=[metadata])
            with Image.open(out_dir / filenames[0]) as saved_img:
                saved_info = dict(saved_img.info)
            with Image.open(__import__("io").BytesIO(__import__("base64").b64decode(encoded[0]))) as roundtrip_img:
                roundtrip_info = dict(roundtrip_img.info)

        self.assertEqual(json.loads(saved_info["krea2_metadata"])["prompt"], "a purple cube")
        self.assertEqual(json.loads(roundtrip_info["krea2_metadata"])["seed"], 123)

    def test_generation_metadata_excludes_image_payloads(self) -> None:
        from generation_metadata import build_generation_metadata
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="a neon chair",
            negative_prompt="blurry",
            diffusion_engine="native_int8_convrot",
            checkpoint="turbo",
            quantization="int8",
            steps=8,
            cfg=0.0,
            seed=42,
            sampler="euler_flow",
            mode="img2img",
            init_image_b64="C" * 128,
            mask_b64="D" * 128,
            inpaint_method="lanpaint_experimental",
            lanpaint_inner_steps=5,
            lanpaint_strength=0.8,
            moodboard_images=["A" * 128],
            ref_image1_b64="B" * 128,
            loras=[{"name": "krea2_darkbrush", "filename": "krea2_darkbrush.safetensors", "strength": 0.7}],
            seed_variance_algorithm="rbg",
            seed_variance_preset="creative",
            seed_variance_model_type="krea2",
            seed_variance_direction="visceral_expression_grit",
            seed_variance_shift_strength=170,
            seed_variance_schedule="step_cutoff",
            seed_variance_cutoff_step=3,
            seed_variance_total_steps=13,
            seed_variance_cutoff_strength=0.53,
        )

        metadata = build_generation_metadata(
            req,
            base_seed=42,
            image_index=1,
            filename="out.png",
            resolved_provider="krea_native",
            runtime={"provider": "native_int8_convrot", "torch_int_mm": True},
            model_runtime={"loaded_checkpoint_path": "models/krea2/diffusion_models/krea2_turbo_int8_convrot.safetensors"},
        )

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["prompt"], "a neon chair")
        self.assertEqual(metadata["seed"], 43)
        self.assertEqual(metadata["diffusion_engine"], "native_int8_convrot")
        self.assertEqual(metadata["engine"]["id"], "native_int8_convrot")
        self.assertEqual(metadata["engine"]["resolved_provider"], "krea_native")
        self.assertEqual(metadata["model"]["quantization"], "int8")
        self.assertIn("krea2_turbo_int8_convrot", metadata["model"]["loaded_checkpoint_path"])
        self.assertEqual(metadata["source"]["kind"], "image_to_image")
        self.assertEqual(metadata["source"]["init_image_count"], 1)
        self.assertTrue(metadata["source"]["init_image_hash"])
        self.assertTrue(metadata["source"]["mask_hash"])
        self.assertTrue(metadata["runtime"]["torch_int_mm"])
        self.assertEqual(metadata["checkpoint"], "turbo")
        self.assertEqual(metadata["quantization"], "int8")
        self.assertEqual(metadata["sampler"], "euler_flow")
        self.assertEqual(metadata["inpaint"]["method"], "lanpaint_experimental")
        self.assertEqual(metadata["inpaint"]["lanpaint_inner_steps"], 5)
        self.assertEqual(metadata["inpaint"]["lanpaint_strength"], 0.8)
        self.assertEqual(metadata["image_references"]["moodboard_count"], 1)
        self.assertEqual(metadata["seed_variance"]["algorithm"], "rbg")
        self.assertEqual(metadata["seed_variance"]["direction"], "visceral_expression_grit")
        self.assertEqual(metadata["seed_variance"]["shift_strength"], 170)
        self.assertEqual(metadata["seed_variance"]["schedule"], "step_cutoff")
        self.assertEqual(metadata["seed_variance"]["cutoff_step"], 3)
        self.assertEqual(metadata["seed_variance"]["total_steps"], 13)
        self.assertAlmostEqual(metadata["seed_variance"]["cutoff_strength"], 0.53)
        self.assertNotIn("A" * 128, json.dumps(metadata))
        self.assertNotIn("C" * 128, json.dumps(metadata))
        self.assertNotIn("D" * 128, json.dumps(metadata))

    def test_generation_request_defaults_keep_experimental_inpaint_off(self) -> None:
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="a quiet forest")

        self.assertEqual(req.sampler, "euler_flow")
        self.assertEqual(req.inpaint_method, "native")
        self.assertEqual(req.lanpaint_inner_steps, 3)
        self.assertEqual(req.lanpaint_strength, 1.0)
        self.assertEqual(req.creativity, "medium")
        self.assertEqual(req.moodboard_strength, 0.35)
        self.assertEqual(req.quantization, "fp8")
        self.assertEqual(req.batch_mode, "safe_queue")
        self.assertFalse(req.parallel_batch_confirmed)

    def test_generation_metadata_records_batch_context(self) -> None:
        from generation_metadata import build_generation_metadata
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="a quiet forest",
            num_images=4,
            batch_mode="parallel",
            parallel_batch_confirmed=True,
        )

        metadata = build_generation_metadata(req, base_seed=100, image_index=2)

        self.assertEqual(metadata["batch"]["mode"], "parallel")
        self.assertEqual(metadata["batch"]["index"], 2)
        self.assertEqual(metadata["batch"]["count"], 4)
        self.assertTrue(metadata["batch"]["parallel"])

    def test_raw_checkpoint_defaults_are_normalized_for_direct_api_requests(self) -> None:
        from inference import normalize_generation_defaults
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="a quiet forest", checkpoint="raw")

        normalize_generation_defaults(req)

        self.assertEqual(req.steps, 52)
        self.assertEqual(req.cfg, 3.5)
        self.assertIsNone(req.mu)
        self.assertEqual(req.quantization, "bf16")

    def test_raw_defaults_do_not_override_explicit_user_values(self) -> None:
        from inference import normalize_generation_defaults
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="a quiet forest", checkpoint="raw", steps=24, cfg=2.0, mu=0.9)

        normalize_generation_defaults(req)

        self.assertEqual(req.steps, 24)
        self.assertEqual(req.cfg, 2.0)
        self.assertEqual(req.mu, 0.9)

    def test_turbo_defaults_are_normalized_for_direct_api_requests(self) -> None:
        from inference import normalize_generation_defaults
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="a quiet forest", checkpoint="turbo")

        normalize_generation_defaults(req)

        self.assertEqual(req.steps, 8)
        self.assertEqual(req.cfg, 0.0)
        self.assertEqual(req.mu, 1.15)
        self.assertEqual(req.quantization, "fp8")

    def test_creativity_high_raises_unset_style_influence(self) -> None:
        from inference import normalize_generation_defaults
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="a quiet forest", creativity="high")

        normalize_generation_defaults(req)

        self.assertEqual(req.creativity, "high")
        self.assertEqual(req.moodboard_strength, 0.5)
        self.assertEqual(req.rebalance_multiplier, 1.15)

    def test_creativity_preserves_explicit_user_values(self) -> None:
        from inference import normalize_generation_defaults
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="a quiet forest",
            creativity="low",
            moodboard_strength=0.9,
            rebalance_multiplier=7.0,
        )

        normalize_generation_defaults(req)

        self.assertEqual(req.moodboard_strength, 0.9)
        self.assertEqual(req.rebalance_multiplier, 7.0)

    def test_lanpaint_method_rejects_non_inpaint_modes(self) -> None:
        from inference import _resolve_native_sampler

        req = SimpleNamespace(
            mode="txt2img",
            sampler="euler_flow",
            inpaint_method="lanpaint_experimental",
            init_image_b64="",
            mask_b64="",
        )

        with self.assertRaisesRegex(ValueError, "only available for inpaint"):
            _resolve_native_sampler(req)

    def test_official_lora_download_uses_loras_subfolder(self) -> None:
        import lora_manager

        kwargs = lora_manager.official_lora_download_kwargs("krea2_darkbrush")

        self.assertEqual(kwargs["repo_id"], "Comfy-Org/Krea-2")
        self.assertEqual(kwargs["filename"], "krea2_darkbrush.safetensors")
        self.assertEqual(kwargs["subfolder"], "loras")

    def test_manual_only_filter_bypass_lora_is_visible_when_missing(self) -> None:
        import lora_manager

        with tempfile.TemporaryDirectory() as td, patch.object(lora_manager, "LORAS_DIR", Path(td)):
            items = lora_manager.list_loras()

        item = next(lora for lora in items if lora["name"] == "krea2filterbypass3")
        self.assertFalse(item["installed"])
        self.assertFalse(item["download_enabled"])
        self.assertIn("manual", item["match_info"].lower())

    def test_flux_fill_call_uses_documented_defaults(self) -> None:
        import flux_fill_provider

        kwargs = flux_fill_provider.flux_fill_call_kwargs(
            prompt="add a lantern",
            image=Image.new("RGB", (512, 512), "white"),
            mask=Image.new("L", (512, 512), 255),
            width=512,
            height=512,
            seed=42,
            steps=8,
        )

        self.assertEqual(kwargs["guidance_scale"], 30.0)
        self.assertEqual(kwargs["num_inference_steps"], 50)
        self.assertEqual(kwargs["max_sequence_length"], 512)
        self.assertEqual(kwargs["height"], 512)
        self.assertEqual(kwargs["width"], 512)

    def test_quality_asset_download_specs_use_official_paths(self) -> None:
        import quality_assets

        specs = {spec.id: spec for spec in quality_assets.asset_specs()}

        self.assertEqual(specs["krea2_turbo_bf16"].repo_id, "Comfy-Org/Krea-2")
        self.assertEqual(specs["krea2_turbo_bf16"].filename, "diffusion_models/krea2_turbo_bf16.safetensors")
        self.assertEqual(specs["krea2_raw_bf16"].filename, "diffusion_models/krea2_raw_bf16.safetensors")
        self.assertEqual(specs["wan_2_1_vae"].filename, "split_files/vae/wan_2.1_vae.safetensors")
        self.assertEqual(specs["qwen3vl_abliterated_fp8"].repo_id, "ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy")
        self.assertEqual(specs["gguf_krea2_turbo_q4km"].filename, "Krea-2-Turbo-Q4_K_M.gguf")
        self.assertEqual(specs["gguf_qwen3vl_4b_q4km"].filename, "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
        self.assertEqual(specs["pid_qwenimage_decoder"].filename, "diffusion_models/pid_qwenimage_1024_to_4096_4step_bf16.safetensors")
        self.assertFalse(specs["krea2_filter_bypass"].download_enabled)
        self.assertEqual(specs["flux_fill"].repo_id, "black-forest-labs/FLUX.1-Fill-dev")

    def test_pid_upscale_dispatches_to_provider(self) -> None:
        from fastapi.testclient import TestClient
        from PIL import Image
        import base64
        import io
        import main

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "black").save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode()

        with (
            patch("main.pipeline.is_loaded", return_value=True),
            patch("pid_decoder_provider.upscale_pid", return_value=Image.new("RGB", (16, 16), "white")) as provider,
            TestClient(main.app) as client,
        ):
            response = client.post("/api/upscale", json={"image_b64": payload, "method": "pid_upscale", "prompt": "test"})

        self.assertEqual(response.status_code, 200)
        provider.assert_called_once()
        self.assertIn("image_b64", response.json())

    def test_flux_asset_status_guides_token_setup(self) -> None:
        import quality_assets

        spec = quality_assets.asset_by_id("flux_fill")
        with patch.object(quality_assets, "asset_installed", return_value=False):
            status = quality_assets.asset_status(spec, has_hf_token=False)
            self.assertTrue(status["gated"])
            self.assertTrue(status["needs_token"])
            self.assertEqual(status["setup_url"], "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev")

            with_token = quality_assets.asset_status(spec, has_hf_token=True)
            self.assertFalse(with_token["needs_token"])

    def test_quality_asset_status_reports_disabled_downloads(self) -> None:
        import quality_assets

        spec = quality_assets.asset_by_id("krea2_filter_bypass")
        status = quality_assets.asset_status(spec, has_hf_token=True)

        self.assertFalse(status["download_enabled"])
        self.assertIn("safety", status["disabled_reason"].lower())

    def test_xperiment_setup_returns_measured_fast_defaults(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import quality_assets

        def fake_installed(spec):
            return True

        with (
            patch.object(quality_assets, "asset_installed", side_effect=fake_installed),
            patch.object(main.settings, "diffusion_engine", "native_int8_convrot"),
            TestClient(main.app) as client,
        ):
            response = client.post("/api/xperiment/setup")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sampler"], {"sampler": "er_sde", "scheduler": "beta57", "steps": 6, "cfg": 0.0})
        self.assertEqual(data["diffusion_engine"], "native_int8_convrot")
        self.assertEqual(data["quantization"], "int8")
        self.assertEqual(data["lora"]["strength"], 0.55)
        self.assertEqual(data["lora"]["block_filter"], "late")
        self.assertEqual(data["loras"][0]["name"], "Krea2-realism-V1")
        self.assertEqual(data["loras"][0]["strength"], 0.55)
        self.assertEqual(data["loras"][0]["block_filter"], "late")
        self.assertEqual(data["loras"][1]["name"], "krea2filterbypass3")
        self.assertEqual(data["loras"][1]["strength"], 4.0)
        self.assertFalse(data["use_prompt_expander"])
        self.assertEqual(data["prompt_expander_backend"], "local")
        self.assertEqual(data["local_llm_backend"], "transformers")
        self.assertEqual(data["local_qwen_model_id"], "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated")
        self.assertIn("1024px", data["benchmark_note"])

    def test_gguf_low_vram_setup_skips_installed_assets_and_sets_paths(self) -> None:
        from fastapi.testclient import TestClient
        import gguf_runtime_installer
        import main
        import quality_assets

        downloaded: list[str] = []

        def fake_installed(spec):
            return spec.id in {"gguf_krea2_turbo_q4km", "wan_2_1_vae"}

        def fake_download(spec, token=None):
            downloaded.append(spec.id)
            return spec.local_path

        with (
            patch.object(quality_assets, "asset_installed", side_effect=fake_installed),
            patch.object(quality_assets, "download_asset", side_effect=fake_download),
            patch.object(gguf_runtime_installer, "install_stable_diffusion_cpp", return_value={"sd_cli_path": "E:\\Krea 2\\tools\\stable-diffusion.cpp\\sd-cli.exe", "skipped": "false"}),
            TestClient(main.app) as client,
        ):
            response = client.post("/api/gguf/setup-low-vram")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["diffusion_engine"], "gguf_external")
        self.assertEqual(data["sd_cli_path"], "E:\\Krea 2\\tools\\stable-diffusion.cpp\\sd-cli.exe")
        self.assertEqual(data["sampler"], {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15})
        self.assertEqual(data["realtime"]["preview_size"], 512)
        self.assertIn("gguf_krea2_turbo_q3km", downloaded)
        self.assertIn("gguf_qwen3vl_4b_q4km", downloaded)


if __name__ == "__main__":
    unittest.main()
