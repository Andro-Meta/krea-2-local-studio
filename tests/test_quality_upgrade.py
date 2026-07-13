from __future__ import annotations

import sys
import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from support import mock_atomic_cancel_capability

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

    def test_quality_asset_download_specs_use_official_paths(self) -> None:
        import quality_assets

        specs = {spec.id: spec for spec in quality_assets.asset_specs()}

        self.assertEqual(specs["krea2_turbo_bf16"].repo_id, "Comfy-Org/Krea-2")
        self.assertEqual(specs["krea2_turbo_bf16"].filename, "diffusion_models/krea2_turbo_bf16.safetensors")
        self.assertEqual(specs["krea2_raw_bf16"].filename, "diffusion_models/krea2_raw_bf16.safetensors")
        self.assertEqual(specs["wan_2_1_vae"].filename, "split_files/vae/wan_2.1_vae.safetensors")
        self.assertEqual(specs["qwen_image_comfy_vae"].filename, "split_files/vae/qwen_image_vae.safetensors")
        self.assertEqual(specs["spacepxl_wan_2x_vae"].repo_id, "spacepxl/Wan2.1-VAE-upscale2x")
        self.assertEqual(specs["qwen3vl_abliterated_fp8"].repo_id, "ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy")
        self.assertEqual(specs["qwen3vl_krea2_fp8"].filename, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual(specs["k2q_turbo_lora_rank64"].repo_id, "silveroxides/K2Q")
        self.assertEqual(specs["k2q_turbo_lora_rank64"].filename, "krea2_turbo_lora_rank_64_final_nodiff.safetensors")
        self.assertEqual(specs["k2q_turbo_lora_rank128"].filename, "krea2_turbo_lora_rank_128_final_nodiff.safetensors")
        self.assertEqual(specs["nk2e_v01_lora"].filename, "comfy/v0.1/NK2E-v0.1.safetensors")
        self.assertEqual(specs["krea2_identity_edit_v1"].repo_id, "conradlocke/krea2-identity-edit")
        self.assertEqual(specs["krea2_identity_edit_v1"].filename, "krea2_identity_edit_v1.safetensors")
        self.assertTrue(str(specs["krea2_identity_edit_v1"].local_path).endswith("models\\loras\\krea2_identity_edit_v1.safetensors"))
        self.assertEqual(specs["k2q_filter_bypass_projectors"].allow_patterns, ["txtfusion.projector_singlelayer/*"])
        self.assertTrue(specs["k2q_filter_bypass_projectors"].download_enabled)
        self.assertEqual(specs["gguf_krea2_turbo_q4km"].filename, "Krea-2-Turbo-Q4_K_M.gguf")
        self.assertEqual(specs["gguf_qwen3vl_4b_q4km"].filename, "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
        self.assertTrue(specs["krea2_filter_bypass"].download_enabled)

    def test_install_bat_downloads_default_comfy_assets(self) -> None:
        install_bat = (ROOT / "install.bat").read_text(encoding="utf-8")

        for asset_id in [
            "krea2_turbo_int8_convrot",
            "qwen3vl_abliterated_fp8",
            "qwen_image_comfy_vae",
            "krea2_filter_bypass",
            "krea2_identity_edit_v1",
            "nk2e_v01_lora",
        ]:
            self.assertIn(asset_id, install_bat)

    def test_install_bat_prompts_for_api_tokens(self) -> None:
        install_bat = (ROOT / "install.bat").read_text(encoding="utf-8")

        self.assertIn("Hugging Face token", install_bat)
        self.assertIn("CivitAI API token", install_bat)
        self.assertIn("set_env_tokens.py", install_bat)
        # Tokens must be captured before the asset downloads they accelerate.
        self.assertLess(install_bat.index("set_env_tokens.py"), install_bat.index("download_quality_assets.py"))

    def test_install_bat_offers_optional_god_mode_assets(self) -> None:
        install_bat = (ROOT / "install.bat").read_text(encoding="utf-8")

        self.assertIn("download_godmode_assets.py", install_bat)

    def test_comfy_installer_requires_default_nodes(self) -> None:
        installer = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")

        for node in ["ComfyUI-Krea2TextEncoder", "ComfyUI-Conditioning-Rebalance", "ComfyUI-INT8-Fast"]:
            self.assertIn(node, installer)
        self.assertIn("Required default ComfyUI node failed to clone", installer)

    def test_comfy_installer_pins_required_krea_deforum_node(self) -> None:
        installer = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")
        helper = (ROOT / "scripts" / "kreadeforum_install.ps1").read_text(encoding="utf-8")

        self.assertIn("Dream-Making-Git/KreaDeforum", installer)
        self.assertIn("49bb6752ab045fac25652f3e9207d4706bf5c646", installer)
        for node_class in [
            "KreaDeforumAnimator",
            "KreaDeforumSaveVideo",
            "KreaDeforumSchedulePreview",
        ]:
            self.assertIn(node_class, installer)
        self.assertIn("kreadeforum_install.ps1", installer)
        self.assertIn("Install-KreaDeforumCheckout", installer)
        self.assertIn("status --porcelain", helper)
        self.assertIn("Test-Path $kreaDeforumRequirements", installer)

    def test_krea_deforum_requirements_are_hash_locked_without_torch(self) -> None:
        installer = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")
        source = (ROOT / "requirements" / "kreadeforum.in").read_text(encoding="utf-8")
        lock = (ROOT / "requirements" / "kreadeforum-windows-py312.txt").read_text(encoding="utf-8")
        direct_packages = [
            "opencv-python-headless",
            "numpy",
            "imageio",
            "imageio-ffmpeg",
            "matplotlib",
            "timm",
        ]

        self.assertIn("49bb6752ab045fac25652f3e9207d4706bf5c646", source)
        self.assertEqual(
            [line for line in source.splitlines() if line and not line.startswith("#")],
            [
                "opencv-python-headless>=4.8.0",
                "numpy>=1.24.0",
                "imageio>=2.31.0",
                "imageio-ffmpeg>=0.4.9",
                "matplotlib>=3.7.0",
                "timm>=0.9.0",
            ],
        )
        for package in direct_packages:
            self.assertRegex(
                lock,
                rf"(?m)^{package}==[^\s\\]+ \\\n\s+--hash=sha256:[0-9a-f]{{64}}",
            )
        self.assertNotRegex(lock, r"(?mi)^(torch|torchvision|torchaudio)==")
        self.assertIn(
            '$kreaDeforumLock = Join-Path $root "requirements\\kreadeforum-windows-py312.txt"',
            installer,
        )
        self.assertIn(
            "$venvPy -m pip install --require-hashes -r $kreaDeforumLock",
            installer,
        )
        self.assertIn("$kreaDeforumLock --no-deps", installer)

    def test_comfy_installer_includes_character_edit_node(self) -> None:
        installer = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")

        self.assertIn("lbouaraba/comfyui-krea2edit", installer)

    def test_comfy_installer_covers_turbo4x_and_god_mode_dependencies(self) -> None:
        installer = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")

        # kjnodes: LazySwitchKJ/PathchSageAttentionKJ in the Turbo 4X template.
        self.assertIn("kijai/ComfyUI-KJNodes", installer)
        # FaceDetailer bbox detector for God Mode stage 4.
        self.assertIn("face_yolov8m", installer)

    def test_godmode_download_script_targets_comfy_model_dirs(self) -> None:
        script = (ROOT / "scripts" / "download_godmode_assets.py").read_text(encoding="utf-8")

        self.assertIn("Comfy-Org/z_image_turbo", script)
        for filename in ["z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"]:
            self.assertIn(filename, script)

    def test_snapshot_asset_status_requires_payload_files(self) -> None:
        import quality_assets

        base_spec = quality_assets.asset_by_id("k2q_filter_bypass_projectors")
        with tempfile.TemporaryDirectory() as td:
            spec = replace(base_spec, local_path=Path(td))
            Path(td, ".gitattributes").write_text("", encoding="utf-8")
            self.assertFalse(quality_assets.asset_status(spec)["installed"])

            payload = Path(td, "txtfusion.projector_singlelayer", "krea2bypass_filtered_01.safetensors")
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"fake")

            self.assertTrue(quality_assets.asset_status(spec)["installed"])

    def test_xperiment_setup_returns_measured_fast_defaults(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import quality_assets

        mock_atomic_cancel_capability(main)
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
        self.assertEqual(data["sampler"], {"sampler": "er_sde", "scheduler": "beta57", "steps": 8, "cfg": 1.0})
        self.assertEqual(data["diffusion_engine"], "native_int8_convrot")
        self.assertEqual(data["quantization"], "int8")
        loras = {item["name"]: item for item in data["loras"]}
        self.assertEqual(loras["Krea2-realism-V1"]["strength"], 0.6)
        self.assertEqual(loras["Krea2-realism-V1"]["block_filter"], "late")
        self.assertIn(data["lora"]["name"], loras)
        if "krea2filterbypass3" in loras:
            self.assertEqual(loras["krea2filterbypass3"]["strength"], 4.0)
        self.assertFalse(data["use_prompt_expander"])
        self.assertEqual(data["prompt_expander_backend"], "local")
        self.assertEqual(data["local_llm_backend"], "transformers")
        self.assertEqual(data["local_qwen_model_id"], "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated")
        self.assertIn("8 steps", data["benchmark_note"])
        self.assertIn("CFG 1", data["benchmark_note"])

    def test_gguf_low_vram_setup_skips_installed_assets_and_sets_paths(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import quality_assets

        mock_atomic_cancel_capability(main)
        downloaded: list[str] = []

        def fake_installed(spec):
            return spec.id in {"gguf_krea2_turbo_q4km", "wan_2_1_vae"}

        def fake_download(spec, token=None):
            downloaded.append(spec.id)
            return spec.local_path

        with (
            patch.object(quality_assets, "asset_installed", side_effect=fake_installed),
            patch.object(quality_assets, "download_asset", side_effect=fake_download),
            patch.object(main, "_write_env", return_value=None),
            patch.object(main.settings, "krea2_auto_checkpoint", ""),
            patch.object(main.settings, "krea2_turbo_path", ""),
            TestClient(main.app) as client,
        ):
            response = client.post("/api/gguf/setup-low-vram")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["diffusion_engine"], "native_gguf")
        self.assertEqual(data["quantization"], "gguf")
        self.assertTrue(data["checkpoint_path"].endswith("Krea-2-Turbo-Q4_K_M.gguf"))
        self.assertTrue(data["vae_path"].endswith("wan_2.1_vae.safetensors"))
        self.assertEqual(data["sampler"], {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15})
        self.assertEqual(downloaded, [])


if __name__ == "__main__":
    unittest.main()
