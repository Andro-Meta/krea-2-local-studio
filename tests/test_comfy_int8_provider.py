from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ComfyInt8ProviderTests(unittest.TestCase):
    def test_builds_krea2_int8_workflow_with_lora_stack(self) -> None:
        from comfy_int8_provider import ComfyInt8Settings, build_comfy_int8_workflow

        req = SimpleNamespace(
            prompt="a red fox",
            negative_prompt="blur",
            mode="txt2img",
            width=720,
            height=1280,
            steps=8,
            cfg=1.0,
            sampler="ddim",
            scheduler="beta57",
            seed=123,
            loras=[
                {"name": "krea2_darkbrush", "filename": "krea2_darkbrush.safetensors", "strength": 0.8, "enabled": True},
                {"name": "off", "filename": "off.safetensors", "strength": 1.0, "enabled": False},
            ],
        )

        workflow = build_comfy_int8_workflow(
            req,
            ComfyInt8Settings(
                int8_model="krea2_turbo_int8.safetensors",
                clip_name="qwen3vl_4b_fp8_scaled.safetensors",
                vae_name="qwen_image_vae.safetensors",
            ),
        )

        self.assertEqual(workflow["1"]["class_type"], "UNETLoader")
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], "krea2_turbo_int8.safetensors")
        self.assertEqual(workflow["2"]["inputs"]["type"], "krea2")
        lora_nodes = [node for node in workflow.values() if node["class_type"] == "LoraLoader"]
        self.assertEqual(len(lora_nodes), 1)
        self.assertEqual(lora_nodes[0]["inputs"]["lora_name"], "krea2_darkbrush.safetensors")
        sampler = next(node for node in workflow.values() if node["class_type"] == "KSampler")
        self.assertEqual(sampler["inputs"]["sampler_name"], "ddim")
        self.assertEqual(sampler["inputs"]["scheduler"], "beta57")
        self.assertEqual(sampler["inputs"]["steps"], 8)
        self.assertEqual(sampler["inputs"]["cfg"], 1.0)

    def test_rejects_non_local_comfy_base_url(self) -> None:
        from comfy_int8_provider import ComfyInt8Settings, comfy_int8_status

        with self.assertRaisesRegex(ValueError, "localhost"):
            comfy_int8_status(ComfyInt8Settings(base_url="https://example.com"))

    def test_api_workflow_dry_run_returns_nodes(self) -> None:
        from fastapi.testclient import TestClient
        import main

        with TestClient(main.app) as client:
            response = client.post("/api/int8/test-workflow")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertGreaterEqual(data["node_count"], 8)
        self.assertIn("workflow", data)


if __name__ == "__main__":
    unittest.main()
