from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class Int8LoaderRoutingTests(unittest.TestCase):
    def test_turbo_int8_uses_otu_w8a8_loader(self) -> None:
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="test",
            mode="txt2img",
            checkpoint="turbo",
            quantization="int8",
            diffusion_engine="native_int8_convrot",
            turbo_int8_variant="ax1y2jp",
            width=1024,
            height=1024,
            steps=8,
            cfg=1.0,
            num_images=1,
            seed=1,
        )
        with patch("comfy_workflows._b64_to_loadimage", side_effect=AssertionError("no image upload")):
            graph, _ = build_graph(req)
        loaders = [n for n in graph.values() if n.get("class_type") in ("UNETLoader", "OTUNetLoaderW8A8", "UnetLoaderGGUF")]
        self.assertEqual(len(loaders), 1)
        self.assertEqual(loaders[0]["class_type"], "OTUNetLoaderW8A8")
        self.assertEqual(loaders[0]["inputs"]["unet_name"], "ax1y2jp-krea2-turbo-int8-convrot.safetensors")
        self.assertFalse(loaders[0]["inputs"]["on_the_fly_quantization"])
        self.assertEqual(loaders[0]["inputs"]["model_type"], "krea2")

    def test_lilcheaty_and_all_int8_variants_route_to_otu(self) -> None:
        from comfy_workflows import _TURBO_INT8_VARIANTS, build_graph
        from schemas import GenerationRequest

        for variant, filename in _TURBO_INT8_VARIANTS.items():
            req = GenerationRequest(
                prompt="test",
                mode="txt2img",
                checkpoint="turbo",
                quantization="int8",
                diffusion_engine="native_int8_convrot",
                turbo_int8_variant=variant,
                width=1024,
                height=1024,
                steps=8,
                cfg=1.0,
                num_images=1,
                seed=1,
            )
            graph, _ = build_graph(req)
            loaders = [n for n in graph.values() if n.get("class_type") == "OTUNetLoaderW8A8"]
            self.assertEqual(len(loaders), 1, variant)
            self.assertEqual(loaders[0]["inputs"]["unet_name"], filename, variant)

    def test_fp8_still_uses_stock_unet_loader(self) -> None:
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="test",
            mode="txt2img",
            checkpoint="turbo",
            quantization="fp8",
            diffusion_engine="native_pytorch",
            width=1024,
            height=1024,
            steps=8,
            cfg=1.0,
            num_images=1,
            seed=1,
        )
        graph, _ = build_graph(req)
        loaders = [n for n in graph.values() if n.get("class_type") in ("UNETLoader", "OTUNetLoaderW8A8")]
        self.assertEqual(len(loaders), 1)
        self.assertEqual(loaders[0]["class_type"], "UNETLoader")


if __name__ == "__main__":
    unittest.main()
