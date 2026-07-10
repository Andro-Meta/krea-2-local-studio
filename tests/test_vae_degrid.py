from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class VaeDegridTests(unittest.TestCase):
    def _txt2img(self, **kwargs):
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="test",
            mode="txt2img",
            checkpoint="turbo",
            quantization="fp8",
            width=1024,
            height=1024,
            steps=8,
            cfg=1.0,
            num_images=1,
            seed=1,
            **kwargs,
        )
        return build_graph(req)[0]

    def test_default_includes_vae_degrid(self) -> None:
        graph = self._txt2img()
        degrids = [n for n in graph.values() if n.get("class_type") == "VAEDeGrid"]
        self.assertEqual(len(degrids), 1)
        self.assertTrue(degrids[0]["inputs"]["enabled"])
        self.assertEqual(degrids[0]["inputs"]["mode"], "auto")

    def test_toggle_off_skips_vae_degrid(self) -> None:
        graph = self._txt2img(vae_degrid=False)
        degrids = [n for n in graph.values() if n.get("class_type") == "VAEDeGrid"]
        self.assertEqual(degrids, [])

    def test_schema_defaults_on(self) -> None:
        from schemas import GenerationRequest

        self.assertTrue(GenerationRequest(prompt="x").vae_degrid)


if __name__ == "__main__":
    unittest.main()
