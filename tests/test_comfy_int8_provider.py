from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class NativeInt8ApiTests(unittest.TestCase):
    def test_native_int8_status_reports_assets_not_comfy(self) -> None:
        from fastapi.testclient import TestClient
        import main

        with TestClient(main.app) as client:
            response = client.get("/api/int8/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("torch_int_mm", data)
        self.assertIn("assets", data)
        self.assertIn("turbo", data["assets"])
        self.assertIn("raw", data["assets"])
        self.assertNotIn("base_url", data)

    def test_comfy_int8_workflow_route_is_removed(self) -> None:
        from fastapi.testclient import TestClient
        import main

        with TestClient(main.app) as client:
            response = client.post("/api/int8/test-workflow")

        self.assertIn(response.status_code, {404, 405})

    def test_native_int8_setup_returns_gui_defaults(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import quality_assets

        def fake_download(spec, token=None):
            return spec.local_path

        with (
            patch.object(quality_assets, "asset_installed", return_value=True),
            patch.object(quality_assets, "download_asset", side_effect=fake_download),
            TestClient(main.app) as client,
        ):
            response = client.post("/api/int8/setup-native")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["diffusion_engine"], "native_int8_convrot")
        self.assertEqual(data["quantization"], "int8")
        self.assertEqual(data["sampler"], {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15})


if __name__ == "__main__":
    unittest.main()
