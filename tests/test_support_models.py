from __future__ import annotations

import tempfile
import unittest
import copy
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import support_models  # noqa: E402


class SupportModelTests(unittest.TestCase):
    def test_status_reports_missing_and_installed_support_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_ai = Path(tmp) / "local_ai"
            qwen_dir = local_ai / "qwen3_vl_4b_instruct"
            qwen_dir.mkdir(parents=True)
            (qwen_dir / "config.json").write_text("{}", encoding="utf-8")
            (qwen_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

            models = copy.deepcopy(support_models.SUPPORT_MODELS)
            for model in models:
                if model["id"] == "qwen3_vl":
                    model["local_dir"] = local_ai / "qwen3_vl_4b_instruct"
                if model["id"] == "qwen_image_vae":
                    model["local_dir"] = local_ai / "qwen_image"
            with patch.object(support_models, "LOCAL_AI_DIR", local_ai), patch.object(support_models, "SUPPORT_MODELS", models):
                statuses = support_models.support_model_status()

        by_id = {item["id"]: item for item in statuses}
        self.assertTrue(by_id["qwen3_vl"]["installed"])
        self.assertFalse(by_id["qwen_image_vae"]["installed"])
        self.assertIn("models", by_id["qwen3_vl"]["cache_dir"])


if __name__ == "__main__":
    unittest.main()
