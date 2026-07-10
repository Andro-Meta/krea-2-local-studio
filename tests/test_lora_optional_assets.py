from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class OptionalLoraAssetTests(unittest.TestCase):
    def test_k2q_and_nk2e_download_kwargs_use_repo_paths(self) -> None:
        from lora_manager import official_lora_download_kwargs

        k2q = official_lora_download_kwargs("k2q_turbo_lora_rank64")
        nk2e = official_lora_download_kwargs("nk2e_v01")

        self.assertEqual(k2q["repo_id"], "silveroxides/K2Q")
        self.assertEqual(k2q["filename"], "krea2_turbo_lora_rank_64_final_nodiff.safetensors")
        self.assertNotIn("subfolder", k2q)
        self.assertEqual(nk2e["repo_id"], "nynxz/NK2E")
        self.assertEqual(nk2e["filename"], "comfy/v0.1/NK2E-v0.1.safetensors")

    def test_missing_optional_lora_reports_inspection_gate(self) -> None:
        import lora_manager

        with patch.object(lora_manager, "LORAS_DIR", Path("missing-loras")):
            items = {item["name"]: item for item in lora_manager.list_loras()}

        self.assertIn("k2q_turbo_lora_rank64", items)
        self.assertIn("nk2e_v01", items)
        self.assertFalse(items["k2q_turbo_lora_rank64"]["installed"])
        self.assertIn("inspected after download", items["k2q_turbo_lora_rank64"]["match_info"])

if __name__ == "__main__":
    unittest.main()
