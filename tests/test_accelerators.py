from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class AcceleratorDiagnosticsTests(unittest.TestCase):
    def test_status_reports_sdpa_default_and_missing_packages(self) -> None:
        from krea2.performance_guard import accelerator_status

        with patch("importlib.util.find_spec", return_value=None):
            status = accelerator_status()

        self.assertTrue(status["sdpa"]["available"])
        self.assertTrue(status["sdpa"]["default"])
        self.assertFalse(status["triton_windows"]["installed"])
        self.assertFalse(status["sageattention"]["installed"])
        self.assertEqual(status["xformers"]["recommendation"], "not recommended for Krea path yet")

    def test_status_detects_installed_sageattention(self) -> None:
        from krea2.performance_guard import accelerator_status

        with patch("importlib.util.find_spec", side_effect=lambda name: object() if name == "sageattention" else None):
            status = accelerator_status()

        self.assertTrue(status["sageattention"]["installed"])
        self.assertEqual(status["sageattention"]["recommendation"], "experimental")


if __name__ == "__main__":
    unittest.main()
