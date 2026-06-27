from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2.performance_guard import attention_acceleration_diagnostic  # noqa: E402


class PerformanceGuardTests(unittest.TestCase):
    def test_missing_optional_package_reports_unavailable(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            result = attention_acceleration_diagnostic()

        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["available"])

    def test_fp8_text_fusion_path_reports_safe_disabled(self) -> None:
        with patch("importlib.util.find_spec", return_value=object()):
            result = attention_acceleration_diagnostic(dtype="fp8", text_fusion=True)

        self.assertEqual(result["status"], "safe_disabled")
        self.assertIn("fp8", result["reason"])

    def test_safe_mocked_package_reports_available_but_off(self) -> None:
        with patch("importlib.util.find_spec", return_value=object()), patch("platform.system", return_value="Linux"):
            result = attention_acceleration_diagnostic(device="cuda", dtype="bf16", text_fusion=False)

        self.assertEqual(result["status"], "available_but_off")
        self.assertTrue(result["available"])


if __name__ == "__main__":
    unittest.main()
