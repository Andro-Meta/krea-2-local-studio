from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import sharing_service  # noqa: E402


class SharingServiceTests(unittest.TestCase):
    def test_start_funnel_uses_krea_path_and_local_port(self) -> None:
        calls = []

        class Result:
            returncode = 0
            stdout = "https://machine.tail.ts.net\n"
            stderr = ""

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        with (
            patch("sharing_service.find_tailscale", return_value="tailscale"),
            patch("sharing_service.subprocess.run", side_effect=fake_run),
        ):
            result = sharing_service.start_funnel()

        self.assertTrue(result["ok"])
        self.assertIn("/krea/", result["url"])
        self.assertIn("--set-path=/krea", calls[0])
        self.assertIn("127.0.0.1:8200", calls[0])

    def test_status_reports_missing_tailscale(self) -> None:
        with patch("sharing_service.find_tailscale", return_value=None):
            status = sharing_service.tailscale_status()

        self.assertFalse(status["installed"])
        self.assertFalse(status["connected"])


if __name__ == "__main__":
    unittest.main()
