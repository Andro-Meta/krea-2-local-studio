from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class CrashReporterTests(unittest.TestCase):
    def test_generation_breadcrumb_redacts_large_payloads(self) -> None:
        from crash_reporter import clear_generation_breadcrumb, stale_generation_breadcrumbs, write_generation_breadcrumb

        with tempfile.TemporaryDirectory() as tmp:
            req = SimpleNamespace(
                prompt="a fox",
                init_image_b64="A" * 2048,
                diffusion_engine="native_int8_convrot",
                quantization="int8",
            )

            path = write_generation_breadcrumb(tmp, job_id="job123", req=req, stage="running")
            data = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(data["job_id"], "job123")
            self.assertEqual(data["stage"], "running")
            self.assertEqual(data["request"]["prompt"], "a fox")
            self.assertTrue(data["request"]["init_image_b64"]["redacted"])
            self.assertEqual(data["request"]["init_image_b64"]["length"], 2048)
            self.assertEqual(len(stale_generation_breadcrumbs(tmp)), 1)

            clear_generation_breadcrumb(tmp, job_id="job123")
            self.assertEqual(stale_generation_breadcrumbs(tmp), [])

    def test_fault_logging_path_is_created(self) -> None:
        from crash_reporter import disable_fault_logging, enable_fault_logging

        with tempfile.TemporaryDirectory() as tmp:
            path = enable_fault_logging(tmp)

            self.assertEqual(path.name, "python-faulthandler.log")
            self.assertTrue(path.parent.exists())
            disable_fault_logging()


if __name__ == "__main__":
    unittest.main()
