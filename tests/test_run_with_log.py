from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RunWithLogTests(unittest.TestCase):
    def test_mirrors_combined_output_to_console_and_log(self) -> None:
        from scripts.run_with_log import run_with_log

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            out = io.StringIO()

            code = run_with_log(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('ready on stdout'); print('uvicorn info on stderr', file=sys.stderr)",
                ],
                log_path=log_path,
                stdout=out,
            )

            self.assertEqual(code, 0)
            self.assertIn("ready on stdout", out.getvalue())
            self.assertIn("uvicorn info on stderr", out.getvalue())
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("ready on stdout", text)
            self.assertIn("uvicorn info on stderr", text)


if __name__ == "__main__":
    unittest.main()
