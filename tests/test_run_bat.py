from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunBatTests(unittest.TestCase):
    def test_uvicorn_launch_uses_explicit_venv_python(self) -> None:
        text = (ROOT / "run.bat").read_text(encoding="utf-8")

        self.assertIn('set "KREA_PYTHON=venv\\Scripts\\python.exe"', text)
        self.assertIn('%KREA_PYTHON% scripts\\run_with_log.py', text)
        self.assertIn('-- %KREA_PYTHON% -u -m uvicorn backend.main:app', text)
        self.assertNotIn('-- python -u -m uvicorn backend.main:app', text)


if __name__ == "__main__":
    unittest.main()
