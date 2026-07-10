from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import settings_env  # noqa: E402


class SettingsSecretTests(unittest.TestCase):
    def test_write_env_preserves_existing_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("CIVITAI_TOKEN=keep-me\nOUTPUT_DIR=old\n", encoding="utf-8")

            settings_env.write_env(env_path, {"OUTPUT_DIR": "new"})

            self.assertIn("CIVITAI_TOKEN=keep-me", env_path.read_text(encoding="utf-8"))
            self.assertIn("OUTPUT_DIR=new", env_path.read_text(encoding="utf-8"))

    def test_effective_secret_falls_back_to_os_environment(self) -> None:
        with (
            patch.dict(settings_env.os.environ, {"CIVITAI_TOKEN": "env-token"}, clear=False),
        ):
            self.assertEqual(settings_env.secret_value("CIVITAI_TOKEN", ""), "env-token")


if __name__ == "__main__":
    unittest.main()
