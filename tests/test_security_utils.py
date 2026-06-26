from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import security_utils  # noqa: E402


class SecurityUtilsTests(unittest.TestCase):
    def test_safe_output_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                security_utils.safe_child_file(root, "../secret.txt")
            with self.assertRaises(ValueError):
                security_utils.safe_child_file(root, "nested/secret.txt")

            self.assertEqual(security_utils.safe_child_file(root, "image-1.png").parent, root.resolve())

    def test_normalize_lora_url_rejects_untrusted_hosts(self) -> None:
        with self.assertRaises(ValueError):
            security_utils.normalize_lora_import_url("https://example.com/model.safetensors")
        with self.assertRaises(ValueError):
            security_utils.normalize_lora_import_url("http://huggingface.co/user/repo/blob/main/model.safetensors")

    def test_normalize_lora_url_rewrites_known_hosts(self) -> None:
        self.assertEqual(
            security_utils.normalize_lora_import_url("https://huggingface.co/u/r/blob/main/m.safetensors"),
            "https://huggingface.co/u/r/resolve/main/m.safetensors",
        )
        self.assertEqual(
            security_utils.normalize_lora_import_url("https://civitai.com/models/123?modelVersionId=456"),
            "https://civitai.com/api/download/models/456",
        )

    def test_safe_lora_filename(self) -> None:
        self.assertEqual(security_utils.safe_lora_filename("", "https://huggingface.co/u/r/resolve/main/foo"), "foo.safetensors")
        self.assertEqual(security_utils.safe_lora_filename("nice.safetensors", "https://x/y"), "nice.safetensors")
        with self.assertRaises(ValueError):
            security_utils.safe_lora_filename("../evil.safetensors", "https://x/y")


if __name__ == "__main__":
    unittest.main()
