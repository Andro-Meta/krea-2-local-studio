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

import huggingface_loras as hf  # noqa: E402


class _Response:
    def __init__(self, *, status_code: int = 200, payload=None, headers=None, error: Exception | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.headers = headers or {}
        self._error = error

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self._error:
            raise self._error
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class HuggingFaceLorasTests(unittest.TestCase):
    def test_parse_next_cursor(self):
        link = (
            '<https://huggingface.co/api/models?search=krea-2&filter=lora&limit=5'
            '&cursor=abc123>; rel="next"'
        )
        self.assertEqual(hf._parse_next_cursor(link), "abc123")
        self.assertIsNone(hf._parse_next_cursor(""))

    def test_is_krea_lora_by_tag(self):
        self.assertTrue(hf._is_krea_lora({
            "id": "someone/style",
            "tags": ["lora", "base_model:krea/Krea-2-Turbo"],
        }))
        self.assertFalse(hf._is_krea_lora({
            "id": "someone/sdxl-style",
            "tags": ["lora", "base_model:stabilityai/stable-diffusion-xl-base-1.0"],
        }))

    def test_browse_filters_and_paginates(self):
        payload = [
            {
                "id": "ostris/krea2_turbo_training_adapter",
                "downloads": 100,
                "likes": 2,
                "tags": ["lora", "base_model:krea/Krea-2-Turbo"],
            },
            {
                "id": "other/sdxl",
                "downloads": 999,
                "likes": 9,
                "tags": ["lora", "base_model:stabilityai/stable-diffusion-xl-base-1.0"],
            },
        ]
        link = '<https://huggingface.co/api/models?cursor=nextpage>; rel="next"'

        with patch.object(hf.requests, "get", return_value=_Response(payload=payload, headers={"Link": link})) as mock_get:
            result = hf.huggingface_browse(query="", sort="Most Downloaded", limit=48)

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["repo_id"], "ostris/krea2_turbo_training_adapter")
        self.assertTrue(result["has_more"])
        self.assertEqual(result["next_cursor"], "nextpage")
        self.assertIn("krea-2", mock_get.call_args.kwargs["params"]["search"])

    def test_install_multi_file_requires_choice(self):
        siblings = {
            "siblings": [
                {"rfilename": "a.safetensors", "size": 10},
                {"rfilename": "b.safetensors", "size": 20},
            ]
        }
        with patch.object(hf.requests, "get", return_value=_Response(payload=siblings)):
            with self.assertRaises(hf.MultiFileRequired) as ctx:
                hf.huggingface_install("owner/repo")
        self.assertEqual(len(ctx.exception.files), 2)

    def test_install_single_file(self):
        siblings = {"siblings": [{"rfilename": "cool.safetensors", "size": 10}]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loras = tmp_path / "loras"
            loras.mkdir()
            cached = tmp_path / "cache" / "cool.safetensors"
            cached.parent.mkdir()
            cached.write_bytes(b"lora")

            with patch.object(hf, "LORAS_DIR", loras), \
                    patch.object(hf.requests, "get", return_value=_Response(payload=siblings)), \
                    patch("huggingface_hub.hf_hub_download", return_value=str(cached)), \
                    patch("lora_manager.inspect_lora", return_value={"compatible": True, "reason": "ok"}):
                result = hf.huggingface_install("owner/repo")

            dest = loras / "owner__repo__cool.safetensors"
            self.assertTrue(dest.exists())
            self.assertEqual(result["filename"], dest.name)
            self.assertTrue(result["compatible"])
            self.assertFalse(result["already_installed"])


if __name__ == "__main__":
    unittest.main()
