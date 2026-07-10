from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import requests

import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import civitai_loras  # noqa: E402


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self._error = error
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self._error:
            raise self._error

    def iter_content(self, chunk_size: int):
        return iter(())


def _version_payload(version_id: int = 3091374) -> dict:
    return {
        "id": version_id,
        "name": "v1",
        "baseModel": "Krea 2",
        "trainedWords": ["lenovo"],
        "model": {"id": 123, "name": "Lenovo Krea2", "type": "LORA", "nsfw": False},
        "files": [
            {
                "primary": True,
                "name": "lenovo_krea2.safetensors",
                "downloadUrl": f"https://civitai.com/api/download/models/{version_id}",
            }
        ],
    }


class CivitaiInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.loras_dir = root / "loras"
        self.cache_path = root / "civitai_lora_cache.json"
        self.loras_dir.mkdir()
        self.patches = [
            patch.object(civitai_loras, "LORAS_DIR", self.loras_dir),
            patch.object(civitai_loras, "CACHE_PATH", self.cache_path),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self.tmpdir.cleanup()

    def test_download_401_reports_token_problem(self) -> None:
        http_error = requests.HTTPError(
            "401 Client Error: Unauthorized for url: https://civitai.com/api/download/models/3091374"
        )
        version_response = _Response(payload=_version_payload())
        download_response = _Response(status_code=401, error=http_error)

        with patch.object(civitai_loras.requests, "get", side_effect=[version_response, download_response]):
            with self.assertRaisesRegex(PermissionError, "Civitai API token"):
                civitai_loras.civitai_install(3091374)

    def test_install_returns_existing_filename_without_redownloading(self) -> None:
        existing = self.loras_dir / "lenovo_krea2.safetensors"
        existing.write_bytes(b"already here")

        with patch.object(civitai_loras.requests, "get", return_value=_Response(payload=_version_payload())) as get:
            result = civitai_loras.civitai_install(3091374)

        self.assertTrue(result["already_installed"])
        self.assertEqual(result["filename"], existing.name)
        self.assertEqual(get.call_count, 1)

        loras = [{"filename": existing.name, "name": existing.stem, "display_name": "", "trigger_words": [], "installed": True}]
        enriched = civitai_loras.enrich_loras(loras)
        self.assertEqual(enriched[0]["civitai_url"], "https://civitai.com/models/123")

    def test_browse_marks_cached_version_as_installed(self) -> None:
        existing = self.loras_dir / "other_name.safetensors"
        existing.write_bytes(b"already here")
        sha = civitai_loras._sha256_file(existing)
        civitai_loras._save_cache(
            {
                "hashes": {civitai_loras._pathkey(existing): sha},
                "meta": {sha: civitai_loras._normalize_version(_version_payload())},
            }
        )
        browse_payload = {"items": [{"id": 123, "name": "Lenovo Krea2", "type": "LORA", "modelVersions": [_version_payload()]}]}

        with patch.object(civitai_loras.requests, "get", return_value=_Response(payload=browse_payload)):
            result = civitai_loras.civitai_browse()

        self.assertTrue(result["items"][0]["installed"])
        self.assertEqual(result["items"][0]["installed_filename"], existing.name)


if __name__ == "__main__":
    unittest.main()
