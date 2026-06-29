from __future__ import annotations

import sys
import tempfile
import unittest
import io
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import share_startup  # noqa: E402


class ShareStartupTests(unittest.TestCase):
    def test_env_bool_accepts_common_true_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("KREA_SHARE_AUTO_FUNNEL=yes\nOTHER=false\n", encoding="utf-8")

            env = share_startup.read_env_file(env_path)

        self.assertTrue(share_startup.env_bool(env, "KREA_SHARE_AUTO_FUNNEL"))
        self.assertFalse(share_startup.env_bool(env, "OTHER"))
        self.assertFalse(share_startup.env_bool(env, "MISSING"))

    def test_wait_for_url_retries_until_ready(self) -> None:
        attempts = {"count": 0}

        def fake_urlopen(*_args, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise OSError("not ready")

            class Response:
                def close(self) -> None:
                    pass

            return Response()

        with (
            patch("share_startup.urllib.request.urlopen", side_effect=fake_urlopen),
            patch("share_startup.time.sleep", return_value=None),
        ):
            self.assertTrue(share_startup.wait_for_url("http://127.0.0.1:8200/krea/api/auth/me", timeout_seconds=5))

        self.assertEqual(attempts["count"], 3)

    def test_wait_for_url_treats_login_gate_as_ready(self) -> None:
        import urllib.error

        def fake_urlopen(url: str, **_kwargs):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", hdrs=None, fp=io.BytesIO())

        with patch("share_startup.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertTrue(
                share_startup.wait_for_url("http://127.0.0.1:8200/krea/api/auth/me", timeout_seconds=1)
            )

    def test_auto_funnel_starts_tailscale_and_funnel(self) -> None:
        calls: list[str] = []

        with (
            patch("share_startup.sharing_service.tailscale_up", side_effect=lambda: calls.append("up") or {"ok": True}),
            patch("share_startup.sharing_service.start_funnel", side_effect=lambda: calls.append("funnel") or {"ok": True, "url": "https://machine.ts.net/krea/"}),
        ):
            url = share_startup.maybe_start_funnel(auto_funnel=True)

        self.assertEqual(calls, ["up", "funnel"])
        self.assertEqual(url, "https://machine.ts.net/krea/")


if __name__ == "__main__":
    unittest.main()
