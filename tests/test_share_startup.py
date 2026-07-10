from __future__ import annotations

import sys
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
            patch("share_startup.sharing_service.funnel_status", return_value={"running": False, "message": "", "url": ""}),
            patch("share_startup.sharing_service.tailscale_up", side_effect=lambda: calls.append("up") or {"ok": True}),
            patch("share_startup.sharing_service.current_server_port", return_value=21079),
            patch(
                "share_startup.sharing_service.start_funnel",
                side_effect=lambda port=None: calls.append(f"funnel:{port}") or {"ok": True, "url": "https://machine.ts.net/krea/"},
            ),
            patch(
                "share_startup.sharing_service.public_funnel_probe_with_retries",
                side_effect=lambda url, **_: calls.append(f"probe:{url}") or {"ok": True, "url": url},
            ),
        ):
            url = share_startup.maybe_start_funnel(auto_funnel=True)

        self.assertEqual(calls, ["up", "funnel:21079", "probe:https://machine.ts.net/krea/"])
        self.assertEqual(url, "https://machine.ts.net/krea/")

    def test_existing_funnel_rebinds_even_without_auto_flag(self) -> None:
        calls: list[str] = []

        with (
            patch(
                "share_startup.sharing_service.funnel_status",
                return_value={"running": True, "message": "/krea proxy http://127.0.0.1:1111", "url": "https://machine.ts.net/krea/"},
            ),
            patch("share_startup.sharing_service.tailscale_up", side_effect=lambda: calls.append("up") or {"ok": True}),
            patch("share_startup.sharing_service.current_server_port", return_value=2222),
            patch(
                "share_startup.sharing_service.start_funnel",
                side_effect=lambda port=None: calls.append(f"funnel:{port}") or {"ok": True, "url": "https://machine.ts.net/krea/"},
            ),
            patch("share_startup.sharing_service.public_funnel_probe_with_retries", return_value={"ok": True}),
        ):
            url = share_startup.maybe_start_funnel(auto_funnel=False)

        self.assertEqual(calls, ["up", "funnel:2222"])
        self.assertEqual(url, "https://machine.ts.net/krea/")

    def test_stale_ingress_triggers_connection_cycle_then_recovers(self) -> None:
        calls: list[str] = []
        probes = {"count": 0}

        def fake_probe(url, **_):
            probes["count"] += 1
            calls.append(f"probe{probes['count']}")
            return {"ok": probes["count"] > 1, "message": "Public Funnel TLS/proxy check failed before reaching Krea."}

        with (
            patch("share_startup.sharing_service.funnel_status", return_value={"running": False, "message": "", "url": ""}),
            patch("share_startup.sharing_service.tailscale_up", return_value={"ok": True}),
            patch("share_startup.sharing_service.current_server_port", return_value=21079),
            patch(
                "share_startup.sharing_service.start_funnel",
                return_value={"ok": True, "url": "https://machine.ts.net/krea/"},
            ),
            patch("share_startup.sharing_service.public_funnel_probe_with_retries", side_effect=fake_probe),
            patch(
                "share_startup.sharing_service.cycle_tailscale_connection",
                side_effect=lambda: calls.append("cycle") or {"ok": True, "message": ""},
            ),
        ):
            url = share_startup.maybe_start_funnel(auto_funnel=True)

        self.assertEqual(calls, ["probe1", "cycle", "probe2"])
        self.assertEqual(url, "https://machine.ts.net/krea/")

    def test_probe_retries_tolerate_ingress_propagation_delay(self) -> None:
        import sharing_service

        attempts = {"count": 0}

        def flaky_probe(url):
            attempts["count"] += 1
            return {"ok": attempts["count"] >= 3, "message": "propagating"}

        with patch("sharing_service.public_funnel_probe", side_effect=flaky_probe):
            result = sharing_service.public_funnel_probe_with_retries("https://machine.ts.net/krea/", attempts=4, delay_seconds=0)

        self.assertTrue(result["ok"])
        self.assertEqual(attempts["count"], 3)


if __name__ == "__main__":
    unittest.main()
