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
    def test_tailscale_connected_requires_running_backend_and_identity(self) -> None:
        class Result:
            returncode = 0
            stderr = ""

            def __init__(self, state: str, *, dns: str = "machine.tail.ts.net.", ips=None):
                import json

                self.stdout = json.dumps(
                    {
                        "BackendState": state,
                        "Self": {"DNSName": dns, "TailscaleIPs": ["100.64.0.1"] if ips is None else ips},
                    }
                )

        with (
            patch("sharing_service.find_tailscale", return_value="tailscale"),
            patch("sharing_service._run_tailscale", return_value=Result("Running")),
        ):
            self.assertTrue(sharing_service.tailscale_status()["connected"])

        for state in ("NeedsLogin", "Stopped", "NoState"):
            with (
                self.subTest(state=state),
                patch("sharing_service.find_tailscale", return_value="tailscale"),
                patch("sharing_service._run_tailscale", return_value=Result(state)),
            ):
                self.assertFalse(sharing_service.tailscale_status()["connected"])

        for dns, ips in (("", ["100.64.0.1"]), ("machine.tail.ts.net.", [])):
            with (
                self.subTest(dns=dns, ips=ips),
                patch("sharing_service.find_tailscale", return_value="tailscale"),
                patch("sharing_service._run_tailscale", return_value=Result("Running", dns=dns, ips=ips)),
            ):
                self.assertFalse(sharing_service.tailscale_status()["connected"])

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

    def test_start_funnel_preserves_existing_root_funnel(self) -> None:
        calls = []

        class Result:
            def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        results = [
            Result(1, stderr="foreground listener already exists for port 443"),
            Result(0, stdout="https://machine.tail.ts.net\n"),
            Result(0, stdout="https://machine.tail.ts.net\n"),
        ]

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == "taskkill":
                return Result(0)
            return results.pop(0)

        with (
            patch("sharing_service.find_tailscale", return_value="tailscale"),
            patch("sharing_service.foreground_funnel_ports", return_value=[(1234, 9000)]),
            patch("sharing_service.subprocess.run", side_effect=fake_run),
        ):
            result = sharing_service.start_funnel(port=45678)

        self.assertTrue(result["ok"])
        self.assertEqual(calls[0], ["tailscale", "funnel", "--set-path=/krea", "--bg", "--yes", "127.0.0.1:45678"])
        self.assertEqual(calls[1], ["taskkill", "/PID", "1234", "/F"])
        self.assertEqual(calls[2], ["tailscale", "funnel", "--set-path=/", "--bg", "--yes", "127.0.0.1:9000"])
        self.assertEqual(calls[3], ["tailscale", "funnel", "--set-path=/krea", "--bg", "--yes", "127.0.0.1:45678"])

    def test_local_krea_target_reports_auth_state(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"authenticated":false,"username":null,"role":null}'

        with patch("sharing_service.urllib.request.urlopen", return_value=Response()):
            status = sharing_service.local_krea_target_status(port=45678)

        self.assertTrue(status["ok"])
        self.assertTrue(status["auth_required"])
        self.assertIn("45678", status["url"])

    def test_local_krea_target_falls_back_to_funnel_proxy_port(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"authenticated":false,"username":null,"role":null}'

        seen_urls = []

        def fake_urlopen(url, timeout=5):
            seen_urls.append(url)
            if ":8200/" in url:
                raise OSError("stale default port")
            return Response()

        with (
            patch("sharing_service.funnel_status", return_value={
                "running": True,
                "message": "|-- /krea proxy http://127.0.0.1:57569",
            }),
            patch("sharing_service.urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            status = sharing_service.local_krea_target_status(port=8200)

        self.assertTrue(status["ok"])
        self.assertIn("57569", status["url"])
        self.assertTrue(any(":8200/" in url for url in seen_urls))
        self.assertTrue(any(":57569/" in url for url in seen_urls))

    def test_local_krea_target_does_not_expose_connection_exception(self) -> None:
        secret = "private-host-detail"
        with (
            patch("sharing_service.funnel_status", return_value={"message": ""}),
            patch(
                "sharing_service.urllib.request.urlopen",
                side_effect=OSError(secret),
            ),
            patch.object(sharing_service.logger, "warning") as warning,
        ):
            status = sharing_service.local_krea_target_status(port=45678)

        self.assertFalse(status["ok"])
        self.assertNotIn(secret, status["message"])
        warning.assert_called()

    def test_repair_funnel_runs_tailscale_up_and_reapplies_krea_path(self) -> None:
        calls = []

        class Result:
            def __init__(self, returncode=0, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def fake_run(args, timeout=30):
            calls.append(args)
            if args[:2] == ["status", "--json"]:
                return Result(stdout='{"BackendState":"Running","Self":{"DNSName":"diffusion.tail.ts.net.","TailscaleIPs":["100.64.0.1"]}}')
            if args[:2] == ["funnel", "status"]:
                return Result(stdout="https://diffusion.tail.ts.net/krea\n")
            if args[:1] == ["up"]:
                return Result()
            if args[:1] == ["funnel"]:
                return Result(stdout="https://diffusion.tail.ts.net\n")
            return Result()

        with (
            patch("sharing_service.find_tailscale", return_value="tailscale"),
            patch("sharing_service._run_tailscale", side_effect=fake_run),
            patch("sharing_service.local_krea_target_status", return_value={"ok": True, "auth_required": True, "message": "ok"}),
            patch("sharing_service.public_funnel_probe", return_value={"ok": True, "message": "public ok"}),
        ):
            result = sharing_service.repair_funnel(port=45678)

        self.assertTrue(result["ok"])
        self.assertTrue(result["local_target"]["ok"])
        self.assertTrue(result["tailscale"]["connected"])
        self.assertTrue(result["funnel"]["running"])
        self.assertIn(["up"], calls)
        self.assertIn(["funnel", "--set-path=/krea", "--bg", "--yes", "127.0.0.1:45678"], calls)

    def test_repair_funnel_warns_when_public_probe_fails(self) -> None:
        class Result:
            def __init__(self, returncode=0, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def fake_run(args, timeout=30):
            if args[:2] == ["status", "--json"]:
                return Result(stdout='{"BackendState":"Running","Self":{"DNSName":"diffusion.tail.ts.net.","TailscaleIPs":["100.64.0.1"]}}')
            if args[:2] == ["funnel", "status"]:
                return Result(stdout="https://diffusion.tail.ts.net/krea\n")
            if args[:1] == ["up"]:
                return Result()
            if args[:1] == ["funnel"]:
                return Result(stdout="https://diffusion.tail.ts.net\n")
            return Result()

        with (
            patch("sharing_service.find_tailscale", return_value="tailscale"),
            patch("sharing_service._run_tailscale", side_effect=fake_run),
            patch("sharing_service.local_krea_target_status", return_value={"ok": True, "auth_required": True, "message": "ok"}),
            patch("sharing_service.public_funnel_probe", return_value={"ok": False, "message": "TLS failed"}),
        ):
            result = sharing_service.repair_funnel(port=45678)

        self.assertFalse(result["ok"])
        self.assertTrue(result["needs_admin_service_restart"])
        self.assertIn("restart", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
