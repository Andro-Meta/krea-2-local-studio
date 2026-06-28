from __future__ import annotations

import http.cookiejar
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SERVER_PY = ROOT / "venv" / "Scripts" / "python.exe"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import share_auth


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ShareServerSmokeTests(unittest.TestCase):
    def test_share_mode_requires_login_under_krea_path(self) -> None:
        if not SERVER_PY.exists():
            self.skipTest("project venv is required for server smoke test")
        port = free_port()
        with tempfile.TemporaryDirectory() as td:
            auth_file = Path(td) / "share_auth.json"
            db_path = Path(td) / "app.db"
            share_auth.add_user(auth_file, "alice", "correct horse", role="admin")
            share_auth.add_user(auth_file, "bob", "correct horse", role="user")
            share_auth.add_user(auth_file, "kid", "correct horse", role="child")
            env = os.environ.copy()
            env.update(
                {
                    "KREA_SHARE_AUTH": "1",
                    "KREA_SHARE_AUTH_FILE": str(auth_file),
                    "KREA_SHARE_SECRET": "smoke-secret",
                    "KREA_PUBLIC_BASE_PATH": "/krea",
                    "KREA2_AUTO_CHECKPOINT": "__disabled_for_smoke__",
                    "DB_PATH": str(db_path),
                }
            )
            proc = subprocess.Popen(
                [
                    str(SERVER_PY),
                    "-m",
                    "uvicorn",
                    "backend.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ],
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.time() + 45
                while time.time() < deadline:
                    try:
                        urllib.request.urlopen(f"{base}/krea/api/auth/me", timeout=1).close()
                        break
                    except Exception:
                        time.sleep(0.25)
                else:
                    self.fail("share-mode server did not start")

                with self.assertRaises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(f"{base}/krea/api/settings", timeout=2)
                self.assertEqual(err.exception.code, 401)
                err.exception.close()

                index = ROOT / "frontend" / "dist" / "index.html"
                if index.exists():
                    match = re.search(r"src=\"\./(assets/[^\"]+\.js)\"", index.read_text(encoding="utf-8"))
                    self.assertIsNotNone(match)
                    with urllib.request.urlopen(f"{base}/krea/{match.group(1)}", timeout=3) as res:
                        self.assertEqual(res.status, 200)
                        self.assertIn("javascript", res.headers.get("Content-Type", ""))

                jar = http.cookiejar.CookieJar()
                opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
                body = json.dumps({"username": "alice", "password": "correct horse"}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/krea/api/auth/login",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(req, timeout=3) as res:
                    self.assertEqual(res.status, 200)

                with opener.open(f"{base}/krea/api/auth/me", timeout=3) as res:
                    data = json.loads(res.read().decode("utf-8"))
                self.assertEqual(data["username"], "alice")
                self.assertTrue(data["authenticated"])
                self.assertEqual(data["role"], "admin")

                bob_jar = http.cookiejar.CookieJar()
                bob = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(bob_jar))
                body = json.dumps({"username": "bob", "password": "correct horse"}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/krea/api/auth/login",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with bob.open(req, timeout=3) as res:
                    self.assertEqual(res.status, 200)
                with self.assertRaises(urllib.error.HTTPError) as err:
                    bob.open(f"{base}/krea/api/settings", timeout=3)
                self.assertEqual(err.exception.code, 403)
                err.exception.close()

                kid_jar = http.cookiejar.CookieJar()
                kid = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(kid_jar))
                body = json.dumps({"username": "kid", "password": "correct horse"}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/krea/api/auth/login",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with kid.open(req, timeout=3) as res:
                    self.assertEqual(res.status, 200)
                body = json.dumps({"prompt": "photorealistic nude woman", "mode": "txt2img"}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/krea/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with kid.open(req, timeout=3) as res:
                    blocked = json.loads(res.read().decode("utf-8"))
                self.assertEqual(blocked["status"], "blocked")
                self.assertGreater(blocked["moderation_event_id"], 0)

                with opener.open(f"{base}/krea/api/moderation/events", timeout=3) as res:
                    events = json.loads(res.read().decode("utf-8"))
                self.assertEqual(events["items"][0]["username"], "kid")
                self.assertEqual(events["items"][0]["action"], "block_prompt")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
