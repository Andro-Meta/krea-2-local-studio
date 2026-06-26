from __future__ import annotations

import re
import json
import shutil
import subprocess
from pathlib import Path

PORT = 8200
PUBLIC_PATH = "/krea"
TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download/windows"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
URL_RE = re.compile(r"https://[\w.-]+\.ts\.net\S*")


def find_tailscale() -> str | None:
    for candidate in (
        r"C:\Program Files\Tailscale\tailscale.exe",
        r"C:\Program Files (x86)\Tailscale\tailscale.exe",
        shutil.which("tailscale"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def public_krea_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith(PUBLIC_PATH):
        return base + "/"
    return base + PUBLIC_PATH + "/"


def _run_tailscale(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    ts = find_tailscale()
    if not ts:
        raise RuntimeError("Tailscale is not installed. Install it from https://tailscale.com/download/windows.")
    return subprocess.run(
        [ts, *args],
        capture_output=True,
        text=True,
        creationflags=NO_WINDOW,
        timeout=timeout,
    )


def tailscale_status() -> dict:
    ts = find_tailscale()
    if not ts:
        return {
            "installed": False,
            "connected": False,
            "tailscale_path": None,
            "download_url": TAILSCALE_DOWNLOAD_URL,
            "message": "Tailscale is not installed.",
        }
    res = _run_tailscale(["status", "--json"], timeout=10)
    connected = res.returncode == 0
    message = (res.stderr or "").strip()
    if connected:
        try:
            data = json.loads(res.stdout or "{}")
            self_node = data.get("Self") or {}
            host = self_node.get("DNSName") or self_node.get("HostName") or "this device"
            message = f"Tailscale connected: {str(host).rstrip('.')}"
        except Exception:
            message = "Tailscale connected."
    elif not message:
        message = "Tailscale is installed but not connected. Run tailscale up."
    return {
        "installed": True,
        "connected": connected,
        "tailscale_path": ts,
        "download_url": TAILSCALE_DOWNLOAD_URL,
        "message": message,
    }


def funnel_status() -> dict:
    ts = find_tailscale()
    if not ts:
        return {"installed": False, "running": False, "url": "", "message": "Tailscale is not installed."}
    res = _run_tailscale(["funnel", "status"], timeout=15)
    output = (res.stdout + res.stderr).strip()
    url = ""
    for match in URL_RE.findall(output):
        url = public_krea_url(match)
        break
    return {
        "installed": True,
        "running": bool(url and PUBLIC_PATH in output),
        "url": url,
        "message": output,
    }


def start_funnel(port: int = PORT) -> dict:
    res = _run_tailscale(["funnel", f"--set-path={PUBLIC_PATH}", "--bg", "--yes", f"127.0.0.1:{port}"], timeout=45)
    output = (res.stdout + res.stderr).strip()
    if res.returncode != 0:
        return {"ok": False, "url": "", "message": output or "Tailscale Funnel failed to start."}
    url = ""
    for match in URL_RE.findall(output):
        url = public_krea_url(match)
        break
    if not url:
        status = funnel_status()
        url = status.get("url", "")
    return {"ok": True, "url": url, "message": output}


def stop_funnel() -> dict:
    res = _run_tailscale(["funnel", f"--set-path={PUBLIC_PATH}", "off"], timeout=20)
    output = (res.stdout + res.stderr).strip()
    return {"ok": res.returncode == 0, "message": output}


def tailscale_up() -> dict:
    res = _run_tailscale(["up"], timeout=60)
    output = (res.stdout + res.stderr).strip()
    return {"ok": res.returncode == 0, "message": output}
