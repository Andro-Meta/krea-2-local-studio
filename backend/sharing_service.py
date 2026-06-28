from __future__ import annotations

import re
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("KREA_SERVER_PORT", "8200"))
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


def foreground_funnel_ports() -> list[tuple[int, int]]:
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { ([string]$_.CommandLine) -match 'tailscale(\\.exe)?\"?\\s+funnel\\s+\\d+' } | "
        "ForEach-Object { if (([string]$_.CommandLine) -match 'funnel\\s+(\\d+)') { "
        "('{0}:{1}' -f $_.ProcessId,$Matches[1]) } }"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            creationflags=NO_WINDOW,
            timeout=5,
        )
    except Exception:
        return []
    found: list[tuple[int, int]] = []
    for line in res.stdout.splitlines():
        pid, _, port = line.partition(":")
        if pid.strip().isdigit() and port.strip().isdigit():
            found.append((int(pid), int(port)))
    return found


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
    args = ["funnel", f"--set-path={PUBLIC_PATH}", "--bg", "--yes", f"127.0.0.1:{port}"]
    res = _run_tailscale(args, timeout=45)
    output = (res.stdout + res.stderr).strip()
    if res.returncode != 0 and "foreground listener already exists for port 443" in output:
        messages = [output] if output else []
        for pid, existing_port in foreground_funnel_ports():
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=NO_WINDOW)
            root_res = _run_tailscale(["funnel", "--set-path=/", "--bg", "--yes", f"127.0.0.1:{existing_port}"], timeout=45)
            root_output = (root_res.stdout + root_res.stderr).strip()
            if root_output:
                messages.append(root_output)
        res = _run_tailscale(args, timeout=45)
        retry_output = (res.stdout + res.stderr).strip()
        if retry_output:
            messages.append(retry_output)
        output = "\n".join(messages)
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


def local_krea_target_status(port: int = PORT) -> dict:
    url = f"http://127.0.0.1:{int(port)}{PUBLIC_PATH}/api/auth/me"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "auth_required": False, "message": f"Local Krea returned HTTP {exc.code}."}
    except Exception:
        return {"ok": False, "url": url, "auth_required": False, "message": "Local Krea is not reachable on the expected sharing port."}
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        data = {}
    # Auth-enabled /api/auth/me returns authenticated=false with no share_auth=false.
    # Auth-disabled local mode returns share_auth=false.
    auth_required = data.get("share_auth") is not False
    if not auth_required:
        return {
            "ok": True,
            "url": url,
            "auth_required": False,
            "message": "Local Krea is reachable, but the login gate is off. Restart with run.bat sharing mode before exposing Funnel.",
        }
    return {"ok": True, "url": url, "auth_required": True, "message": "Local Krea is reachable and login-gated."}


def repair_funnel(port: int = PORT) -> dict:
    local = local_krea_target_status(port)
    tailscale = tailscale_status()
    up_result = {"ok": False, "message": ""}
    if tailscale.get("installed"):
        try:
            up_result = tailscale_up()
        except Exception:
            up_result = {"ok": False, "message": "Could not run tailscale up from the GUI. Try opening Tailscale or running tailscale up manually."}
    funnel_result = start_funnel(port) if local.get("ok") and tailscale.get("installed") else {"ok": False, "url": "", "message": "Local Krea or Tailscale is not ready."}
    funnel = funnel_status()
    ok = bool(local.get("ok") and local.get("auth_required") and tailscale.get("installed") and funnel.get("running"))
    needs_admin_restart = bool(tailscale.get("installed") and funnel.get("running") and local.get("ok") and not ok)
    message = "Sharing is configured." if ok else (
        "Funnel is configured but public access may still fail. If the public URL returns 500/TLS errors, restart the Tailscale Windows service as Administrator."
        if needs_admin_restart else "Sharing is not ready. Check local Krea, login gate, and Tailscale status."
    )
    return {
        "ok": ok,
        "message": message,
        "local_target": local,
        "tailscale": tailscale_status(),
        "tailscale_up": up_result,
        "start_funnel": funnel_result,
        "funnel": funnel,
        "needs_admin_service_restart": needs_admin_restart,
    }


def stop_funnel() -> dict:
    res = _run_tailscale(["funnel", f"--set-path={PUBLIC_PATH}", "off"], timeout=20)
    output = (res.stdout + res.stderr).strip()
    return {"ok": res.returncode == 0, "message": output}


def tailscale_up() -> dict:
    res = _run_tailscale(["up"], timeout=60)
    output = (res.stdout + res.stderr).strip()
    return {"ok": res.returncode == 0, "message": output}
