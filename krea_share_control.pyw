#!/usr/bin/env python
"""Krea 2 Studio sharing control panel.

Starts a localhost-only, login-gated Krea server and exposes it through
Tailscale Funnel under /krea so it does not take over another Funnel root.
"""
from __future__ import annotations

import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import share_auth

PORT = 8200
PUBLIC_PATH = "/krea"
AUTH_FILE = ROOT / "share_auth.json"
ENV_FILE = ROOT / ".env"
PY = sys.executable
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TS_DL = "https://tailscale.com/download/windows"
URL_RE = re.compile(r"https://[\w.-]+\.ts\.net\S*")


def find_tailscale() -> str | None:
    for p in (
        r"C:\Program Files\Tailscale\tailscale.exe",
        r"C:\Program Files (x86)\Tailscale\tailscale.exe",
        shutil.which("tailscale"),
    ):
        if p and Path(p).exists():
            return p
    return None


def read_env_file() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def port_open(host: str = "127.0.0.1", port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def port_owner_pid(port: int = PORT) -> int | None:
    cmd = (
        "$c = Get-NetTCPConnection -LocalPort "
        + str(port)
        + " -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "$c.OwningProcess"
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
        return None
    out = res.stdout.strip()
    return int(out) if out.isdigit() else None


def process_command_line(pid: int) -> str:
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"ProcessId="
        + str(pid)
        + "\" | Select-Object -ExpandProperty CommandLine"
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
        return ""
    return res.stdout.strip()


def stop_other_krea_controls() -> None:
    current = os.getpid()
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { ([string]$_.CommandLine) -like '*krea_share_control.pyw*' "
        + f"-and $_.ProcessId -ne {current}"
        + " } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, creationflags=NO_WINDOW)


def kill_all_project_krea_servers() -> list[int]:
    root = str(ROOT).replace("'", "''").lower()
    cmd = (
        "$root='"
        + root
        + "'; Get-CimInstance Win32_Process | "
        "Where-Object { $cmd=([string]$_.CommandLine).ToLower(); "
        "($cmd -like ('*' + $root + '*')) -and ($cmd -like '*backend.main:app*') } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            creationflags=NO_WINDOW,
            timeout=10,
        )
    except Exception:
        return []
    return [int(line.strip()) for line in res.stdout.splitlines() if line.strip().isdigit()]


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


def kill_project_krea_server(port: int = PORT) -> int | None:
    pid = port_owner_pid(port)
    if not pid:
        return None
    cmdline = process_command_line(pid)
    normalized_root = str(ROOT).lower()
    normalized_cmd = cmdline.lower()
    is_project_krea = (
        normalized_root in normalized_cmd
        and "uvicorn" in normalized_cmd
        and "backend.main:app" in normalized_cmd
    )
    if not is_project_krea:
        return None
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=NO_WINDOW)
    deadline = time.time() + 8
    while time.time() < deadline:
        if not port_open(port=port):
            return pid
        time.sleep(0.25)
    return pid


def public_krea_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith(PUBLIC_PATH):
        return base + "/"
    return base + PUBLIC_PATH + "/"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.server: subprocess.Popen | None = None
        self.funnel: subprocess.Popen | None = None
        self.url: str | None = None
        self.ts = find_tailscale()
        self.q: queue.Queue[tuple[str, str]] = queue.Queue()
        self.share_secret = secrets.token_urlsafe(32)

        root.title("Krea 2 Studio — Sharing")
        root.geometry("610x540")
        pad = {"padx": 10, "pady": 4}

        top = ttk.LabelFrame(root, text="Status")
        top.pack(fill="x", **pad)
        self.s_srv = ttk.Label(top, text="• Krea server: stopped", foreground="#888")
        self.s_srv.pack(anchor="w", padx=8, pady=2)
        self.s_fun = ttk.Label(top, text="• Public link: stopped", foreground="#888")
        self.s_fun.pack(anchor="w", padx=8, pady=2)
        urlrow = ttk.Frame(top)
        urlrow.pack(fill="x", padx=8, pady=4)
        self.url_var = tk.StringVar(value="(start sharing to get a link)")
        ttk.Entry(urlrow, textvariable=self.url_var, state="readonly").pack(side="left", fill="x", expand=True)
        ttk.Button(urlrow, text="Copy", width=6, command=self.copy_url).pack(side="left", padx=4)
        ttk.Button(urlrow, text="Open", width=6, command=self.open_url).pack(side="left")

        ctl = ttk.Frame(root)
        ctl.pack(fill="x", **pad)
        self.btn_start = ttk.Button(ctl, text="Start sharing", command=self.start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(ctl, text="Stop sharing", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(ctl, text="Open local", command=lambda: webbrowser.open(f"http://127.0.0.1:{PORT}{PUBLIC_PATH}/")).pack(side="left", padx=4)
        ttk.Button(ctl, text="Run tailscale up", command=self.tailscale_up).pack(side="left", padx=4)
        if not self.ts:
            ttk.Button(ctl, text="Install Tailscale", command=lambda: webbrowser.open(TS_DL)).pack(side="right", padx=4)

        lf = ttk.LabelFrame(root, text="Logins")
        lf.pack(fill="both", expand=True, **pad)
        self.users = tk.Listbox(lf, height=7)
        self.users.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        ub = ttk.Frame(lf)
        ub.pack(side="left", fill="y", pady=6)
        ttk.Button(ub, text="Add login", command=self.add_login).pack(fill="x", padx=6, pady=3)
        ttk.Button(ub, text="Revoke", command=self.revoke).pack(fill="x", padx=6, pady=3)
        ttk.Button(ub, text="Refresh", command=self.refresh_users).pack(fill="x", padx=6, pady=3)

        lg = ttk.LabelFrame(root, text="Activity")
        lg.pack(fill="both", expand=True, **pad)
        self.logbox = tk.Text(lg, height=7, state="disabled", wrap="word", font=("Consolas", 9))
        self.logbox.pack(fill="both", expand=True, padx=6, pady=6)

        self.refresh_users()
        if self.ts:
            self.log(f"Tailscale found: {self.ts}")
            self.log(f"Public sharing will use path {PUBLIC_PATH}, not the root Funnel URL.")
        else:
            self.log("Tailscale not found. Install it, run tailscale up, then reopen this panel.")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(300, self._drain)

    def log(self, msg: str) -> None:
        self.logbox.config(state="normal")
        self.logbox.insert("end", msg.rstrip() + "\n")
        self.logbox.see("end")
        self.logbox.config(state="disabled")

    def refresh_users(self) -> None:
        self.users.delete(0, "end")
        self._usernames = share_auth.list_users(AUTH_FILE)
        for u in self._usernames:
            self.users.insert("end", u)

    def add_login(self) -> None:
        name = simpledialog.askstring("Add login", "Username:", parent=self.root)
        if not name:
            return
        pw = simpledialog.askstring("Add login", f"Password for {name} (min 8 chars):", parent=self.root, show="•")
        if not pw:
            return
        try:
            share_auth.add_user(AUTH_FILE, name.strip(), pw)
        except ValueError as e:
            return messagebox.showwarning("Add login", str(e))
        self.refresh_users()
        self.log(f"login '{name.strip()}' saved")

    def revoke(self) -> None:
        sel = self.users.curselection()
        if not sel:
            return
        u = self._usernames[sel[0]]
        if messagebox.askyesno("Revoke", f"Revoke access for '{u}'?"):
            share_auth.remove_user(AUTH_FILE, u)
            self.refresh_users()
            self.log(f"revoked '{u}'")

    def copy_url(self) -> None:
        if self.url:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.url)
            self.log("URL copied to clipboard")

    def open_url(self) -> None:
        if self.url:
            webbrowser.open(self.url)

    def tailscale_up(self) -> None:
        if not self.ts:
            webbrowser.open(TS_DL)
            return
        self.log("running tailscale up...")
        subprocess.Popen([self.ts, "up"], creationflags=NO_WINDOW)

    def start(self) -> None:
        if not share_auth.list_users(AUTH_FILE):
            return messagebox.showwarning("Start sharing", "Add at least one login first.")
        if self.server:
            return
        if port_open():
            killed_pids = kill_all_project_krea_servers()
            for pid in killed_pids:
                self.log(f"stopped existing Krea server on port {PORT} (PID {pid})")
            deadline = time.time() + 8
            while time.time() < deadline and port_open():
                time.sleep(0.25)
            if port_open():
                return messagebox.showerror(
                    "Start sharing",
                    f"Port {PORT} is already in use by another process. Stop it before sharing.",
                )

        env = os.environ.copy()
        file_env = read_env_file()
        auto_checkpoint = file_env.get("KREA2_AUTO_CHECKPOINT") or file_env.get("KREA2_TURBO_PATH", "")
        auto_quant = file_env.get("KREA2_AUTO_QUANT") or "fp8"
        env.update(
            {
                "KREA_SHARE_AUTH": "1",
                "KREA_SHARE_AUTH_FILE": str(AUTH_FILE),
                "KREA_SHARE_SECRET": self.share_secret,
                "KREA_SHARE_COOKIE_SECURE": "0",
                "KREA_PUBLIC_BASE_PATH": PUBLIC_PATH,
                "KREA2_AUTO_CHECKPOINT": auto_checkpoint,
                "KREA2_AUTO_QUANT": auto_quant,
            }
        )
        if auto_checkpoint:
            self.log(f"auto-loading model at startup: {auto_checkpoint} [{auto_quant}]")
        else:
            self.log("WARNING: no KREA2_AUTO_CHECKPOINT or KREA2_TURBO_PATH found in .env")
        self.server = subprocess.Popen(
            [PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "info"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW,
        )
        self.s_srv.config(text=f"• Krea server: RUNNING (127.0.0.1:{PORT}, auth on)", foreground="#1a7f37")
        self.log(f"Krea server started on 127.0.0.1:{PORT}{PUBLIC_PATH}/")

        if self.ts:
            self._start_funnel()
        else:
            self.s_fun.config(text="• Public link: needs Tailscale", foreground="#b00")
            self.url = f"http://127.0.0.1:{PORT}{PUBLIC_PATH}/"
            self.url_var.set(self.url + "  (local only)")

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

    def _run_ts(self, args: list[str]) -> subprocess.CompletedProcess:
        assert self.ts is not None
        return subprocess.run(
            [self.ts, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW,
            timeout=30,
        )

    def _set_url_from_text(self, text: str) -> bool:
        for m in URL_RE.finditer(text):
            self.url = public_krea_url(m.group(0))
            self.url_var.set(self.url)
            self.s_fun.config(text="• Public link: LIVE", foreground="#1a7f37")
            self.log(f"public link LIVE: {self.url}")
            return True
        return False

    def _start_funnel(self) -> None:
        self.s_fun.config(text="• Public link: starting...", foreground="#b8860b")
        self.log("starting Tailscale Funnel at /krea...")
        args = ["funnel", f"--set-path={PUBLIC_PATH}", "--bg", "--yes", f"127.0.0.1:{PORT}"]
        res = self._run_ts(args)
        output = (res.stdout + res.stderr).strip()
        if output:
            self.log(output)

        if res.returncode != 0 and "foreground listener already exists for port 443" in output:
            for pid, existing_port in foreground_funnel_ports():
                self.log(f"preserving existing root Funnel on port {existing_port} (PID {pid})")
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=NO_WINDOW)
                root_res = self._run_ts(["funnel", "--set-path=/", "--bg", "--yes", f"127.0.0.1:{existing_port}"])
                root_output = (root_res.stdout + root_res.stderr).strip()
                if root_output:
                    self.log(root_output)
            res = self._run_ts(args)
            output = (res.stdout + res.stderr).strip()
            if output:
                self.log(output)

        if res.returncode != 0:
            self.s_fun.config(text="• Public link: failed", foreground="#b00")
            messagebox.showerror("Tailscale Funnel", output or "Could not start Tailscale Funnel.")
            return

        status = self._run_ts(["funnel", "status"])
        status_text = status.stdout + status.stderr
        if status_text.strip():
            self.log(status_text.strip())
        if not self._set_url_from_text(status_text or output):
            self.url = f"http://127.0.0.1:{PORT}{PUBLIC_PATH}/"
            self.url_var.set(self.url + "  (local verified; public URL not parsed)")
            self.s_fun.config(text="• Public link: check status", foreground="#b8860b")

    def _read_funnel(self) -> None:
        if not self.funnel or not self.funnel.stdout:
            return
        found_url = False
        for line in self.funnel.stdout:
            self.q.put(("log", "funnel: " + line.rstrip()))
            m = URL_RE.search(line)
            if m:
                found_url = True
                self.q.put(("url", public_krea_url(m.group(0))))
        if not found_url and self.ts:
            try:
                status = subprocess.run(
                    [self.ts, "funnel", "status"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=NO_WINDOW,
                    timeout=10,
                )
                for m in URL_RE.finditer(status.stdout + status.stderr):
                    self.q.put(("url", public_krea_url(m.group(0))))
                    break
            except Exception as e:
                self.q.put(("log", f"funnel status failed: {e}"))

    def _drain(self) -> None:
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log(val)
                elif kind == "url":
                    self.url = val
                    self.url_var.set(self.url)
                    self.s_fun.config(text="• Public link: LIVE", foreground="#1a7f37")
                    self.log(f"public link LIVE: {self.url}")
        except queue.Empty:
            pass
        self.root.after(300, self._drain)

    def stop(self) -> None:
        if self.ts:
            try:
                self.log("turning off only the /krea Funnel route...")
                subprocess.run(
                    [self.ts, "funnel", f"--set-path={PUBLIC_PATH}", "off"],
                    capture_output=True,
                    creationflags=NO_WINDOW,
                    timeout=15,
                )
            except Exception as e:
                self.log(f"could not turn off /krea Funnel route: {e}")
        for proc in (self.funnel, self.server):
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self.funnel = self.server = None
        self.url = None
        self.s_srv.config(text="• Krea server: stopped", foreground="#888")
        self.s_fun.config(text="• Public link: stopped", foreground="#888")
        self.url_var.set("(start sharing to get a link)")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.log("stopped — /krea link is down")

    def on_close(self) -> None:
        if self.server and not messagebox.askyesno("Quit", "Stop sharing and close?"):
            return
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    if "--start" in sys.argv:
        root.after(600, app.start)
    root.mainloop()
