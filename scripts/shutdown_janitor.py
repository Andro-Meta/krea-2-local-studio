"""Detached shutdown janitor for run.bat sessions.

Closing the run.bat console with the X button gives console-attached processes
only ~5 seconds before Windows force-kills them - not enough for the in-process
cleanup (WMI scan + taskkill) to stop the detached ComfyUI engine. This script
is spawned DETACHED from the console, waits for the wrapper process to exit
(any exit path: X button, Ctrl+C, crash), then kills the leftover server
process tree and the ComfyUI engine so VRAM/RAM is always freed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def wait_for_pid_exit(pid: int, *, poll_seconds: float = 1.0) -> None:
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        INFINITE = 0xFFFFFFFF
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if handle:
            try:
                kernel32.WaitForSingleObject(handle, INFINITE)
                return
            finally:
                kernel32.CloseHandle(handle)
        # Could not open the process (already gone or access denied): poll.
    import os

    while True:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return
        time.sleep(poll_seconds)


def pid_is_krea_server(pid: int) -> bool:
    """Guard against PID reuse: only kill the PID if it still looks like our
    uvicorn server process."""
    if pid <= 0:
        return False
    if sys.platform != "win32":
        return True
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        command_line = (result.stdout or "").lower()
        return "uvicorn" in command_line and "backend.main" in command_line
    except Exception:
        return False


def kill_pid_tree(pid: int) -> None:
    if pid <= 0 or not pid_is_krea_server(pid):
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        import os
        import signal

        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass


def stop_comfyui() -> None:
    try:
        from memory_manager import _terminate_pid, detect_krea_runtime_processes

        for proc in detect_krea_runtime_processes():
            if proc.get("kind") == "comfyui":
                try:
                    _terminate_pid(int(proc["pid"]))
                except Exception:
                    pass
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for a watched PID to exit, then clean up Krea runtime processes.")
    parser.add_argument("--watch-pid", type=int, required=True, help="PID whose exit triggers cleanup (the run_with_log wrapper).")
    parser.add_argument("--kill-pid", type=int, default=0, help="Server process tree to kill after the watched PID exits.")
    parser.add_argument("--stop-comfyui", action="store_true", help="Also stop the detached ComfyUI engine.")
    args = parser.parse_args()

    wait_for_pid_exit(args.watch_pid)
    # Small grace: on Ctrl+C the wrapper's own cleanup usually finishes first,
    # making this a no-op; on X-button close it does the real work.
    time.sleep(1.0)
    kill_pid_tree(args.kill_pid)
    if args.stop_comfyui:
        stop_comfyui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
