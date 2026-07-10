from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


_comfy_stopped = False


def _stop_comfyui_once() -> None:
    """Stop ComfyUI at most once (guards against handler + finally double-calls)."""
    global _comfy_stopped
    if _comfy_stopped:
        return
    _comfy_stopped = True
    _stop_comfyui()


def _stop_comfyui() -> None:
    """Kill the ComfyUI process(es) started for this session, freeing VRAM/RAM."""
    try:
        root = Path(__file__).resolve().parents[1]
        backend = root / "backend"
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        from memory_manager import detect_krea_runtime_processes, _terminate_pid  # type: ignore
        for proc in detect_krea_runtime_processes():
            if proc.get("kind") == "comfyui":
                try:
                    _terminate_pid(int(proc["pid"]))
                except Exception:
                    pass
    except Exception:
        pass


def _pump(src: TextIO, *targets: TextIO) -> None:
    for line in src:
        for target in targets:
            target.write(line)
            target.flush()


def _tail(path: Path, prefix: str, targets: tuple[TextIO, ...], stop: threading.Event) -> None:
    """Follow a growing log file (e.g. ComfyUI's) and mirror new lines to targets,
    so a separately-launched process's output shows up in this console + log."""
    while not stop.is_set() and not path.exists():
        stop.wait(0.5)
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            while not stop.is_set():
                line = f.readline()
                if line:
                    for target in targets:
                        target.write(prefix + line)
                        target.flush()
                else:
                    stop.wait(0.3)
    except Exception:
        pass


def _spawn_shutdown_janitor(server_pid: int, *, stop_comfyui: bool) -> None:
    """Launch a detached janitor that outlives this console.

    Closing the run.bat window with the X button only gives console-attached
    processes ~5s before Windows hard-kills them, which is not enough to stop
    the detached ComfyUI engine. The janitor is not attached to this console,
    waits for this wrapper to exit (X button, Ctrl+C, or crash), then kills any
    leftover server tree and ComfyUI so VRAM/RAM is always freed.
    """
    try:
        janitor = Path(__file__).resolve().parent / "shutdown_janitor.py"
        args = [sys.executable, str(janitor), "--watch-pid", str(os.getpid()), "--kill-pid", str(int(server_pid))]
        if stop_comfyui:
            args.append("--stop-comfyui")
        # CREATE_NO_WINDOW (not DETACHED_PROCESS): the janitor gets its own
        # hidden console, detaching it from run.bat's console so the X button
        # can't kill it. DETACHED_PROCESS would make the venv launcher's child
        # interpreter allocate a NEW VISIBLE console window instead.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
    except Exception:
        pass


def run_with_log(command: list[str], *, log_path: Path, stdout: TextIO = sys.stdout,
                 tail_paths: tuple[Path, ...] = (), stop_comfyui_on_exit: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        stop_tails = threading.Event()
        tail_threads = []
        for tp in tail_paths:
            prefix = f"[{tp.stem}] "
            t = threading.Thread(target=_tail, args=(tp, prefix, (stdout, log), stop_tails), daemon=True)
            t.start()
            tail_threads.append(t)
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _spawn_shutdown_janitor(proc.pid, stop_comfyui=stop_comfyui_on_exit)
        assert proc.stdout is not None
        thread = threading.Thread(target=_pump, args=(proc.stdout, stdout, log), daemon=True)
        thread.start()
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            # Ctrl+C: the child (same console group) already received the signal
            # and is shutting itself down. Give it a moment to exit gracefully,
            # then terminate / kill if it hangs. Return 130 (standard SIGINT code).
            try:
                code = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
            code = code if code is not None else 130
        finally:
            stop_tails.set()
            thread.join(timeout=5)
            try:
                proc.stdout.close()
            except Exception:
                pass
        return int(code)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command while mirroring stdout/stderr to a log file.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--stop-comfyui", action="store_true",
                        help="When the wrapped server exits, also stop the ComfyUI image engine (free VRAM/RAM).")
    parser.add_argument("--tail", action="append", default=[],
                        help="Also follow this log file and mirror it into the console + server log "
                             "(e.g. the detached ComfyUI logs). May be given multiple times.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tail_paths = tuple(Path(p) for p in (args.tail or []))
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")

    if not args.stop_comfyui:
        return run_with_log(command, log_path=Path(args.log), tail_paths=tail_paths, stop_comfyui_on_exit=False)

    # Best-effort teardown of ComfyUI when the server stops (Ctrl+C / SIGBREAK).
    # ComfyUI is launched detached, so it does NOT receive the console Ctrl+C -
    # only we can stop it. Do it IMMEDIATELY in the handler (before cmd's
    # "Terminate batch job? (Y/N)" can hard-kill this process and skip cleanup),
    # then interrupt the blocking wait. The once-guard + finally cover the paths
    # where the handler doesn't fire (SIGTERM, normal child exit).
    def _handler(signum, frame):
        _stop_comfyui_once()
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except Exception:
                pass
    try:
        return run_with_log(command, log_path=Path(args.log), tail_paths=tail_paths, stop_comfyui_on_exit=True)
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_comfyui_once()


if __name__ == "__main__":
    raise SystemExit(main())
