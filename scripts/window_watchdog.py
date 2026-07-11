"""Black-box window forensics watchdog.

Records every top-level window the moment it becomes visible, with the owning
process, its command line, and full parent chain. When a borderless untitled
window large enough to be "the black box" appears, it additionally samples the
window's pixels to confirm it is black and saves a screenshot as evidence.

Run modes:
    python scripts/window_watchdog.py            # watch forever, log JSONL
    python scripts/window_watchdog.py --snapshot # dump all current windows and exit

Log: logs/window_watchdog.jsonl  (one JSON object per event)
Evidence shots: logs/watchdog_shots/
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "window_watchdog.jsonl"
SHOTS_DIR = ROOT / "logs" / "watchdog_shots"

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_CAPTION = 0x00C00000
DWMWA_CLOAKED = 14

EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def _get_window_long(hwnd: int, index: int) -> int:
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        return user32.GetWindowLongPtrW(hwnd, index)
    return user32.GetWindowLongW(hwnd, index)


def _window_info(hwnd: int) -> dict:
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title, 512)
    cls = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, cls, 256)
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    cloaked = wt.DWORD()
    dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    style = _get_window_long(hwnd, GWL_STYLE)
    exstyle = _get_window_long(hwnd, GWL_EXSTYLE)
    return {
        "hwnd": hwnd,
        "title": title.value,
        "class": cls.value,
        "rect": [rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top],
        "pid": pid.value,
        "cloaked": cloaked.value,
        "style": f"0x{style & 0xFFFFFFFF:08X}",
        "exstyle": f"0x{exstyle & 0xFFFFFFFF:08X}",
        "has_caption": bool(style & WS_CAPTION),
    }


def _process_chain(pid: int, depth: int = 5) -> list[dict]:
    """Process + ancestors: the 'who spawned this' evidence."""
    chain: list[dict] = []
    try:
        proc: psutil.Process | None = psutil.Process(pid)
    except psutil.Error:
        return [{"pid": pid, "error": "process gone"}]
    while proc is not None and depth > 0:
        try:
            chain.append({
                "pid": proc.pid,
                "name": proc.name(),
                "exe": proc.exe() if proc.is_running() else "",
                "cmdline": " ".join(proc.cmdline()),
                "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(timespec="seconds"),
            })
            proc = proc.parent()
        except psutil.Error:
            break
        depth -= 1
    return chain


def _enum_visible_windows() -> dict[int, dict]:
    found: dict[int, dict] = {}

    def cb(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            info = _window_info(hwnd)
            w, h = info["rect"][2], info["rect"][3]
            if w > 0 and h > 0:
                found[hwnd] = info
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return found


def _is_black_box_candidate(info: dict) -> bool:
    """Borderless, untitled, not cloaked, big enough to be the mystery box."""
    w, h = info["rect"][2], info["rect"][3]
    return (
        w >= 250 and h >= 150
        and not info["has_caption"]
        and info["cloaked"] == 0
        and not info["title"].strip()
    )


def _sample_blackness(info: dict) -> float | None:
    """Fraction of sampled pixels that are near-black inside the window rect."""
    try:
        from PIL import ImageGrab

        left, top, w, h = info["rect"]
        img = ImageGrab.grab(bbox=(left, top, left + w, top + h))
        img = img.convert("RGB").resize((16, 16))
        pixels = list(img.getdata())
        dark = sum(1 for r, g, b in pixels if r < 24 and g < 24 and b < 24)
        return round(dark / len(pixels), 3)
    except Exception:
        return None


def _save_evidence_shot(tag: str) -> str:
    try:
        from PIL import ImageGrab

        SHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SHOTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}.png"
        ImageGrab.grab(all_screens=True).save(path)
        return str(path)
    except Exception as exc:
        return f"screenshot failed: {exc}"


def _log(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.now().isoformat(timespec="milliseconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def snapshot() -> None:
    windows = _enum_visible_windows()
    for info in windows.values():
        info["process_chain"] = _process_chain(info["pid"])
        info["black_box_candidate"] = _is_black_box_candidate(info)
        if info["black_box_candidate"]:
            info["blackness"] = _sample_blackness(info)
        print(json.dumps(info, ensure_ascii=False, indent=2))


def watch(poll_seconds: float = 0.4) -> None:
    _log({"event": "watchdog_start", "pid": psutil.Process().pid})
    known: dict[int, str] = {}  # hwnd -> last-seen signature
    first_pass = True
    while True:
        windows = _enum_visible_windows()
        for hwnd, info in windows.items():
            sig = f"{info['class']}|{info['pid']}"
            if known.get(hwnd) == sig:
                continue
            known[hwnd] = sig
            if first_pass:
                continue  # only log windows that appear AFTER we started
            info["process_chain"] = _process_chain(info["pid"])
            candidate = _is_black_box_candidate(info)
            info["event"] = "window_appeared"
            info["black_box_candidate"] = candidate
            if candidate:
                info["blackness"] = _sample_blackness(info)
                if info["blackness"] is not None and info["blackness"] >= 0.85:
                    info["event"] = "BLACK_BOX_DETECTED"
                    info["screenshot"] = _save_evidence_shot(f"pid{info['pid']}_{info['class'][:20]}")
            _log(info)
        # Forget hwnds that vanished so a reused handle logs again.
        for hwnd in list(known):
            if hwnd not in windows:
                del known[hwnd]
        first_pass = False
        time.sleep(poll_seconds)


def _already_running() -> bool:
    """Single-instance guard (named mutex) so autostart + manual launches
    don't stack multiple watchdogs.

    Must use use_last_error=True + ctypes.get_last_error(): calling
    kernel32.GetLastError() directly can read a stale value because ctypes
    itself makes API calls in between, which let duplicates slip through."""
    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, "Global\\KreaWindowWatchdog")
    return ctypes.get_last_error() == ERROR_ALREADY_EXISTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="dump current windows and exit")
    parser.add_argument("--poll", type=float, default=0.4, help="poll interval seconds")
    args = parser.parse_args()
    if args.snapshot:
        snapshot()
        return
    if _already_running():
        return
    try:
        watch(args.poll)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
