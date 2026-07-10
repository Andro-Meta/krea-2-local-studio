"""
Rebuild the frontend bundle (frontend/dist) only when the source is newer than
the last build. Called from run.bat so a restart always serves your latest UI
changes instead of a stale bundle. Never blocks startup: if npm is missing or the
build fails, it warns and lets the server start with the existing bundle.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DIST_INDEX = FRONTEND / "dist" / "index.html"

# Files whose changes should trigger a rebuild.
WATCH_DIRS = [FRONTEND / "src"]
WATCH_FILES = [
    FRONTEND / "index.html",
    FRONTEND / "package.json",
    FRONTEND / "vite.config.ts",
    FRONTEND / "tsconfig.json",
]


def _newest_source_mtime() -> float:
    newest = 0.0
    for d in WATCH_DIRS:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    newest = max(newest, f.stat().st_mtime)
    for f in WATCH_FILES:
        if f.exists():
            newest = max(newest, f.stat().st_mtime)
    return newest


def main() -> int:
    if not FRONTEND.exists():
        return 0
    dist_mtime = DIST_INDEX.stat().st_mtime if DIST_INDEX.exists() else 0.0
    src_mtime = _newest_source_mtime()

    if DIST_INDEX.exists() and dist_mtime >= src_mtime:
        print("[frontend] bundle is up to date; skipping rebuild.", flush=True)
        return 0

    npm = shutil.which("npm")
    if not npm:
        print("[frontend] WARNING: npm not found on PATH; cannot rebuild the UI. "
              "Serving the existing bundle. Run install.bat or `npm run build` in frontend/.",
              flush=True)
        return 0

    reason = "no bundle found" if not DIST_INDEX.exists() else "source changed since last build"
    print(f"[frontend] Rebuilding UI bundle ({reason})... this takes ~30s.", flush=True)
    try:
        proc = subprocess.run([npm, "run", "build"], cwd=str(FRONTEND))
    except Exception as e:  # noqa: BLE001
        print(f"[frontend] WARNING: build could not start ({e}); serving existing bundle.", flush=True)
        return 0
    if proc.returncode != 0:
        print("[frontend] WARNING: UI build failed; serving the existing bundle. "
              "Check the errors above.", flush=True)
        return 0
    print("[frontend] UI bundle rebuilt.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
