from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path
from typing import TextIO


def _pump(src: TextIO, *targets: TextIO) -> None:
    for line in src:
        for target in targets:
            target.write(line)
            target.flush()


def run_with_log(command: list[str], *, log_path: Path, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        threads = [
            threading.Thread(target=_pump, args=(proc.stdout, stdout, log), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, stderr, log), daemon=True),
        ]
        for thread in threads:
            thread.start()
        code = proc.wait()
        for thread in threads:
            thread.join(timeout=5)
        proc.stdout.close()
        proc.stderr.close()
        return int(code)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command while mirroring stdout/stderr to a log file.")
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")
    return run_with_log(command, log_path=Path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
