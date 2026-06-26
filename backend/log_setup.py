from __future__ import annotations
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: Path) -> None:
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        logs_dir / "krea2_studio.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


def flush_all() -> None:
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception as exc:
            logging.getLogger(__name__).debug("Log handler flush failed: %s", exc)
