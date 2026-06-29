from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import sharing_service  # noqa: E402


def read_env_file(path: Path = ROOT / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def env_bool(env: dict[str, str], key: str, default: bool = False) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def wait_for_url(url: str, *, timeout_seconds: int = 90, interval_seconds: float = 0.5) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = urllib.request.urlopen(url, timeout=2)
            response.close()
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                exc.close()
                return True
            time.sleep(interval_seconds)
        except Exception:
            time.sleep(interval_seconds)
    return False


def maybe_start_funnel(*, auto_funnel: bool) -> str:
    if not auto_funnel:
        return ""
    up = sharing_service.tailscale_up()
    if not up.get("ok"):
        return ""
    result = sharing_service.start_funnel()
    return str(result.get("url") or "") if result.get("ok") else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for Krea startup, optionally start Tailscale Funnel, then open a browser.")
    parser.add_argument("--ready-url", required=True)
    parser.add_argument("--open-url", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--auto-funnel", action="store_true")
    args = parser.parse_args()

    if not wait_for_url(args.ready_url, timeout_seconds=args.timeout):
        return 1
    public_url = maybe_start_funnel(auto_funnel=args.auto_funnel)
    webbrowser.open(public_url or args.open_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
