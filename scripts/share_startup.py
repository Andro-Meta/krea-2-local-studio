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
from sharing_service import PUBLIC_PATH  # noqa: E402


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
    """Rebind /krea Funnel to the live KREA_SERVER_PORT when sharing is enabled.

    run.bat picks a random port each launch, so a stale Funnel target is the usual
    cause of ERR_CONNECTION_CLOSED on the public URL after Ctrl+C restart.
    """
    status = sharing_service.funnel_status()
    funnel_already = bool(status.get("running") or (PUBLIC_PATH in str(status.get("message") or "")))
    if not auto_funnel and not funnel_already:
        return ""
    up = sharing_service.tailscale_up()
    if not up.get("ok"):
        print(f"[share] Tailscale up failed: {up.get('message') or up}", flush=True)
        return ""
    port = sharing_service.current_server_port()
    result = sharing_service.start_funnel(port)
    if not result.get("ok"):
        print(f"[share] Funnel start failed: {result.get('message') or result}", flush=True)
        return ""
    url = str(result.get("url") or status.get("url") or "")
    if not url:
        print("[share] Funnel rebound but no public URL was reported.", flush=True)
        return ""
    probe = sharing_service.public_funnel_probe_with_retries(url)
    if probe.get("ok"):
        print(f"[share] Public Funnel ready: {url}", flush=True)
        return url
    # The Windows Funnel ingress session commonly goes stale across restarts:
    # `funnel status` looks correct but the edge returns 500 / closes TLS.
    # Re-registering the node (down/up) heals it without touching the serve
    # path config, so other apps' Funnel entries on this machine survive.
    print(
        f"[share] Public probe failed ({probe.get('message') or probe}); "
        "cycling the Tailscale connection to re-register the Funnel ingress...",
        flush=True,
    )
    cycle = sharing_service.cycle_tailscale_connection()
    if not cycle.get("ok"):
        print(f"[share] Tailscale reconnect failed: {cycle.get('message') or cycle}", flush=True)
        return url
    probe = sharing_service.public_funnel_probe_with_retries(url, attempts=6, delay_seconds=5.0)
    if probe.get("ok"):
        print(f"[share] Public Funnel ready after reconnect: {url}", flush=True)
        return url
    print(
        f"[share] Funnel is bound to 127.0.0.1:{port} but the public URL still fails: "
        f"{probe.get('message') or probe}. Local URL works. "
        "Try System > Tailscale Sharing > Repair, or restart the Tailscale Windows service as Administrator.",
        flush=True,
    )
    return url


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
