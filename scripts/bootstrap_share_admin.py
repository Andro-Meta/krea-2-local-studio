from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from share_auth import bootstrap_first_admin, resolve_bootstrap_credential_path  # noqa: E402


def main() -> int:
    credential_path = resolve_bootstrap_credential_path(ROOT)
    try:
        created = bootstrap_first_admin(ROOT / "share_auth.json", credential_path)
    except Exception as exc:
        print(f"ERROR: Could not create the first-admin credential file: {exc}", file=sys.stderr)
        return 1
    if created:
        print(f"First admin credentials written to: {created}")
        print("WARNING: This one-time file contains a password and is deleted after the first admin login.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
