"""Write optional API tokens into .env (called by install.bat).

Usage: set_env_tokens.py <hf_token> <civitai_token>
Blank arguments are skipped; existing values are only overwritten when a new
non-empty value is supplied.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env_path = ROOT / ".env"
    if not env_path.exists():
        template = ROOT / ".env.example"
        env_path.write_text(template.read_text(encoding="utf-8") if template.exists() else "", encoding="utf-8")
    text = env_path.read_text(encoding="utf-8", errors="replace")

    updates = {
        "HF_TOKEN": (sys.argv[1] if len(sys.argv) > 1 else "").strip(),
        "CIVITAI_TOKEN": (sys.argv[2] if len(sys.argv) > 2 else "").strip(),
    }
    for key, value in updates.items():
        if not value:
            continue
        if re.search(rf"^{key}=", text, flags=re.M):
            text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
        else:
            text += f"\n{key}={value}\n"
        print(f"  {key} saved to .env")
    env_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
