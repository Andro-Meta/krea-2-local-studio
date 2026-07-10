from __future__ import annotations

import os
from pathlib import Path

SECRET_ENV_KEYS = {"HF_TOKEN", "CIVITAI_TOKEN", "IDEOGRAM_API_KEY", "OPENROUTER_API_KEY"}


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def write_env(path: Path, env: dict[str, str]) -> None:
    existing = read_env(path)
    merged = {**existing, **env}
    lines = [f"{key}={value}" for key, value in merged.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def secret_value(env_key: str, setting_value: str | None, env_file: dict[str, str] | None = None) -> str:
    return (setting_value or os.environ.get(env_key) or (env_file or {}).get(env_key) or "").strip()
