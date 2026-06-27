from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

ITERS = 200_000
SESSION_TTL_SECONDS = 12 * 60 * 60
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


def resolve_auth_enabled(config_value: str | None, *, has_users: bool) -> bool:
    if config_value is not None:
        normalized = config_value.strip().lower()
        if normalized in TRUTHY:
            return True
        if normalized in FALSY:
            return False
    return has_users


def resolve_auto_funnel_enabled(config_value: str | None, *, auth_enabled: bool, has_admin: bool = True) -> bool:
    if not auth_enabled or not has_admin or config_value is None:
        return False
    return config_value.strip().lower() in TRUTHY


def load_users(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_users(path: Path, users: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(users, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        ITERS,
    ).hex()


def _normalize_role(role: str) -> str:
    return "admin" if role == "admin" else "user"


def is_valid_username(username: str) -> bool:
    return bool(USERNAME_RE.fullmatch(username.strip()))


def add_user(path: Path, username: str, password: str, role: str | None = None) -> None:
    username = username.strip()
    if not is_valid_username(username):
        raise ValueError("username must be 1-64 characters: letters, numbers, dots, dashes, or underscores")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    users = load_users(path)
    salt = os.urandom(16).hex()
    assigned_role = _normalize_role(role or ("admin" if not users else "user"))
    users[username] = {"salt": salt, "hash": _hash_password(password, salt), "role": assigned_role}
    save_users(path, users)


def remove_user(path: Path, username: str) -> bool:
    users = load_users(path)
    if username not in users:
        return False
    if _normalize_role(users[username].get("role", "admin")) == "admin" and _admin_count(users) <= 1:
        raise ValueError("cannot remove the last admin")
    del users[username]
    save_users(path, users)
    return True


def list_users(path: Path) -> list[str]:
    return sorted(load_users(path))


def list_user_records(path: Path) -> list[dict[str, str]]:
    users = load_users(path)
    return [
        {"username": username, "role": _normalize_role(rec.get("role", "admin"))}
        for username, rec in sorted(users.items())
    ]


def get_user_role(path: Path, username: str) -> str | None:
    rec = load_users(path).get(username)
    if not rec:
        return None
    return _normalize_role(rec.get("role", "admin"))


def _admin_count(users: dict[str, dict[str, str]]) -> int:
    return sum(1 for rec in users.values() if _normalize_role(rec.get("role", "admin")) == "admin")


def has_admin(path: Path) -> bool:
    return _admin_count(load_users(path)) > 0


def set_user_role(path: Path, username: str, role: str) -> bool:
    users = load_users(path)
    if username not in users:
        return False
    new_role = _normalize_role(role)
    old_role = _normalize_role(users[username].get("role", "admin"))
    if old_role == "admin" and new_role != "admin" and _admin_count(users) <= 1:
        raise ValueError("cannot demote the last admin")
    users[username]["role"] = new_role
    save_users(path, users)
    return True


def is_admin(path: Path, username: str | None) -> bool:
    return bool(username and get_user_role(path, username) == "admin")


def verify_user(path: Path, username: str, password: str) -> bool:
    if not is_valid_username(username):
        return False
    rec = load_users(path).get(username)
    if not rec:
        return False
    salt = rec.get("salt", "")
    expected = rec.get("hash", "")
    if not salt or not expected:
        return False
    actual = _hash_password(password, salt)
    return hmac.compare_digest(actual, expected)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(payload_b64: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()


def create_session_token(username: str, secret: str, now: int | None = None) -> str:
    issued = int(time.time() if now is None else now)
    payload = {"sub": username, "iat": issued, "exp": issued + SESSION_TTL_SECONDS}
    payload_b64 = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64, secret)}"


def verify_session_token(
    token: str | None,
    secret: str,
    users_path: Path,
    now: int | None = None,
) -> str | None:
    if not token or "." not in token or not secret:
        return None
    payload_b64, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(payload_b64, secret)):
        return None
    try:
        payload: dict[str, Any] = json.loads(_unb64(payload_b64).decode("utf-8"))
    except Exception:
        return None
    username = payload.get("sub")
    exp = payload.get("exp")
    current = int(time.time() if now is None else now)
    if not isinstance(username, str) or not isinstance(exp, int) or exp < current:
        return None
    if username not in load_users(users_path):
        return None
    return username


def make_secret() -> str:
    return secrets.token_urlsafe(32)
