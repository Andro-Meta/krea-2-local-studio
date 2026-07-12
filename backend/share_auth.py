from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ITERS = 200_000
SESSION_TTL_SECONDS = 12 * 60 * 60
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}
ROLES = {"admin", "user", "child"}


class BootstrapCredentialDeletionError(RuntimeError):
    pass


def resolve_bootstrap_credential_path(
    base_dir: Path,
    *,
    marker_override: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    configured = marker_override or values.get("KREA_BOOTSTRAP_CREDENTIAL_FILE")
    if not configured:
        return base_dir / "data" / "private" / "first-admin-credential.json"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else base_dir / path


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
    normalized = str(role or "").strip().lower()
    return normalized if normalized in ROLES else "user"


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


def _restrict_credential_acl(path: Path) -> None:
    """Apply and verify mandatory owner-only credential permissions."""
    if os.name != "nt":
        path.chmod(0o600)
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o600:
            raise RuntimeError("credential permissions are not 0600")
        return
    username = os.environ.get("USERNAME", "").strip()
    if not username:
        raise RuntimeError("current Windows user is unavailable for credential ACL")
    try:
        result = subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"{username}:F",
                "SYSTEM:F",
            ],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("credential ACL command failed") from exc
    if result.returncode != 0:
        raise RuntimeError("credential ACL command returned an error")


def _secure_delete_credential(path: Path) -> None:
    """Overwrite the short-lived secret where possible, then unlink it."""
    if not path.exists():
        return
    try:
        size = path.stat().st_size
        with path.open("r+b", buffering=0) as stream:
            stream.write(b"\0" * size)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        # Unlink remains mandatory even if the filesystem rejects overwriting.
        pass
    path.unlink(missing_ok=True)


def bootstrap_first_admin(
    users_path: Path,
    credential_path: Path,
    *,
    username: str = "admin",
    password: str | None = None,
    generated_at: datetime | None = None,
) -> Path | None:
    """Create the first admin and a one-time credential file without printing it."""
    if load_users(users_path):
        return None
    password = password or secrets.token_urlsafe(15)
    generated_at = generated_at or datetime.now(timezone.utc)
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": username,
        "password": password,
        "generated_at": generated_at.isoformat(),
    }
    temp_path = credential_path.with_name(
        f".{credential_path.name}.{secrets.token_hex(8)}.tmp"
    )
    published = False
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _restrict_credential_acl(temp_path)
        # A same-filesystem hard link atomically publishes the already-secured
        # inode and refuses to overwrite any stale credential marker.
        os.link(temp_path, credential_path)
        published = True
        temp_path.unlink()
        add_user(users_path, username, password, role="admin")
    except Exception:
        cleanup_error: OSError | None = None
        cleanup_candidates = [temp_path]
        if published:
            cleanup_candidates.append(credential_path)
        for candidate in cleanup_candidates:
            try:
                _secure_delete_credential(candidate)
            except OSError as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise RuntimeError("credential setup failed and secure cleanup also failed") from cleanup_error
        raise
    return credential_path


def consume_bootstrap_credential(credential_path: Path, username: str, *, attempts: int = 3) -> bool:
    """Remove the one-time credential after its bootstrap admin signs in."""
    try:
        payload = json.loads(credential_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapCredentialDeletionError("one-time bootstrap credential could not be read") from exc
    if not isinstance(payload, dict) or payload.get("username") != username:
        return False
    last_error: OSError | None = None
    attempt_count = max(1, attempts)
    for attempt in range(attempt_count):
        try:
            credential_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempt_count:
                time.sleep(0.1)
    raise BootstrapCredentialDeletionError(
        "one-time bootstrap credential could not be deleted"
    ) from last_error


def verify_login(
    users_path: Path,
    username: str,
    password: str,
    *,
    bootstrap_credential_path: Path | None = None,
) -> bool:
    """Verify a login and consume its matching one-time bootstrap credential."""
    if not verify_user(users_path, username, password):
        return False
    if bootstrap_credential_path is not None:
        consume_bootstrap_credential(bootstrap_credential_path, username)
    return True


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
