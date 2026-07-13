from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.animation_plan import (
    DEFAULT_ANIMATION_CHUNK_SIZE,
    build_chunk_ranges,
    build_seed_plan,
)
from backend.schemas import AnimateRequest


SCHEMA_VERSION = 2
DEFAULT_CHUNK_SIZE = DEFAULT_ANIMATION_CHUNK_SIZE
LOCAL_OWNER_SEGMENT = "_local"
STATUSES = frozenset(
    {"queued", "running", "finalizing", "done", "error", "blocked", "cancelled"}
)
RECOVERABLE_STATUSES = frozenset({"queued", "running", "finalizing"})
TERMINAL_STATUSES = frozenset({"done", "error", "blocked", "cancelled"})
_JOB_ID_RE = re.compile(r"[A-Fa-f0-9](?:[A-Fa-f0-9-]{0,126}[A-Fa-f0-9])?")
_FRAME_INDEX_RE = re.compile(r".*?(\d+)\.(?:png|jpe?g|webp)", re.IGNORECASE)
_MAX_ERROR_LENGTH = 1024
_CREATE_MARKER = ".creating"
_LOCK_FILE = ".animation-store.lock"
_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_TRANSITIONS = {
    "queued": {"running", "error", "blocked", "cancelled"},
    "running": {"queued", "finalizing", "error", "blocked", "cancelled"},
    "finalizing": {"done", "error", "blocked", "cancelled"},
    "done": set(),
    "error": set(),
    "blocked": set(),
    "cancelled": set(),
}


class AnimationStateError(ValueError):
    """Persisted animation state is corrupt or unsupported."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_job_id(job_id: object) -> str:
    if not isinstance(job_id, str) or _JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError("job_id must be a safe UUID or hexadecimal-hyphen identifier")
    if job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
        raise ValueError("job_id contains an unsafe path component")
    return job_id.lower()


def _owner_segment(owner: object) -> str:
    if owner is None:
        return LOCAL_OWNER_SEGMENT
    if not isinstance(owner, str):
        raise ValueError("owner must be text or None")
    return "u-" + hashlib.sha256(owner.encode("utf-8")).hexdigest()


def _beneath(root: Path, path: Path) -> Path:
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"path escapes configured root: {path}") from exc
    return lexical_path


def _is_reparse(path: Path) -> bool:
    try:
        if hasattr(os.path, "isjunction") and os.path.isjunction(path):
            return True
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _assert_no_link_components(
    root: Path, path: Path, *, allow_final: bool = False
) -> Path:
    lexical = _beneath(root, path)
    relative = lexical.relative_to(_beneath(root, root))
    current = _beneath(root, root)
    for index, part in enumerate(relative.parts):
        current = current / part
        if not os.path.lexists(current):
            continue
        is_link = current.is_symlink() or _is_reparse(current)
        if is_link and not (allow_final and index == len(relative.parts) - 1):
            raise ValueError(f"path contains a link or reparse point: {current}")
    return lexical


def _mkdir_parents_no_links(root: Path, path: Path) -> Path:
    lexical = _beneath(root, path)
    current = _beneath(root, root)
    for part in lexical.relative_to(current).parts:
        current = current / part
        if not os.path.lexists(current):
            try:
                current.mkdir()
            except FileExistsError:
                pass
        if current.is_symlink() or _is_reparse(current):
            raise ValueError(f"path contains a link or reparse point: {current}")
        if not current.is_dir():
            raise ValueError(f"path component is not a directory: {current}")
    return lexical


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AnimationStateError(f"duplicate JSON object name: {key[:64]}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise AnimationStateError(f"invalid JSON constant: {value[:32]}")


def _root_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root))
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


def _relative_media_path(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AnimationStateError(f"corrupt {field_name}: expected a relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or path.anchor
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise AnimationStateError(f"corrupt {field_name}: unsafe relative path")
    return path.as_posix()


def _canonical_mismatch(
    actual: object, expected: object, path: str = "$"
) -> str | None:
    if type(actual) is not type(expected):
        return (
            f"{path} expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if isinstance(expected, dict):
        if len(actual) != len(expected):
            return f"{path} has missing or unexpected keys"
        unmatched = list(actual.items())
        for expected_key, expected_value in expected.items():
            matches = [
                (index, actual_value)
                for index, (actual_key, actual_value) in enumerate(unmatched)
                if type(actual_key) is type(expected_key)
                and actual_key == expected_key
            ]
            if len(matches) != 1:
                return f"{path} has ambiguous, missing, or extra typed keys"
            index, actual_value = matches[0]
            unmatched.pop(index)
            mismatch = _canonical_mismatch(
                actual_value, expected_value, f"{path}.{expected_key}"
            )
            if mismatch is not None:
                return mismatch
        if unmatched:
            return f"{path} has ambiguous, missing, or extra typed keys"
        return None
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} expected {len(expected)} items, got {len(actual)}"
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            mismatch = _canonical_mismatch(
                actual_item, expected_item, f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return f"{path} does not match its validated value"
    return None


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; Windows commonly rejects directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


@dataclass
class AnimationProject:
    schema_version: int
    revision: int
    job_id: str
    owner: str | None
    role: str
    status: str
    request: dict[str, Any]
    total_frames: int
    chunk_ranges: list[list[int]]
    completed_frames: int
    completed_chunks: int
    next_chunk_index: int
    active_chunk_index: int | None
    frame_files: list[str] = field(default_factory=list)
    frame_integrity: list[dict[str, Any]] = field(default_factory=list)
    seed_base: int = 0
    seed_plan: list[int] = field(default_factory=list)
    error: str = ""
    video_path: str | None = None
    poster_path: str | None = None
    gallery_id: int | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


_STATE_KEYS = frozenset(item.name for item in fields(AnimationProject))
_REQUEST_KEYS = frozenset(AnimateRequest.model_fields)


class AnimationStore:
    def __init__(self, state_root: Path, outputs_root: Path):
        self.state_root = Path(state_root).resolve()
        self.outputs_root = Path(outputs_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self._lock = _root_lock(self.state_root)
        self._transaction_state = threading.local()
        self._lock_path = self.state_root / _LOCK_FILE
        with self._lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())

    @contextmanager
    def _transaction(self):
        with self._lock:
            depth = getattr(self._transaction_state, "depth", 0)
            if depth:
                self._transaction_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._transaction_state.depth -= 1
                return
            handle = self._lock_path.open("r+b")
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                self._transaction_state.depth = 1
                try:
                    yield
                finally:
                    self._transaction_state.depth = 0
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _state_path(self, job_id: str) -> Path:
        job_id = _validate_job_id(job_id)
        return _assert_no_link_components(
            self.state_root, self.state_root / f"{job_id}.json"
        )

    def _project_dir(self, owner: str | None, job_id: str) -> Path:
        segment = _owner_segment(owner)
        job_id = _validate_job_id(job_id)
        return _beneath(
            self.outputs_root,
            self.outputs_root / segment / "animations" / job_id,
        )

    def create(
        self,
        request: AnimateRequest,
        *,
        owner: str | None,
        role: str,
        job_id: str,
    ) -> AnimationProject:
        with self._transaction():
            job_id = _validate_job_id(job_id)
            _owner_segment(owner)
            if not isinstance(role, str) or not role or len(role) > 64:
                raise ValueError("role must be non-empty text no longer than 64 characters")
            state_path = self._state_path(job_id)
            if state_path.exists():
                raise FileExistsError(f"animation project already exists: {job_id}")
            request_snapshot = request.model_dump()
            total_frames = request.total_frames
            seed_base = (
                secrets.randbits(32) if request.seed == -1 else int(request.seed)
            )
            timestamp = _now()
            project = AnimationProject(
                schema_version=SCHEMA_VERSION,
                revision=0,
                job_id=job_id,
                owner=owner,
                role=role,
                status="queued",
                request=request_snapshot,
                total_frames=total_frames,
                chunk_ranges=[
                    list(bounds)
                    for bounds in build_chunk_ranges(
                        total_frames, DEFAULT_CHUNK_SIZE, request.diffusion_cadence
                    )
                ],
                completed_frames=0,
                completed_chunks=0,
                next_chunk_index=0,
                active_chunk_index=None,
                seed_base=seed_base,
                seed_plan=build_seed_plan(
                    seed_base, request.seed_behavior, total_frames
                ),
                created_at=timestamp,
                updated_at=timestamp,
            )
            project_dir = self._project_dir(owner, job_id)
            parent = _mkdir_parents_no_links(self.outputs_root, project_dir.parent)
            if os.path.lexists(project_dir):
                raise FileExistsError(f"animation project already exists: {job_id}")
            temporary_dir = parent / f".creating-{job_id}-{uuid.uuid4().hex}"
            temporary_dir.mkdir()
            state_written = False
            layout_path = temporary_dir
            try:
                marker = temporary_dir / _CREATE_MARKER
                with marker.open("x", encoding="utf-8") as handle:
                    handle.write(job_id)
                    handle.flush()
                    os.fsync(handle.fileno())
                (temporary_dir / "frames").mkdir()
                (temporary_dir / "staging").mkdir()
                self._after_create_layout()
                os.rename(temporary_dir, project_dir)
                layout_path = project_dir
                marker = project_dir / _CREATE_MARKER
                self._after_create_rename()
                self._write(project)
                state_written = True
                try:
                    marker.unlink()
                except OSError:
                    pass
            except Exception:
                if not state_written:
                    try:
                        self._remove_without_following(layout_path)
                    except OSError:
                        pass
                raise
            return self._clone(project)

    @staticmethod
    def _after_create_layout() -> None:
        """Crash-test seam after temporary layout construction."""

    @staticmethod
    def _after_create_rename() -> None:
        """Crash-test seam after final-directory publication."""

    def load(self, job_id: str) -> AnimationProject:
        with self._transaction():
            return self._load_unlocked(job_id)

    def _load_unlocked(
        self, job_id: str, *, validate_media: bool = True
    ) -> AnimationProject:
        job_id = _validate_job_id(job_id)
        path = self._state_path(job_id)
        return self._read_project_file(
            path, expected_job_id=job_id, validate_media=validate_media
        )

    def _read_project_file(
        self,
        path: Path,
        *,
        expected_job_id: str,
        validate_media: bool,
    ) -> AnimationProject:
        _assert_no_link_components(self.state_root, path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(
                    handle,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
        except FileNotFoundError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AnimationStateError(
                f"animation state {expected_job_id} contains invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise AnimationStateError("corrupt animation state: expected an object")
        version = payload.get("schema_version")
        if type(version) is int and version > SCHEMA_VERSION:
            raise AnimationStateError(
                f"future schema version {version} is not supported"
            )
        if type(version) is int and version < SCHEMA_VERSION:
            raise AnimationStateError(
                f"older schema version {version} is not supported"
            )
        payload_keys = set(payload)
        missing = sorted(_STATE_KEYS - payload_keys)
        unexpected = sorted(payload_keys - _STATE_KEYS)
        if missing:
            raise AnimationStateError(
                "corrupt animation state: missing fields: " + ", ".join(missing)
            )
        if unexpected:
            raise AnimationStateError(
                "corrupt animation state: unexpected fields: "
                + ", ".join(unexpected)
            )
        try:
            project = AnimationProject(**payload)
        except (TypeError, ValueError) as exc:
            raise AnimationStateError(
                f"corrupt animation state {expected_job_id}: invalid fields"
            ) from exc
        self._validate_project(
            project,
            expected_job_id=expected_job_id,
            validate_media=validate_media,
        )
        return self._clone(project)

    def load_for_owner(
        self, job_id: str, *, username: str, is_admin: bool = False
    ) -> AnimationProject:
        try:
            project = self.load(job_id)
        except (FileNotFoundError, ValueError, AnimationStateError) as exc:
            raise FileNotFoundError("animation project not found") from exc
        if not is_admin and project.owner != username:
            raise FileNotFoundError("animation project not found")
        return project

    def project_dir(self, job_id: str) -> Path:
        project = self.load(job_id)
        return _assert_no_link_components(
            self.outputs_root, self._project_dir(project.owner, project.job_id)
        )

    def frame_paths(self, job_id: str, *, verify: bool = True) -> list[Path]:
        with self._transaction():
            project = self._load_unlocked(job_id)
            if verify:
                self._verify_frame_integrity_unlocked(project)
            root = self._project_dir(project.owner, project.job_id)
            return [
                _assert_no_link_components(root, root / relative)
                for relative in project.frame_files
            ]

    def stage_chunk(
        self, job_id: str, chunk_index: int, frame_bytes: list[bytes]
    ) -> list[str]:
        with self._transaction():
            project = self._load_unlocked(job_id)
            if project.active_chunk_index != chunk_index:
                raise ValueError("chunk does not match the active chunk")
            start, end = project.chunk_ranges[chunk_index]
            if len(frame_bytes) != end - start:
                raise ValueError("chunk returned an unexpected frame count")
            staging = _assert_no_link_components(
                self.outputs_root,
                self._project_dir(project.owner, project.job_id) / "staging",
            )
            names: list[str] = []
            for offset, raw in enumerate(frame_bytes):
                if not isinstance(raw, bytes) or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("animation frame must be PNG bytes")
                name = f"frame_{start + offset:06d}.png"
                temporary = staging / f".{name}.{uuid.uuid4().hex}.tmp"
                target = staging / name
                try:
                    with temporary.open("xb") as handle:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                names.append(name)
            return names

    def discard_staging(self, job_id: str) -> None:
        with self._transaction():
            project = self._load_unlocked(job_id, validate_media=False)
            staging = self._project_dir(project.owner, project.job_id) / "staging"
            if not staging.exists() or staging.is_symlink() or _is_reparse(staging):
                return
            for child in staging.iterdir():
                self._remove_without_following(child)

    def publish_result(
        self,
        job_id: str,
        *,
        video_path: str,
        poster_path: str,
        gallery_id: int,
    ) -> AnimationProject:
        with self._transaction():
            project = self._load_unlocked(job_id)
            if project.status == "done":
                return project
            if project.status != "finalizing":
                raise ValueError("animation is not finalizing")
            project.video_path = _relative_media_path(video_path, "video_path")
            project.poster_path = _relative_media_path(poster_path, "poster_path")
            project.gallery_id = int(gallery_id)
            project.status = "done"
            return self._persist_updated(project)

    def save(self, project: AnimationProject) -> None:
        with self._transaction():
            current = self._load_unlocked(project.job_id)
            if project.revision != current.revision:
                raise AnimationStateError("stale animation project revision")
            candidate = self._clone(project)
            self._validate_project(candidate)
            immutable = {
                "schema_version",
                "job_id",
                "owner",
                "role",
                "request",
                "total_frames",
                "chunk_ranges",
                "completed_frames",
                "completed_chunks",
                "next_chunk_index",
                "active_chunk_index",
                "frame_files",
                "frame_integrity",
                "seed_base",
                "seed_plan",
                "created_at",
            }
            if any(
                getattr(candidate, name) != getattr(current, name)
                for name in immutable
            ):
                raise AnimationStateError("save cannot alter immutable project state")
            if candidate.status != current.status and (
                current.status in TERMINAL_STATUSES
                or candidate.status not in _TRANSITIONS[current.status]
            ):
                raise AnimationStateError("save requested an invalid lifecycle transition")
            candidate.revision = current.revision + 1
            candidate.updated_at = _now()
            self._write(candidate)
            project.revision = candidate.revision
            project.updated_at = candidate.updated_at

    def begin_chunk(self, job_id: str, chunk_index: int) -> AnimationProject:
        with self._transaction():
            project = self._load_unlocked(job_id)
            if project.active_chunk_index is not None:
                raise ValueError("another chunk is already active")
            if chunk_index != project.next_chunk_index:
                raise ValueError("chunk_index must be the exact next chunk")
            if chunk_index >= len(project.chunk_ranges):
                raise ValueError("there is no next chunk")
            if project.status not in {"queued", "running"}:
                raise ValueError(f"cannot begin a chunk while status is {project.status}")
            project.active_chunk_index = chunk_index
            project.status = "running"
            project.error = ""
            return self._persist_updated(project)

    def commit_chunk(
        self, job_id: str, chunk_index: int, relative_frames: list[str]
    ) -> AnimationProject:
        with self._transaction():
            project = self._load_unlocked(job_id)
            if project.active_chunk_index != chunk_index:
                raise ValueError("chunk does not match the active chunk")
            start, end = project.chunk_ranges[chunk_index]
            expected_count = end - start
            if not isinstance(relative_frames, list) or len(relative_frames) != expected_count:
                raise ValueError(
                    f"chunk requires exactly {expected_count} frame files"
                )

            normalized: list[str] = []
            integrity: list[dict[str, Any]] = []
            project_dir = self._project_dir(project.owner, job_id)
            frames_root = _assert_no_link_components(
                self.outputs_root, project_dir / "frames"
            )
            staging_root = _assert_no_link_components(
                self.outputs_root, project_dir / "staging"
            )
            prepared: list[tuple[Path, Path, Path, dict[str, Any]]] = []
            for offset, value in enumerate(relative_frames):
                normalized_name = self._normalize_frame_name(value)
                canonical_name = f"frame_{start + offset:06d}.png"
                state_name = f"frames/{canonical_name}"
                if state_name in normalized or state_name in project.frame_files:
                    raise ValueError("duplicate frame file")
                match = _FRAME_INDEX_RE.fullmatch(normalized_name)
                if match is None or int(match.group(1)) != start + offset:
                    raise ValueError("frame files must be in contiguous global order")
                staged_source = staging_root / normalized_name
                if staged_source.is_symlink() or _is_reparse(staged_source):
                    raise ValueError("frame file must not be a symlink")
                source = _assert_no_link_components(staging_root, staged_source)
                if not source.exists():
                    raise FileNotFoundError(f"frame file does not exist: {normalized_name}")
                if not source.is_file():
                    raise ValueError("frame path must reference a regular file")
                temporary_fd, temporary_name = tempfile.mkstemp(
                    prefix=f".stage-{start + offset:06d}-",
                    suffix=".tmp",
                    dir=staging_root,
                )
                os.close(temporary_fd)
                temporary = Path(temporary_name)
                try:
                    digest, size = self._copy_verified_file(source, temporary)
                except BaseException:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
                normalized.append(state_name)
                metadata = {
                    "path": state_name,
                    "sha256": digest,
                    "size": size,
                }
                integrity.append(metadata)
                prepared.append(
                    (temporary, source, frames_root / canonical_name, metadata)
                )

            journal = self.state_root / (
                f".frame-{project.job_id}-{chunk_index}-{uuid.uuid4().hex}.journal.json"
            )
            journal_payload = {
                "version": 1,
                "job_id": project.job_id,
                "chunk_index": chunk_index,
                "expected_revision": project.revision,
                "entries": [
                    {
                        "staging": source.name,
                        "canonical": metadata["path"],
                        "sha256": metadata["sha256"],
                        "size": metadata["size"],
                    }
                    for _, source, _, metadata in prepared
                ],
            }
            self._atomic_json_write(journal, journal_payload)
            published: list[tuple[Path, Path]] = []
            try:
                self._before_frame_publish()
                for temporary, source, target, _ in prepared:
                    if target.is_symlink() or _is_reparse(target):
                        raise ValueError("canonical frame target is a link")
                    os.replace(temporary, target)
                    published.append((target, staging_root / target.name))

                self._before_state_publish()
                project.frame_files.extend(normalized)
                project.frame_integrity.extend(integrity)
                project.completed_frames += expected_count
                project.completed_chunks += 1
                project.next_chunk_index += 1
                project.active_chunk_index = None
                project.status = (
                    "finalizing"
                    if project.completed_frames == project.total_frames
                    else "queued"
                )
                result = self._persist_updated(project)
            except Exception:
                for target, rollback in reversed(published):
                    try:
                        if os.path.lexists(target):
                            os.replace(target, rollback)
                    except OSError:
                        pass
                try:
                    journal.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            finally:
                for temporary, _, _, _ in prepared:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
            for _, source, _, _ in prepared:
                if os.path.lexists(source):
                    try:
                        self._remove_without_following(source)
                    except OSError:
                        pass
            try:
                journal.unlink(missing_ok=True)
            except OSError:
                pass
            return result

    @staticmethod
    def _before_frame_publish() -> None:
        """Test seam after stable copies, before atomic canonical publication."""

    @staticmethod
    def _before_state_publish() -> None:
        """Crash-test seam after frames, before state CAS."""

    def _copy_verified_file(
        self, source: Path, destination: Path
    ) -> tuple[str, int]:
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode) or source.is_symlink() or _is_reparse(source):
            raise ValueError("frame source must be a regular non-link file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("frame source changed while opening")
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(descriptor, "rb", closefd=False) as reader:
                with destination.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    writer.flush()
                    os.fsync(writer.fileno())
            self._after_frame_copy(source)
            after = source.lstat()
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if after_identity != before_identity:
                raise ValueError("frame source changed while copying")
            return digest.hexdigest(), size
        finally:
            os.close(descriptor)

    @staticmethod
    def _after_frame_copy(source: Path) -> None:
        """Test seam before source stability recheck."""

    def mark_status(
        self, job_id: str, status: str, error: str = ""
    ) -> AnimationProject:
        with self._transaction():
            project = self._load_unlocked(job_id)
            if status not in STATUSES:
                raise ValueError(f"unknown animation status: {status}")
            if status == project.status:
                return project
            if project.status in TERMINAL_STATUSES:
                raise ValueError(f"status {project.status} is terminal")
            if status not in _TRANSITIONS[project.status]:
                raise ValueError(
                    f"invalid status transition: {project.status} -> {status}"
                )
            project.status = status
            project.error = self._clean_error(error) if status == "error" else ""
            if status in {"queued", "error", "blocked", "cancelled"}:
                project.active_chunk_index = None
            return self._persist_updated(project)

    def prepare_recovery(self, job_id: str) -> AnimationProject:
        with self._transaction():
            project = self._load_unlocked(job_id)
            self._verify_frame_integrity_unlocked(project)
            if project.status not in RECOVERABLE_STATUSES | {"error"}:
                raise ValueError(f"status {project.status} cannot be recovered")
            project.active_chunk_index = None
            project.error = ""
            project.status = (
                "finalizing"
                if project.completed_frames == project.total_frames
                else "queued"
            )
            return self._persist_updated(project)

    def recoverable(self) -> list[AnimationProject]:
        with self._transaction():
            projects: list[AnimationProject] = []
            for path in sorted(self.state_root.glob("*.json"), key=lambda item: item.name):
                job_id = path.stem
                try:
                    project = self._load_unlocked(job_id)
                    if project.status in RECOVERABLE_STATUSES:
                        self._verify_frame_integrity_unlocked(project)
                        projects.append(project)
                except (OSError, ValueError, AnimationStateError):
                    continue
            return projects

    def active_for_owner(self, owner: str | None) -> list[AnimationProject]:
        with self._transaction():
            projects: list[AnimationProject] = []
            for path in sorted(self.state_root.glob("*.json")):
                try:
                    project = self._load_unlocked(path.stem)
                except (OSError, ValueError, AnimationStateError):
                    continue
                if project.owner == owner and project.status not in TERMINAL_STATUSES:
                    projects.append(project)
            return projects

    def verify_frame_integrity(self, job_id: str) -> bool:
        with self._transaction():
            project = self._load_unlocked(job_id)
            self._verify_frame_integrity_unlocked(project)
            return True

    def _verify_frame_integrity_unlocked(
        self, project: AnimationProject
    ) -> None:
        project_dir = self._project_dir(project.owner, project.job_id)
        frames = _assert_no_link_components(
            self.outputs_root, project_dir / "frames"
        )
        for metadata in project.frame_integrity:
            frame = _assert_no_link_components(
                frames, frames / Path(metadata["path"]).name
            )
            digest = hashlib.sha256()
            size = 0
            with frame.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            if size != metadata["size"] or digest.hexdigest() != metadata["sha256"]:
                raise AnimationStateError(
                    f"frame integrity check failed: {metadata['path']}"
                )

    def reconcile_staging(self) -> list[str]:
        with self._transaction():
            removed: list[str] = []
            removed.extend(self._reconcile_frame_journals())
            for tombstone in sorted(self.state_root.glob(".deleted-*.json")):
                match = re.fullmatch(
                    r"\.deleted-([a-f0-9-]+)-[a-f0-9]+\.json",
                    tombstone.name,
                )
                if match is None:
                    continue
                try:
                    project = self._read_project_file(
                        tombstone,
                        expected_job_id=match.group(1),
                        validate_media=False,
                    )
                    removed.extend(self._cleanup_deleted(project, tombstone))
                except (OSError, ValueError, AnimationStateError):
                    continue

            for path in self.state_root.iterdir():
                if path.name.endswith(".tmp") and (
                    path.is_symlink() or self._is_reparse(path) or path.is_file()
                ):
                    try:
                        self._remove_without_following(path)
                        removed.append(f"state/{path.name}")
                    except OSError:
                        pass

            removed.extend(self._reconcile_creation_markers())

            for state_path in sorted(self.state_root.glob("*.json")):
                try:
                    project = self._load_unlocked(state_path.stem)
                except (FileNotFoundError, ValueError, AnimationStateError):
                    continue
                staging = self._project_dir(project.owner, project.job_id) / "staging"
                if (
                    staging.is_symlink()
                    or self._is_reparse(staging)
                    or not staging.exists()
                ):
                    continue
                for child in list(staging.iterdir()):
                    if self._is_retryable_staging_frame(project, child):
                        continue
                    try:
                        relatives = self._listed_tree(child, staging)
                        self._remove_without_following(child)
                    except OSError:
                        continue
                    for relative in relatives:
                        removed.append(
                            f"outputs/{_owner_segment(project.owner)}/animations/"
                            f"{project.job_id}/staging/{relative}"
                        )
            return sorted(set(removed))

    def _reconcile_frame_journals(self) -> list[str]:
        removed: list[str] = []
        for journal in sorted(self.state_root.glob(".frame-*.journal.json")):
            if journal.is_symlink() or _is_reparse(journal):
                continue
            try:
                payload = self._read_frame_journal(journal)
                job_id = payload["job_id"]
                project = self._load_unlocked(job_id, validate_media=False)
                project_dir = self._project_dir(project.owner, job_id)
                frames = _assert_no_link_components(
                    self.outputs_root, project_dir / "frames"
                )
                staging = _assert_no_link_components(
                    self.outputs_root, project_dir / "staging"
                )
                committed = (
                    project.revision > payload["expected_revision"]
                    and project.completed_chunks > payload["chunk_index"]
                )
                chunk_index = payload["chunk_index"]
                if not 0 <= chunk_index < len(project.chunk_ranges):
                    raise AnimationStateError("invalid frame journal chunk")
                start, end = project.chunk_ranges[chunk_index]
                if len(payload["entries"]) != end - start:
                    raise AnimationStateError("invalid frame journal frame count")
                for offset, entry in enumerate(payload["entries"]):
                    expected_name = f"frame_{start + offset:06d}.png"
                    if (
                        not isinstance(entry, dict)
                        or set(entry)
                        != {"staging", "canonical", "sha256", "size"}
                        or not isinstance(entry.get("sha256"), str)
                        or re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is None
                        or type(entry.get("size")) is not int
                        or entry["size"] < 0
                    ):
                        raise AnimationStateError("invalid frame journal entry")
                    staging_name = self._normalize_frame_name(entry["staging"])
                    canonical = _relative_media_path(
                        entry["canonical"], "journal canonical"
                    )
                    if (
                        canonical != f"frames/{expected_name}"
                        or _FRAME_INDEX_RE.fullmatch(staging_name) is None
                        or int(_FRAME_INDEX_RE.fullmatch(staging_name).group(1))
                        != start + offset
                    ):
                        raise AnimationStateError("invalid frame journal path")
                    canonical_path = frames / Path(canonical).name
                    staging_path = staging / staging_name
                    cleanup = staging_path if committed else canonical_path
                    if os.path.lexists(cleanup):
                        self._remove_without_following(cleanup)
                        removed.append(
                            f"outputs/{_owner_segment(project.owner)}/animations/"
                            f"{job_id}/{cleanup.relative_to(project_dir).as_posix()}"
                        )
                journal.unlink()
                removed.append(f"state/{journal.name}")
            except (OSError, ValueError, AnimationStateError):
                continue
        return removed

    def _read_frame_journal(self, journal: Path) -> dict[str, Any]:
        _assert_no_link_components(self.state_root, journal)
        with journal.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "version",
                "job_id",
                "chunk_index",
                "expected_revision",
                "entries",
            }
            or payload.get("version") != 1
            or type(payload.get("chunk_index")) is not int
            or payload["chunk_index"] < 0
            or type(payload.get("expected_revision")) is not int
            or payload["expected_revision"] < 0
            or not isinstance(payload.get("entries"), list)
        ):
            raise AnimationStateError("invalid frame journal")
        job_id = _validate_job_id(payload.get("job_id"))
        if payload["job_id"] != job_id:
            raise AnimationStateError("non-canonical frame journal job_id")
        for entry in payload["entries"]:
            if (
                not isinstance(entry, dict)
                or set(entry)
                != {"staging", "canonical", "sha256", "size"}
                or not isinstance(entry.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is None
                or type(entry.get("size")) is not int
                or entry["size"] < 0
            ):
                raise AnimationStateError("invalid frame journal entry")
            self._normalize_frame_name(entry.get("staging"))
            canonical = _relative_media_path(
                entry.get("canonical"), "journal canonical"
            )
            if canonical is None or not canonical.startswith("frames/"):
                raise AnimationStateError("invalid frame journal path")
        return payload

    @staticmethod
    def _is_retryable_staging_frame(
        project: AnimationProject, path: Path
    ) -> bool:
        if path.is_symlink() or _is_reparse(path) or not path.is_file():
            return False
        match = _FRAME_INDEX_RE.fullmatch(path.name)
        if match is None or project.next_chunk_index >= len(project.chunk_ranges):
            return False
        start, end = project.chunk_ranges[project.next_chunk_index]
        return start <= int(match.group(1)) < end

    def delete(
        self, job_id: str, *, username: str, is_admin: bool = False
    ) -> bool:
        with self._transaction():
            job_id = _validate_job_id(job_id)
            try:
                project = self._load_unlocked(job_id, validate_media=False)
            except FileNotFoundError:
                tombstones = sorted(
                    self.state_root.glob(f".deleted-{job_id}-*.json")
                )
                for tombstone in tombstones:
                    try:
                        deleted = self._read_project_file(
                            tombstone,
                            expected_job_id=job_id,
                            validate_media=False,
                        )
                    except (OSError, ValueError, AnimationStateError):
                        continue
                    if not is_admin and deleted.owner != username:
                        return False
                    try:
                        self._cleanup_deleted(deleted, tombstone)
                    except OSError:
                        pass
                    return True
                return False
            if not is_admin and project.owner != username:
                return False

            state_path = self._state_path(job_id)
            project_dir = self._project_dir(project.owner, job_id)
            parent = _assert_no_link_components(
                self.outputs_root, project_dir.parent
            )
            suffix = uuid.uuid4().hex
            trash = _beneath(parent, parent / f".trash-{job_id}-{suffix}")
            tombstone = _beneath(
                self.state_root,
                self.state_root / f".deleted-{job_id}-{suffix}.json",
            )
            os.replace(state_path, tombstone)
            _fsync_directory(self.state_root)
            try:
                if os.path.lexists(project_dir):
                    _assert_no_link_components(
                        self.outputs_root, project_dir, allow_final=True
                    )
                    os.replace(project_dir, trash)
                self._cleanup_deleted(project, tombstone)
            except OSError:
                pass
            return True

    def _cleanup_deleted(
        self, project: AnimationProject, tombstone: Path
    ) -> list[str]:
        removed: list[str] = []
        parent = _assert_no_link_components(
            self.outputs_root,
            self._project_dir(project.owner, project.job_id).parent,
        )
        original = self._project_dir(project.owner, project.job_id)
        if os.path.lexists(original):
            _assert_no_link_components(
                self.outputs_root, original, allow_final=True
            )
            retry_trash = _beneath(
                parent,
                parent / f".trash-{project.job_id}-{uuid.uuid4().hex}",
            )
            os.replace(original, retry_trash)
        for trash in sorted(parent.glob(f".trash-{project.job_id}-*")):
            _assert_no_link_components(parent, trash, allow_final=True)
            self._remove_without_following(trash)
            removed.append(
                f"outputs/{_owner_segment(project.owner)}/animations/{trash.name}"
            )
        for journal in sorted(self.state_root.glob(".frame-*.journal.json")):
            try:
                payload = self._read_frame_journal(journal)
            except (OSError, ValueError, AnimationStateError):
                continue
            if payload["job_id"] != project.job_id:
                continue
            self._remove_without_following(journal)
            removed.append(f"state/{journal.name}")
        tombstone.unlink(missing_ok=True)
        removed.append(f"state/{tombstone.name}")
        return removed

    def _reconcile_creation_markers(self) -> list[str]:
        removed: list[str] = []
        for owner_dir in list(self.outputs_root.iterdir()):
            if owner_dir.is_symlink() or _is_reparse(owner_dir) or not owner_dir.is_dir():
                continue
            animations = owner_dir / "animations"
            if (
                animations.is_symlink()
                or _is_reparse(animations)
                or not animations.is_dir()
            ):
                continue
            for project_dir in list(animations.iterdir()):
                if (
                    project_dir.name.startswith(".creating-")
                    and not project_dir.is_symlink()
                    and not _is_reparse(project_dir)
                    and project_dir.is_dir()
                ):
                    self._remove_without_following(project_dir)
                    removed.append(
                        f"outputs/{owner_dir.name}/animations/{project_dir.name}"
                    )
                    continue
                if (
                    project_dir.is_symlink()
                    or _is_reparse(project_dir)
                    or not project_dir.is_dir()
                ):
                    continue
                marker = project_dir / _CREATE_MARKER
                if (
                    marker.is_symlink()
                    or _is_reparse(marker)
                    or not marker.is_file()
                ):
                    continue
                try:
                    job_id = _validate_job_id(project_dir.name)
                except ValueError:
                    continue
                state_path = self._state_path(job_id)
                valid_state = False
                if state_path.is_file():
                    try:
                        self._load_unlocked(job_id, validate_media=False)
                        valid_state = True
                    except (OSError, ValueError, AnimationStateError):
                        pass
                if valid_state:
                    marker.unlink()
                    removed.append(
                        f"outputs/{owner_dir.name}/animations/{job_id}/{_CREATE_MARKER}"
                    )
                else:
                    self._remove_without_following(project_dir)
                    removed.append(
                        f"outputs/{owner_dir.name}/animations/{job_id}/{_CREATE_MARKER}"
                    )
        return removed

    def _persist_updated(self, project: AnimationProject) -> AnimationProject:
        expected_revision = project.revision
        disk = self._load_unlocked(project.job_id)
        if disk.revision != expected_revision:
            raise AnimationStateError("stale animation project revision")
        candidate = self._clone(project)
        candidate.revision = expected_revision + 1
        candidate.updated_at = _now()
        self._validate_project(candidate)
        self._write(candidate)
        return candidate

    def _write(self, project: AnimationProject) -> None:
        self._validate_project(project)
        destination = self._state_path(project.job_id)
        self._atomic_json_write(destination, project.to_dict())

    def _atomic_json_write(self, destination: Path, payload: object) -> None:
        destination = _beneath(self.state_root, destination)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_root,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
            _fsync_directory(self.state_root)
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass

    def _validate_project(
        self,
        project: AnimationProject,
        expected_job_id: str | None = None,
        *,
        validate_media: bool = True,
    ) -> None:
        if not isinstance(project, AnimationProject):
            raise AnimationStateError("invalid animation project object")
        if type(project.schema_version) is not int:
            raise AnimationStateError("corrupt schema_version: expected integer")
        if project.schema_version > SCHEMA_VERSION:
            raise AnimationStateError(
                f"future schema version {project.schema_version} is not supported"
            )
        if project.schema_version != SCHEMA_VERSION:
            raise AnimationStateError(
                f"unsupported schema version {project.schema_version}"
            )
        if type(project.revision) is not int or project.revision < 0:
            raise AnimationStateError("corrupt revision")
        try:
            canonical_job_id = _validate_job_id(project.job_id)
            _owner_segment(project.owner)
        except ValueError as exc:
            raise AnimationStateError(f"corrupt animation identity: {exc}") from exc
        if project.job_id != canonical_job_id:
            raise AnimationStateError("corrupt animation identity: job_id is not lowercase")
        if expected_job_id is not None and project.job_id != expected_job_id:
            raise AnimationStateError("corrupt animation state: job_id does not match file")
        if not isinstance(project.role, str) or not project.role or len(project.role) > 64:
            raise AnimationStateError("corrupt role")
        if not isinstance(project.status, str) or project.status not in STATUSES:
            raise AnimationStateError("corrupt status")
        if not isinstance(project.request, dict):
            raise AnimationStateError("corrupt request snapshot")
        if set(project.request) != _REQUEST_KEYS:
            raise AnimationStateError(
                "corrupt request snapshot: fields do not match current schema"
            )
        try:
            request = AnimateRequest.model_validate(
                copy.deepcopy(project.request), strict=True
            )
        except Exception as exc:
            raise AnimationStateError("corrupt request snapshot: validation failed") from exc
        mismatch = _canonical_mismatch(
            project.request, request.model_dump(mode="python")
        )
        if mismatch is not None:
            raise AnimationStateError(
                "corrupt request snapshot: non-canonical data: "
                + mismatch[:256]
            )
        if type(project.total_frames) is not int or project.total_frames != request.total_frames:
            raise AnimationStateError("corrupt total_frames")
        expected_ranges = [
            list(bounds)
            for bounds in build_chunk_ranges(
                project.total_frames, DEFAULT_CHUNK_SIZE, request.diffusion_cadence
            )
        ]
        if project.chunk_ranges != expected_ranges or any(
            type(value) is not int for bounds in project.chunk_ranges for value in bounds
        ):
            raise AnimationStateError("corrupt chunk_ranges")
        integers = (
            project.completed_frames,
            project.completed_chunks,
            project.next_chunk_index,
            project.seed_base,
        )
        if any(type(value) is not int for value in integers):
            raise AnimationStateError("corrupt animation counters")
        if not 0 <= project.completed_frames <= project.total_frames:
            raise AnimationStateError("corrupt completed_frames")
        if not 0 <= project.completed_chunks <= len(project.chunk_ranges):
            raise AnimationStateError("corrupt completed_chunks")
        if (
            project.next_chunk_index != project.completed_chunks
            or project.completed_frames
            != (
                project.chunk_ranges[project.completed_chunks - 1][1]
                if project.completed_chunks
                else 0
            )
        ):
            raise AnimationStateError("corrupt chunk progress counters")
        if project.active_chunk_index is not None and (
            type(project.active_chunk_index) is not int
            or project.active_chunk_index != project.next_chunk_index
            or project.active_chunk_index >= len(project.chunk_ranges)
        ):
            raise AnimationStateError("corrupt active_chunk_index")
        if project.active_chunk_index is not None and project.status != "running":
            raise AnimationStateError("corrupt status for active chunk")
        complete = project.completed_frames == project.total_frames
        if (project.status in {"finalizing", "done"} and not complete) or (
            project.status in {"queued", "running"} and complete
        ):
            raise AnimationStateError("corrupt status for frame progress")
        if not isinstance(project.frame_files, list) or len(project.frame_files) != project.completed_frames:
            raise AnimationStateError("corrupt frame_files count")
        if len(set(project.frame_files)) != len(project.frame_files):
            raise AnimationStateError("corrupt duplicate frame_files")
        if (
            not isinstance(project.frame_integrity, list)
            or len(project.frame_integrity) != len(project.frame_files)
        ):
            raise AnimationStateError("corrupt frame integrity metadata")
        for path, metadata in zip(project.frame_files, project.frame_integrity):
            if (
                not isinstance(metadata, dict)
                or set(metadata) != {"path", "sha256", "size"}
                or metadata.get("path") != path
                or not isinstance(metadata.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", metadata["sha256"]) is None
                or type(metadata.get("size")) is not int
                or metadata["size"] < 0
            ):
                raise AnimationStateError("corrupt frame integrity metadata")
        project_dir = self._project_dir(project.owner, project.job_id)
        frames_root = project_dir / "frames"
        if validate_media:
            try:
                _assert_no_link_components(self.outputs_root, project_dir)
                frames_root = _assert_no_link_components(
                    self.outputs_root, frames_root
                )
            except ValueError as exc:
                raise AnimationStateError("corrupt project media path") from exc
        for index, value in enumerate(project.frame_files):
            normalized = _relative_media_path(value, "frame_files")
            if (
                normalized is None
                or not normalized.startswith("frames/")
                or "/" in normalized.removeprefix("frames/")
            ):
                raise AnimationStateError("corrupt frame_files path")
            match = _FRAME_INDEX_RE.fullmatch(Path(normalized).name)
            if match is None or int(match.group(1)) != index:
                raise AnimationStateError("corrupt frame_files ordering")
            if validate_media:
                frame = frames_root / Path(normalized).name
                if frame.is_symlink():
                    raise AnimationStateError("corrupt frame file: symlink is not allowed")
                try:
                    resolved_frame = _assert_no_link_components(frames_root, frame)
                except ValueError as exc:
                    raise AnimationStateError("corrupt frame file path") from exc
                if not resolved_frame.is_file():
                    raise AnimationStateError(
                        "corrupt frame file: committed file is missing"
                    )
        if type(project.seed_base) is not int or not 0 <= project.seed_base < (1 << 64):
            raise AnimationStateError("corrupt seed_base")
        if (
            not isinstance(project.seed_plan, list)
            or any(type(value) is not int for value in project.seed_plan)
            or project.seed_plan
            != build_seed_plan(
                project.seed_base, request.seed_behavior, project.total_frames
            )
        ):
            raise AnimationStateError("corrupt seed_plan")
        if not isinstance(project.error, str) or len(project.error) > _MAX_ERROR_LENGTH:
            raise AnimationStateError("corrupt error text")
        for field_name in ("video_path", "poster_path"):
            media_path = _relative_media_path(
                getattr(project, field_name), field_name
            )
            if media_path is not None and validate_media:
                try:
                    _assert_no_link_components(project_dir, project_dir / media_path)
                except ValueError as exc:
                    raise AnimationStateError(
                        f"corrupt {field_name}: path escapes project"
                    ) from exc
        if project.gallery_id is not None and (
            type(project.gallery_id) is not int or project.gallery_id < 0
        ):
            raise AnimationStateError("corrupt gallery_id")
        if not isinstance(project.created_at, str) or not isinstance(
            project.updated_at, str
        ):
            raise AnimationStateError("corrupt timestamps")
        try:
            created = datetime.fromisoformat(project.created_at)
            updated = datetime.fromisoformat(project.updated_at)
        except ValueError as exc:
            raise AnimationStateError("corrupt timestamps") from exc
        if (
            created.tzinfo is None
            or updated.tzinfo is None
            or created.utcoffset() != timedelta(0)
            or updated.utcoffset() != timedelta(0)
            or updated < created
        ):
            raise AnimationStateError(
                "corrupt timestamps: expected ordered timezone-aware UTC values"
            )
        self._state_path(project.job_id)
        if validate_media:
            try:
                _assert_no_link_components(self.outputs_root, project_dir / "frames")
                _assert_no_link_components(self.outputs_root, project_dir / "staging")
            except ValueError as exc:
                raise AnimationStateError("corrupt project media path") from exc

    @staticmethod
    def _normalize_frame_name(value: object) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("frame name must be non-empty text")
        value = value.replace("\\", "/")
        if value.startswith("frames/"):
            value = value.removeprefix("frames/")
        path = Path(value)
        if path.is_absolute() or path.anchor or len(path.parts) != 1 or value in {".", ".."}:
            raise ValueError("frame name must be a safe file relative to frames")
        return value

    @staticmethod
    def _clean_error(error: object) -> str:
        if not isinstance(error, str):
            raise ValueError("error must be text")
        return "".join(
            character
            for character in error.strip()
            if character in "\n\t" or ord(character) >= 32
        )[:_MAX_ERROR_LENGTH]

    @staticmethod
    def _clone(project: AnimationProject) -> AnimationProject:
        return AnimationProject(**project.to_dict())

    @staticmethod
    def _is_reparse(path: Path) -> bool:
        return _is_reparse(path)

    @classmethod
    def _remove_without_following(cls, path: Path) -> None:
        if path.is_symlink():
            path.unlink(missing_ok=True)
            return
        if cls._is_reparse(path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                path.rmdir()
            return
        if not path.is_dir():
            path.unlink(missing_ok=True)
            return
        for child in path.iterdir():
            cls._remove_without_following(child)
        path.rmdir()

    @classmethod
    def _listed_tree(cls, path: Path, root: Path) -> list[str]:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or cls._is_reparse(path) or not path.is_dir():
            return [relative]
        result = [relative]
        for child in path.iterdir():
            result.extend(cls._listed_tree(child, root))
        return result
