from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

ALLOWED_VIDEO_TYPES = frozenset(
    {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}
)
_UPLOAD_ID = re.compile(r"[a-f0-9]{32}")
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class UploadQuotaError(ValueError):
    pass


class AnimationUploadStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        max_frames: int,
        max_dimension: int,
        max_duration: float = 60.0,
        ttl_seconds: int,
        max_user_uploads: int = 3,
        max_user_bytes: int = 512 * 1024 * 1024,
        max_global_uploads: int = 32,
        max_global_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.max_frames = int(max_frames)
        self.max_dimension = int(max_dimension)
        self.max_duration = float(max_duration)
        self.ttl_seconds = int(ttl_seconds)
        self.max_user_uploads = int(max_user_uploads)
        self.max_user_bytes = int(max_user_bytes)
        self.max_global_uploads = int(max_global_uploads)
        self.max_global_bytes = int(max_global_bytes)
        self._lock_path = self.root / ".upload-store.lock"
        with self._lock_path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
        key = os.path.normcase(str(self.root))
        with _LOCKS_GUARD:
            self._thread_lock = _LOCKS.setdefault(key, threading.RLock())

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            handle = self._lock_path.open("r+b")
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _reservation_path(self, upload_id: str) -> Path:
        self._validate_id(upload_id)
        return self.root / f".reserve-{upload_id}.json"

    def reserve(
        self,
        owner: str | None,
        declared_size: int,
        active_upload_ids: set[str] | None = None,
    ) -> tuple[str, Path]:
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or declared_size < 1
            or declared_size > self.max_bytes
        ):
            raise UploadQuotaError("upload size exceeds the per-file quota")
        with self._locked():
            self._cleanup_unlocked(active_upload_ids)
            records = self._quota_records_unlocked()
            owner_records = [item for item in records if item["owner"] == owner]
            if len(owner_records) + 1 > self.max_user_uploads:
                raise UploadQuotaError("per-user upload count quota reached")
            if sum(item["size"] for item in owner_records) + declared_size > self.max_user_bytes:
                raise UploadQuotaError("per-user upload byte quota reached")
            if len(records) + 1 > self.max_global_uploads:
                raise UploadQuotaError("global upload count quota reached")
            if sum(item["size"] for item in records) + declared_size > self.max_global_bytes:
                raise UploadQuotaError("global upload byte quota reached")
            upload_id = uuid.uuid4().hex
            descriptor, name = tempfile.mkstemp(
                prefix=f".{upload_id}.", suffix=".upload", dir=self.root
            )
            os.close(descriptor)
            temporary = Path(name)
            payload = {
                "upload_id": upload_id,
                "owner": owner,
                "created_at": time.time(),
                "size": declared_size,
                "temporary": temporary.name,
            }
            try:
                self._atomic_json(
                    self._reservation_path(upload_id), payload
                )
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            return upload_id, temporary

    def temporary(self) -> tuple[str, Path]:
        upload_id = uuid.uuid4().hex
        descriptor, name = tempfile.mkstemp(
            prefix=f".{upload_id}.", suffix=".upload", dir=self.root
        )
        os.close(descriptor)
        return upload_id, Path(name)

    def reservation(self, upload_id: str) -> dict | None:
        try:
            payload = json.loads(
                self._reservation_path(upload_id).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("upload_id") != upload_id
            or Path(str(payload.get("temporary", ""))).name
            != payload.get("temporary")
        ):
            return None
        return payload

    def abort(
        self, upload_id: str, *, username: str | None, is_admin: bool = False
    ) -> bool:
        with self._locked():
            reservation = self.reservation(upload_id)
            if reservation is None:
                return False
            if not is_admin and reservation.get("owner") != username:
                return False
            (self.root / reservation["temporary"]).unlink(missing_ok=True)
            self._reservation_path(upload_id).unlink(missing_ok=True)
            return True

    def finalize(
        self,
        upload_id: str,
        temporary: Path,
        *,
        owner: str | None,
        content_type: str,
        size: int,
        sha256: str,
    ) -> dict:
        with self._locked():
            self._validate_id(upload_id)
            reservation = self.reservation(upload_id)
            if reservation is not None and (
                reservation.get("owner") != owner
                or reservation.get("temporary") != temporary.name
                or size > int(reservation.get("size", 0))
            ):
                raise ValueError("upload reservation does not match")
            if content_type not in ALLOWED_VIDEO_TYPES:
                raise ValueError("unsupported video content type")
            if size < 1 or size > self.max_bytes:
                raise ValueError("video upload exceeds the byte limit")
            metadata = self.validate_video(temporary)
            extension = {
                "video/mp4": ".mp4",
                "video/quicktime": ".mov",
                "video/webm": ".webm",
                "video/x-matroska": ".mkv",
            }[content_type]
            destination = self.root / f"{upload_id}{extension}"
            payload = {
                "upload_id": upload_id,
                "owner": owner,
                "created_at": time.time(),
                "size": size,
                "sha256": sha256,
                "content_type": content_type,
                "filename": destination.name,
                **metadata,
            }
            meta_path = self.root / f"{upload_id}.json"
            try:
                os.replace(temporary, destination)
                self._atomic_json(meta_path, payload)
                self._reservation_path(upload_id).unlink(missing_ok=True)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            return dict(payload)

    def validate_video(self, path: Path) -> dict:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("video is corrupt or unsupported")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (
                frame_count < 1
                or frame_count > self.max_frames
                or not math.isfinite(fps)
                or fps <= 0
                or width < 1
                or height < 1
                or max(width, height) > self.max_dimension
                or frame_count / fps > self.max_duration
            ):
                raise ValueError("video metadata exceeds configured limits")
            ok, first = capture.read()
            if not ok or first is None:
                raise ValueError("video first frame is not decodable")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ok, last = capture.read()
            if not ok or last is None:
                raise ValueError("video last frame is not decodable")
            if first.shape[:2] != (height, width) or last.shape[:2] != (height, width):
                raise ValueError("video dimensions are inconsistent")
            return {
                "frame_count": frame_count,
                "fps": fps,
                "duration": frame_count / fps,
                "width": width,
                "height": height,
            }
        finally:
            capture.release()

    def sample_images(self, path: Path) -> list[Image.Image]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("video is corrupt or unsupported")
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            result = []
            for index in sorted({0, max(0, count - 1)}):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError("video sample frame is not decodable")
                result.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return result
        finally:
            capture.release()

    def resolve(
        self, upload_id: str, *, username: str | None, is_admin: bool = False
    ) -> Path:
        metadata = self.metadata(upload_id)
        if not is_admin and metadata.get("owner") != username:
            raise FileNotFoundError("animation upload not found")
        path = (self.root / metadata["filename"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise FileNotFoundError("animation upload not found") from exc
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("animation upload not found")
        return path

    def metadata(self, upload_id: str) -> dict:
        self._validate_id(upload_id)
        path = self.root / f"{upload_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise FileNotFoundError("animation upload not found") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("upload_id") != upload_id
            or Path(str(payload.get("filename", ""))).name != payload.get("filename")
        ):
            raise FileNotFoundError("animation upload not found")
        return payload

    def delete(
        self, upload_id: str, *, username: str | None, is_admin: bool = False
    ) -> bool:
        with self._locked():
            try:
                path = self.resolve(upload_id, username=username, is_admin=is_admin)
            except FileNotFoundError:
                return False
            path.unlink(missing_ok=True)
            (self.root / f"{upload_id}.json").unlink(missing_ok=True)
            return True

    def cleanup(self, active_upload_ids: set[str] | None = None) -> list[str]:
        with self._locked():
            return self._cleanup_unlocked(active_upload_ids)

    def _cleanup_unlocked(
        self, active_upload_ids: set[str] | None = None
    ) -> list[str]:
        active = active_upload_ids or set()
        removed: list[str] = []
        cutoff = time.time() - self.ttl_seconds
        for reservation_path in self.root.glob(".reserve-*.json"):
            match = re.fullmatch(r"\.reserve-([a-f0-9]{32})\.json", reservation_path.name)
            if match is None:
                continue
            reservation = self.reservation(match.group(1))
            if reservation is None or float(reservation.get("created_at", 0)) < cutoff:
                if reservation:
                    (self.root / reservation["temporary"]).unlink(missing_ok=True)
                reservation_path.unlink(missing_ok=True)
                removed.append(match.group(1))
        for metadata_path in self.root.glob("*.json"):
            upload_id = metadata_path.stem
            if upload_id in active:
                continue
            try:
                metadata = self.metadata(upload_id)
                if float(metadata["created_at"]) >= cutoff:
                    continue
                media = self.root / metadata["filename"]
                media.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                removed.append(upload_id)
            except (OSError, ValueError, FileNotFoundError, KeyError):
                continue
        for orphan in self.root.iterdir():
            if orphan.stat().st_mtime >= cutoff:
                continue
            if orphan == self._lock_path or orphan.name.startswith(".reserve-"):
                continue
            if orphan.name.startswith("."):
                orphan.unlink(missing_ok=True)
                continue
            if orphan.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
                upload_id = orphan.stem
                if upload_id not in active and not (
                    self.root / f"{upload_id}.json"
                ).exists():
                    orphan.unlink(missing_ok=True)
                    removed.append(upload_id)
        return removed

    def _quota_records_unlocked(self) -> list[dict]:
        records: list[dict] = []
        for path in self.root.glob("*.json"):
            if path.name.startswith(".reserve-"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(payload, dict) and type(payload.get("size")) is int:
                records.append(payload)
        for path in self.root.glob(".reserve-*.json"):
            match = re.fullmatch(r"\.reserve-([a-f0-9]{32})\.json", path.name)
            reservation = self.reservation(match.group(1)) if match else None
            if reservation and type(reservation.get("size")) is int:
                records.append(reservation)
        return records

    @staticmethod
    def _atomic_json(destination: Path, payload: dict) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_id(upload_id: str) -> None:
        if not isinstance(upload_id, str) or _UPLOAD_ID.fullmatch(upload_id) is None:
            raise FileNotFoundError("animation upload not found")
