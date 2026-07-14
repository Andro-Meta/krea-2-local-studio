from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

if __package__:
    from . import comfy_client
    from .animation_plan import (
        DEFAULT_ANIMATION_CHUNK_SIZE,
        apply_prompt_strength_boost,
        build_chunk_ranges,
        build_seed_plan,
        evaluate_schedule,
        numeric_chunk_schedule,
        parse_prompt_schedule,
        prompt_chunk_schedule,
    )
    from .animation_state import AnimationProject
    from .comfy_workflows import GraphBuilder, build_krea_model_bundle
    from .schemas import AnimateRequest, GenerationRequest
else:
    import comfy_client
    from animation_plan import (
        DEFAULT_ANIMATION_CHUNK_SIZE,
        apply_prompt_strength_boost,
        build_chunk_ranges,
        build_seed_plan,
        evaluate_schedule,
        numeric_chunk_schedule,
        parse_prompt_schedule,
        prompt_chunk_schedule,
    )
    from animation_state import AnimationProject
    from comfy_workflows import GraphBuilder, build_krea_model_bundle
    from schemas import AnimateRequest, GenerationRequest

logger = logging.getLogger("krea2.comfy.deforum")


class ComfyDeforumError(RuntimeError):
    """Sanitized adapter failure safe for persistence and API responses."""
KREADEFORUM_REVISION = "49bb6752ab045fac25652f3e9207d4706bf5c646"
KREADEFORUM_PATCH_VERSION = "krea2-chunking-v2"
KREADEFORUM_PATCHED_ANIMATOR_SHA256 = (
    "2dd533428c84809c5768951d414b7edac451c4c9ba09e1ab6ced132f713f4461"
)
KREADEFORUM_PATCH_SHA256 = (
    "2ef30ed45db588cad4472ac8edffce00f9a89bf249b9c4460e19e213df7f0978"
)
MIDAS_READINESS_MARKER = (
    Path(__file__).resolve().parent.parent
    / "ComfyUI"
    / "models"
    / "midas"
    / "krea-midas-small-ready.json"
)
REQUIRED_NODES = (
    "KreaDeforumAnimator",
    "KreaDeforumSaveVideo",
    "KreaDeforumSchedulePreview",
    "KreaDeforumChunkAdapterVersion",
)
_PATCHED_ANIMATOR_INPUTS = frozenset(
    {
        "frame_offset",
        "init_image_is_previous",
        "reference_image",
        "seed_plan",
        "hybrid_video_has_context",
        "prompt_blend_frames",
    }
)
_STATUS_LOCK = threading.Lock()
_STATUS_CACHE: dict | None = None
_STATUS_CACHE_TIME = 0.0
_STATUS_SUCCESS: dict | None = None
_STATUS_SUCCESS_TIME = 0.0
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_FRAME_BYTES = 32 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_NUMERIC_LITERAL = (
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)
_NUMERIC_CHUNK_ENTRY = re.compile(
    rf"\s*(\d+)\s*:\s*\(\s*({_NUMERIC_LITERAL})\s*\)\s*"
)
_NUMERIC_SCHEDULE_FIELDS = (
    "cfg_schedule",
    "strength_schedule",
    "zoom_schedule",
    "angle_schedule",
    "translation_x_schedule",
    "translation_y_schedule",
    "translation_z_schedule",
    "rotation_3d_x_schedule",
    "rotation_3d_y_schedule",
    "rotation_3d_z_schedule",
)


def _copy_status(value: dict) -> dict:
    return {
        **value,
        "missing_nodes": list(value["missing_nodes"]),
        "incompatible_capabilities": list(
            value.get("incompatible_capabilities", [])
        ),
    }


def midas_readiness() -> dict[str, object]:
    reason = (
        "MiDaS 3D setup is incomplete. Run install.bat, then restart ComfyUI."
    )
    try:
        payload = json.loads(MIDAS_READINESS_MARKER.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"midas_ready": False, "midas_reason": reason}
    except (OSError, ValueError, TypeError):
        return {
            "midas_ready": False,
            "midas_reason": "MiDaS readiness marker is invalid. Re-run install.bat.",
        }
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("model") != "MiDaS_small"
        or not isinstance(payload.get("weights_path"), str)
        or not isinstance(payload.get("hub_repo_path"), str)
    ):
        return {
            "midas_ready": False,
            "midas_reason": "MiDaS readiness marker is invalid. Re-run install.bat.",
        }
    weights = Path(payload["weights_path"])
    repository = Path(payload["hub_repo_path"])
    if not weights.is_file() or not repository.is_dir():
        return {
            "midas_ready": False,
            "midas_reason": "MiDaS cache is incomplete. Re-run install.bat.",
        }
    return {
        "midas_ready": True,
        "midas_reason": "MiDaS_small readiness marker and cache verified.",
    }


def _capability_issues(object_info: dict) -> list[str]:
    issues: list[str] = []
    animator = object_info.get("KreaDeforumAnimator")
    if isinstance(animator, dict):
        inputs = animator.get("input", {})
        names = set(inputs.get("required", {})) | set(inputs.get("optional", {}))
        if not _PATCHED_ANIMATOR_INPUTS <= names:
            issues.append("patched animator input contract is incompatible")
    version_node = object_info.get("KreaDeforumChunkAdapterVersion")
    if isinstance(version_node, dict):
        version = (
            version_node.get("input", {})
            .get("required", {})
            .get("version")
        )
        default = (
            version[1].get("default")
            if isinstance(version, (list, tuple))
            and len(version) > 1
            and isinstance(version[1], dict)
            else None
        )
        if default != KREADEFORUM_PATCH_VERSION:
            issues.append("chunk adapter version capability is incompatible")
    return issues


def _required_object_info(timeout: float) -> dict:
    result: dict = {}
    for class_type in REQUIRED_NODES:
        payload = comfy_client.object_info(class_type, timeout=timeout)
        if not isinstance(payload, dict):
            raise TypeError("ComfyUI object_info response was not an object")
        node = payload.get(class_type)
        if isinstance(node, dict):
            result[class_type] = node
    return result


def status(
    timeout: float = 5.0,
    *,
    force_refresh: bool = False,
    cache_ttl: float = 5.0,
    stale_ttl: float = 60.0,
) -> dict:
    global _STATUS_CACHE, _STATUS_CACHE_TIME
    global _STATUS_SUCCESS, _STATUS_SUCCESS_TIME
    midas = midas_readiness()
    with _STATUS_LOCK:
        now = time.monotonic()
        if (
            not force_refresh
            and _STATUS_CACHE is not None
            and now - _STATUS_CACHE_TIME < cache_ttl
        ):
            return {**_copy_status(_STATUS_CACHE), **midas}
    try:
        object_info = _required_object_info(timeout)
    except Exception as exc:
        logger.warning(
            "KreaDeforum capability probe failed (%s)",
            type(exc).__name__,
        )
        logger.debug("KreaDeforum capability probe details", exc_info=True)
        with _STATUS_LOCK:
            now = time.monotonic()
            if (
                _STATUS_SUCCESS is not None
                and now - _STATUS_SUCCESS_TIME <= stale_ttl
            ):
                return {
                    **_copy_status(_STATUS_SUCCESS),
                    **midas,
                    "probe_failed": True,
                    "stale": True,
                }
        return {
            "available": False,
            "missing_nodes": [],
            "revision": KREADEFORUM_REVISION,
            "external": True,
            "license": "unspecified",
            "patch_version": KREADEFORUM_PATCH_VERSION,
            "patched_animator_sha256": KREADEFORUM_PATCHED_ANIMATOR_SHA256,
            "patch_sha256": KREADEFORUM_PATCH_SHA256,
            "probe_failed": True,
            "stale": False,
            "incompatible_capabilities": [],
            **midas,
        }
    missing = [node for node in REQUIRED_NODES if node not in object_info]
    incompatible = _capability_issues(object_info)
    result = {
        "available": not missing and not incompatible,
        "missing_nodes": missing,
        "incompatible_capabilities": incompatible,
        "revision": KREADEFORUM_REVISION,
        "external": True,
        "license": "unspecified",
        "patch_version": KREADEFORUM_PATCH_VERSION,
        "patched_animator_sha256": KREADEFORUM_PATCHED_ANIMATOR_SHA256,
        "patch_sha256": KREADEFORUM_PATCH_SHA256,
        "probe_failed": False,
        "stale": False,
        **midas,
    }
    with _STATUS_LOCK:
        _STATUS_CACHE = {
            **result,
            "missing_nodes": list(missing),
        }
        _STATUS_CACHE_TIME = time.monotonic()
        if result["available"]:
            _STATUS_SUCCESS = _copy_status(result)
            _STATUS_SUCCESS_TIME = _STATUS_CACHE_TIME
    return _copy_status(result)


class _UploadLease:
    """Own Comfy API uploads made by one animation graph build.

    ComfyUI's standard upload API has no delete operation. Closing therefore
    drops only this adapter's ownership records; it never guesses at or deletes
    arbitrary files from a Comfy input directory.
    """

    def __init__(self, input_root: Path | None = None) -> None:
        self._owned: list[str] = []
        self.input_root = input_root

    @property
    def managed(self) -> bool:
        return self.input_root is not None

    def upload(self, raw: bytes) -> str:
        name = f"krea_deforum_{uuid.uuid4().hex}.png"
        try:
            response = requests.post(
                f"{comfy_client.comfy_base_url()}/upload/image",
                files={"image": (name, raw, "image/png")},
                data={"overwrite": "false"},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.exception("KreaDeforum owned image upload failed")
            raise ComfyDeforumError("ComfyUI image upload failed.") from exc
        returned = payload.get("name") or payload.get("filename")
        if (
            not isinstance(returned, str)
            or returned != name
            or Path(returned).name != returned
            or not returned.startswith("krea_deforum_")
        ):
            raise ValueError("ComfyUI returned an invalid owned upload filename")
        if self.input_root is not None:
            self._owned.append(returned)
        return returned

    def metadata(self) -> dict[str, str]:
        return {
            "input_cleanup": (
                "ComfyUI upload API does not expose safe deletion; only "
                "adapter ownership records are released"
            )
        }

    def close(self) -> None:
        if self.input_root is not None:
            root = Path(os.path.abspath(self.input_root))
            if (
                not root.is_dir()
                or root.is_symlink()
                or _is_reparse(root)
                or root.resolve() != root
            ):
                self._owned.clear()
                return
            for name in self._owned:
                try:
                    if (
                        Path(name).name != name
                        or not name.startswith("krea_deforum_")
                    ):
                        continue
                    target = root / name
                    target.relative_to(root)
                    if target.is_symlink() or _is_reparse(target):
                        continue
                    target.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove adapter-owned Comfy input %s", name
                    )
        self._owned.clear()


def _managed_local_input_root() -> Path | None:
    parsed = urlparse(comfy_client.comfy_base_url())
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        return None
    try:
        if __package__:
            from .settings import BASE_DIR
        else:
            from settings import BASE_DIR

        root = Path(BASE_DIR) / "ComfyUI" / "input"
    except (ImportError, ModuleNotFoundError):
        return None
    return Path(os.path.abspath(root))


def _new_upload_lease() -> _UploadLease:
    return _UploadLease(input_root=_managed_local_input_root())


def _png_dimensions(
    raw: bytes, width: int, height: int, *, context: str
) -> tuple[int, int]:
    if (
        len(raw) < 33
        or not raw.startswith(_PNG_SIGNATURE)
        or raw[8:12] != b"\x00\x00\x00\r"
        or raw[12:16] != b"IHDR"
    ):
        raise ValueError(f"{context} is a malformed PNG")
    image_width, image_height = struct.unpack(">II", raw[16:24])
    if image_width < 1 or image_height < 1:
        raise ValueError(f"{context} is a malformed PNG")
    if image_width * image_height > width * height * 4:
        raise ValueError(f"{context} exceeds the decompression pixel cap")
    if (image_width, image_height) != (width, height):
        raise ValueError(f"{context} dimensions do not match the request")
    return image_width, image_height


def _decode_init_png(value: str, width: int, height: int) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("init_image_b64 must be non-empty base64 text")
    encoded = value.split(",", 1)[-1] if value.lstrip().startswith("data:") else value
    encoded_limit = ((MAX_INPUT_BYTES + 2) // 3) * 4
    if len(encoded) > encoded_limit:
        raise ValueError("init image exceeds the encoded size cap")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("init_image_b64 is not valid base64") from exc
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("init image exceeds the decoded byte size cap")
    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("init image must be a PNG")
    _png_dimensions(raw, width, height, context="init image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG":
                    raise ValueError("init image must be a PNG")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("init image must be a single-frame PNG")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("init image must be a single-frame PNG")
                normalized = image.convert("RGB")
                output = io.BytesIO()
                normalized.save(output, format="PNG")
                result = output.getvalue()
                if len(result) > MAX_INPUT_BYTES:
                    raise ValueError(
                        "normalized init image exceeds the decoded byte size cap"
                    )
                return result
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("init image is a malformed PNG") from exc


def _default_loader_request(req: AnimateRequest) -> GenerationRequest:
    try:
        if __package__:
            from .settings import settings
        else:
            from settings import settings

        engine = settings.diffusion_engine
        quantization = settings.krea2_auto_quant
    except (ImportError, ModuleNotFoundError):
        engine = "native_pytorch"
        quantization = "fp8"
    valid_engines = {
        "native_pytorch",
        "native_gguf",
        "native_int8_convrot",
        "gguf_external",
        "int8_convrot_external",
    }
    valid_quantizations = {"bf16", "fp16", "fp8", "gguf", "int8"}
    if engine not in valid_engines:
        raise ValueError(f"unsupported diffusion engine: {engine}")
    if quantization not in valid_quantizations:
        raise ValueError(f"unsupported Krea quantization: {quantization}")
    if engine == "native_gguf":
        quantization = "gguf"
    elif engine == "native_int8_convrot":
        quantization = "int8"
    return GenerationRequest(
        prompt="",
        negative_prompt=req.negative_prompt,
        checkpoint="turbo",
        diffusion_engine=engine,
        quantization=quantization,
        width=req.width,
        height=req.height,
        steps=req.steps,
        sampler=req.sampler_name,
        scheduler=req.scheduler,
        seed=req.seed,
    )


def _is_reparse(path: Path) -> bool:
    try:
        if hasattr(os.path, "isjunction") and os.path.isjunction(path):
            return True
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _controlled_video_path(source: Path, root: Path) -> Path:
    supplied_root = Path(root)
    if (
        not os.path.lexists(supplied_root)
        or supplied_root.is_symlink()
        or _is_reparse(supplied_root)
    ):
        raise ValueError(
            "controlled root must be an existing regular directory, not a "
            "link or reparse point"
        )
    try:
        root_stat = supplied_root.lstat()
    except OSError as exc:
        raise ValueError("controlled root must be an existing regular directory") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("controlled root must be an existing regular directory")
    root = Path(os.path.abspath(supplied_root))
    source = Path(os.path.abspath(source))
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("source video escapes the controlled root") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink() or _is_reparse(current):
            raise ValueError("source video path contains a link or reparse point")
    try:
        canonical_root = root.resolve(strict=True)
        canonical_source = source.resolve(strict=True)
        canonical_source.relative_to(canonical_root)
    except (OSError, ValueError) as exc:
        raise ValueError("source video escapes the controlled root") from exc
    if not canonical_source.is_file():
        raise ValueError("source video must be a regular file")
    parsed = urlparse(comfy_client.comfy_base_url())
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "Video Input cannot pass a local path to remote ComfyUI; use a shared "
            "controlled media root"
        )
    return canonical_source


def _after_video_source_open(path: Path) -> None:
    """Race-test seam after the no-follow source descriptor is opened."""


def _after_video_source_copy(path: Path) -> None:
    """Mutation-test seam before descriptor stability verification."""


class _VideoSliceLease:
    def __init__(self) -> None:
        self.root: Path | None = None

    def prepare(
        self,
        source: Path,
        controlled_root: Path,
        *,
        start: int,
        end: int,
        include_previous_context: bool = False,
    ) -> Path:
        import cv2

        source = _controlled_video_path(source, controlled_root)
        before = source.lstat()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(source, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise ValueError("source video changed while opening")
            _after_video_source_open(source)
            self.root = Path(tempfile.mkdtemp(prefix="krea-deforum-video-"))
            private_source = self.root / "source.mp4"
            digest = hashlib.sha256()
            with os.fdopen(os.dup(descriptor), "rb") as reader:
                with private_source.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
                        digest.update(chunk)
            _after_video_source_copy(source)
            after = os.fstat(descriptor)
            identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if hasattr(opened, "st_ctime_ns"):
                identity_fields += ("st_ctime_ns",)
            if any(
                getattr(opened, field) != getattr(after, field)
                for field in identity_fields
            ):
                raise ValueError("source video changed while copying")
            with private_source.open("rb") as copied:
                copied_digest = hashlib.file_digest(copied, "sha256").hexdigest()
            if copied_digest != digest.hexdigest():
                raise ValueError("source video copy integrity check failed")
        finally:
            os.close(descriptor)

        capture = cv2.VideoCapture(str(private_source))
        if not capture.isOpened():
            raise ValueError("source video is corrupt or unsupported")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width < 1 or height < 1:
            capture.release()
            raise ValueError("source video metadata is invalid")
        slice_path = self.root / "chunk.mp4"
        writer = cv2.VideoWriter(
            str(slice_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("could not encode the bounded source video slice")
        selected = 0
        index = 0
        slice_start = start - 1 if include_previous_context and start > 0 else start
        try:
            while index < end:
                ok, frame = capture.read()
                if not ok:
                    break
                if index >= slice_start:
                    if frame.shape[:2] != (height, width):
                        raise ValueError("source video dimensions changed")
                    writer.write(frame)
                    selected += 1
                index += 1
        finally:
            capture.release()
            writer.release()
        expected = end - slice_start
        if selected != expected:
            raise ValueError(
                "source video has insufficient frames for the requested chunk"
            )
        return slice_path.resolve()

    def close(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None


def validate_animation_runtime(
    project: AnimationProject,
    req: AnimateRequest,
    loader_request: GenerationRequest | None = None,
    *,
    capability_status: dict | None = None,
) -> GenerationRequest:
    if project.request != req.model_dump(mode="python"):
        raise ValueError("animation request does not match the project snapshot")
    canonical = [
        list(bounds)
        for bounds in build_chunk_ranges(
            project.total_frames,
            DEFAULT_ANIMATION_CHUNK_SIZE,
            req.diffusion_cadence,
        )
    ]
    if project.total_frames != req.total_frames or project.chunk_ranges != canonical:
        raise ValueError("project runtime canonical chunk ranges do not match")
    expected_seeds = build_seed_plan(
        project.seed_base, req.seed_behavior, project.total_frames
    )
    if project.seed_plan != expected_seeds:
        raise ValueError("whole project seed plan failed preflight validation")
    managed = _managed_local_input_root() is not None
    if req.animation_mode == "Video Input" and not managed:
        raise ValueError(
            "Video Input cannot use remote ComfyUI; managed local source "
            "sharing is required"
        )
    # Multi-chunk init/ref frames use Comfy's HTTP /upload/image API, so remote
    # Comfy is allowed. Video Input still needs a local filesystem slice above.
    if capability_status is not None and not capability_status.get(
        "available", False
    ):
        details = (
            capability_status.get("incompatible_capabilities")
            or capability_status.get("missing_nodes")
            or ["patched chunk adapter capability"]
        )
        raise ValueError(
            "KreaDeforum runtime is incompatible: " + ", ".join(details)
        )
    return loader_request or _default_loader_request(req)


def _validate_chunk(
    req: AnimateRequest, project: AnimationProject, start: int, end: int
) -> None:
    if project.request != req.model_dump(mode="python"):
        raise ValueError("animation request does not match the project snapshot")
    if project.total_frames != req.total_frames:
        raise ValueError("project total_frames does not match the request")
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("chunk bounds must be non-empty integers")
    canonical = [
        list(bounds)
        for bounds in build_chunk_ranges(
            project.total_frames,
            DEFAULT_ANIMATION_CHUNK_SIZE,
            req.diffusion_cadence,
        )
    ]
    if project.chunk_ranges != canonical:
        raise ValueError("project chunk_ranges do not match canonical chunk ranges")
    chunk_index = (
        project.active_chunk_index
        if project.active_chunk_index is not None
        else project.next_chunk_index
    )
    if (
        type(chunk_index) is not int
        or not 0 <= chunk_index < len(canonical)
        or [start, end] != canonical[chunk_index]
    ):
        raise ValueError("start/end must exactly match the active or next chunk range")


def _seed_adapter(
    req: AnimateRequest, project: AnimationProject, start: int, end: int
) -> tuple[int, str, list[str]]:
    behavior = req.seed_behavior
    warnings_out: list[str] = []
    return project.seed_plan[start], behavior, warnings_out


def _numeric_schedules(req: AnimateRequest, start: int, end: int) -> dict[str, str]:
    schedules: dict[str, str] = {}
    for field in _NUMERIC_SCHEDULE_FIELDS:
        values = evaluate_schedule(getattr(req, field), req.total_frames)
        rendered = numeric_chunk_schedule(values, start, end)
        _validate_numeric_chunk_schedule(rendered)
        schedules[field] = rendered
    return schedules


def _validate_numeric_chunk_schedule(schedule: str) -> None:
    if not isinstance(schedule, str) or not schedule.strip():
        raise ValueError("chunk schedule must contain numeric literal entries")
    entries = schedule.split(",")
    for expected_frame, entry in enumerate(entries):
        match = _NUMERIC_CHUNK_ENTRY.fullmatch(entry)
        if match is None or int(match.group(1)) != expected_frame:
            raise ValueError(
                "chunk schedule entries must contain sequential numeric literals"
            )
        try:
            value = float(match.group(2))
        except ValueError as exc:
            raise ValueError(
                "chunk schedule entries must contain finite numeric literals"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                "chunk schedule entries must contain finite numeric literals"
            )


def _build_animation_chunk_graph(
    req: AnimateRequest,
    project: AnimationProject,
    *,
    start: int,
    end: int,
    init_image_b64: str | None,
    reference_image_b64: str | None,
    source_video_path: Path | None,
    controlled_video_root: Path | None,
    loader_request: GenerationRequest | None,
    lease: _UploadLease,
    video_lease: _VideoSliceLease,
) -> tuple[dict, dict]:
    loader_request = validate_animation_runtime(
        project, req, loader_request
    )
    _validate_chunk(req, project, start, end)
    if start > 0 and not init_image_b64:
        raise ValueError("later animation chunks require init_image_b64")
    if (
        start > 0
        and req.color_coherence == "Match Frame 0 LAB"
        and not reference_image_b64
    ):
        raise ValueError(
            "later LAB-coherent chunks require reference_image_b64"
        )
    if req.animation_mode == "Video Input":
        if source_video_path is None or controlled_video_root is None:
            raise ValueError(
                "Video Input requires source_video_path and controlled_video_root"
            )
        has_context = (
            start > 0
            and req.hybrid_mode == "optical_flow"
        )
        video = video_lease.prepare(
            source_video_path,
            controlled_video_root,
            start=start,
            end=end,
            include_previous_context=has_context,
        )
    else:
        if source_video_path is not None or controlled_video_root is not None:
            raise ValueError("source video paths are valid only for Video Input")
        video = None
        has_context = False

    seed, seed_behavior, seed_warnings = _seed_adapter(req, project, start, end)
    schedules = _numeric_schedules(req, start, end)
    prompt_values = parse_prompt_schedule(req.prompt_schedule, req.total_frames)
    strength_values = apply_prompt_strength_boost(
        evaluate_schedule(req.strength_schedule, req.total_frames),
        prompt_values,
        boost=float(req.prompt_strength_boost),
        window=int(req.prompt_strength_boost_frames),
    )
    schedules["strength_schedule"] = numeric_chunk_schedule(
        strength_values, start, end
    )
    _validate_numeric_chunk_schedule(schedules["strength_schedule"])
    prompt_schedule = prompt_chunk_schedule(prompt_values, start, end)

    graph = GraphBuilder()
    model, clip, vae = build_krea_model_bundle(
        graph, loader_request
    )
    inputs = {
        "model": model,
        "clip": clip,
        "vae": vae,
        "width": req.width,
        "height": req.height,
        "max_frames": end - start,
        "steps": req.steps,
        "sampler_name": req.sampler_name,
        "scheduler": req.scheduler,
        "seed": seed,
        "seed_behavior": seed_behavior,
        "seed_plan": json.dumps(project.seed_plan[start:end], separators=(",", ":")),
        "frame_offset": start,
        "init_image_is_previous": start > 0,
        "animation_mode": req.animation_mode,
        "border_mode": req.border_mode,
        "prompt_schedule": prompt_schedule,
        "negative_prompt": req.negative_prompt,
        "cfg_schedule": schedules["cfg_schedule"],
        "strength_schedule": schedules["strength_schedule"],
        "zoom_schedule": schedules["zoom_schedule"],
        "angle_schedule": schedules["angle_schedule"],
        "translation_x_schedule": schedules["translation_x_schedule"],
        "translation_y_schedule": schedules["translation_y_schedule"],
        "translation_z_schedule": schedules["translation_z_schedule"],
        "rotation_3d_x_schedule": schedules["rotation_3d_x_schedule"],
        "rotation_3d_y_schedule": schedules["rotation_3d_y_schedule"],
        "rotation_3d_z_schedule": schedules["rotation_3d_z_schedule"],
        "color_coherence": req.color_coherence,
        "diffusion_cadence": req.diffusion_cadence,
        "prompt_blend_frames": int(req.prompt_blend_frames),
    }
    cleanup_metadata: dict[str, str] = {}
    if video is not None:
        hybrid_schedule = numeric_chunk_schedule(
            evaluate_schedule(req.hybrid_strength_schedule, req.total_frames),
            start,
            end,
        )
        _validate_numeric_chunk_schedule(hybrid_schedule)
        inputs.update(
            {
                "hybrid_video_path": str(video),
                "hybrid_strength_schedule": hybrid_schedule,
                "hybrid_mode": req.hybrid_mode,
                "hybrid_video_has_context": has_context,
            }
        )
    if init_image_b64:
        raw = _decode_init_png(init_image_b64, req.width, req.height)
        uploaded = lease.upload(raw)
        graph.add("LoadImage", {"image": uploaded}, node_id="init_image")
        inputs["init_image"] = ["init_image", 0]
        cleanup_metadata = lease.metadata()
        seed_warnings.append(cleanup_metadata.get("input_cleanup", ""))
    if reference_image_b64:
        raw = _decode_init_png(reference_image_b64, req.width, req.height)
        uploaded = lease.upload(raw)
        graph.add("LoadImage", {"image": uploaded}, node_id="reference_image")
        inputs["reference_image"] = ["reference_image", 0]
    animator_id = graph.add(
        "KreaDeforumAnimator", inputs, node_id="deforum_animator"
    )
    graph.add(
        "SaveImageWebsocket",
        {"images": [animator_id, 0]},
        node_id=comfy_client.WS_IMAGE_NODE,
    )
    metadata = {
        "start": start,
        "end": end,
        "frame_count": end - start,
        "resolved_seed": seed,
        "seed_base": project.seed_base,
        "seed_behavior": seed_behavior,
        "sampler": req.sampler_name,
        "scheduler": req.scheduler,
        "external_revision": KREADEFORUM_REVISION,
        "warnings": [item for item in seed_warnings if item],
        "requirements": {"requires_midas": req.animation_mode == "3D"},
        **cleanup_metadata,
    }
    return graph.graph(), metadata


class AnimationGraphLease:
    def __init__(
        self,
        graph: dict,
        metadata: dict,
        upload_lease: _UploadLease,
        video_lease: _VideoSliceLease,
    ) -> None:
        self.graph = graph
        self.metadata = metadata
        self._upload_lease = upload_lease
        self._video_lease = video_lease
        self._closed = False

    def __iter__(self):
        yield self.graph
        yield self.metadata

    def __enter__(self) -> "AnimationGraphLease":
        if self._closed:
            raise RuntimeError("animation graph lease is already closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._upload_lease.close()
        self._video_lease.close()


def build_animation_chunk_graph(
    req: AnimateRequest,
    project: AnimationProject,
    *,
    start: int,
    end: int,
    init_image_b64: str | None = None,
    reference_image_b64: str | None = None,
    source_video_path: Path | None = None,
    controlled_video_root: Path | None = None,
    loader_request: GenerationRequest | None = None,
) -> AnimationGraphLease:
    lease = _new_upload_lease()
    video_lease = _VideoSliceLease()
    try:
        graph, metadata = _build_animation_chunk_graph(
            req,
            project,
            start=start,
            end=end,
            init_image_b64=init_image_b64,
            reference_image_b64=reference_image_b64,
            source_video_path=source_video_path,
            controlled_video_root=controlled_video_root,
            loader_request=loader_request,
            lease=lease,
            video_lease=video_lease,
        )
        return AnimationGraphLease(graph, metadata, lease, video_lease)
    except BaseException:
        lease.close()
        video_lease.close()
        raise


def _validate_frame(blob: bytes, req: AnimateRequest) -> None:
    if not isinstance(blob, bytes):
        raise ValueError("ComfyUI frame output must be PNG bytes")
    if len(blob) > MAX_FRAME_BYTES:
        raise ValueError("ComfyUI frame exceeds the maximum byte size")
    if not blob.startswith(_PNG_SIGNATURE):
        raise ValueError("ComfyUI frame output is not a PNG")
    _png_dimensions(
        blob, req.width, req.height, context="ComfyUI frame output"
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(blob)) as image:
                if image.format != "PNG":
                    raise ValueError("ComfyUI frame output is not a PNG")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("ComfyUI frame must be a single-frame PNG")
                image.verify()
            with Image.open(io.BytesIO(blob)) as image:
                image.load()
                if image.size != (req.width, req.height):
                    raise ValueError("ComfyUI frame dimensions do not match the request")
                if image.mode != "RGB":
                    raise ValueError("ComfyUI frame must be RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("ComfyUI frame is a malformed PNG") from exc


def render_animation_chunk(
    req: AnimateRequest,
    project: AnimationProject,
    *,
    start: int,
    end: int,
    init_image_b64: str | None = None,
    reference_image_b64: str | None = None,
    source_video_path: Path | None = None,
    controlled_video_root: Path | None = None,
    loader_request: GenerationRequest | None = None,
    progress_cb=None,
    prompt_id_cb=None,
    client: comfy_client.ComfyClient | None = None,
) -> list[bytes]:
    availability = status(force_refresh=True)
    if not availability["available"]:
        details = availability.get("incompatible_capabilities") or availability[
            "missing_nodes"
        ]
        raise RuntimeError(
            "KreaDeforum nodes are unavailable: " + ", ".join(details)
        )
    validate_animation_runtime(
        project,
        req,
        loader_request,
        capability_status=availability,
    )
    with build_animation_chunk_graph(
            req,
            project,
            start=start,
            end=end,
            init_image_b64=init_image_b64,
            reference_image_b64=reference_image_b64,
            source_video_path=source_video_path,
            controlled_video_root=controlled_video_root,
            loader_request=loader_request,
        ) as graph_lease:
        graph = graph_lease.graph
        timeout = min(1800, max(60, 60 + (end - start) * req.steps * 3))
        try:
            blobs = (client or comfy_client.ComfyClient()).run(
                graph,
                progress_cb=progress_cb,
                prompt_id_cb=prompt_id_cb,
                image_node_id=comfy_client.WS_IMAGE_NODE,
                timeout=timeout,
            )
        except Exception as exc:
            logger.exception("KreaDeforum chunk execution failed")
            raise ComfyDeforumError(
                "KreaDeforum chunk execution failed."
            ) from exc
        expected = end - start
        if len(blobs) != expected:
            raise ValueError(
                f"ComfyUI must return exactly {expected} animation frame outputs"
            )
        for blob in blobs:
            _validate_frame(blob, req)
        return blobs
