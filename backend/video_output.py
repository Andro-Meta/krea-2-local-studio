from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image


class VideoDependencyError(RuntimeError):
    """No verified browser-safe H.264 encoder is available."""


class _ImageioFfmpegWriter:
    def __init__(self, module, path: Path, *, fps: int, size: tuple[int, int]) -> None:
        self._generator = module.write_frames(
            str(path),
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            output_params=["-movflags", "+faststart"],
            ffmpeg_log_level="error",
        )
        self._generator.send(None)

    def append_data(self, frame: np.ndarray) -> None:
        self._generator.send(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self._generator.close()


class _SystemFfmpegWriter:
    def __init__(self, executable: str, path: Path, *, fps: int, size: tuple[int, int]) -> None:
        width, height = size
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append_data(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("H.264 encoder input closed unexpectedly")
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        if self._process.stderr:
            self._process.stderr.read()
        returncode = self._process.wait()
        if returncode:
            raise RuntimeError(
                "H.264 encoding failed. Check ffmpeg/libx264 installation."
            ) from None


def _system_ffmpeg_with_libx264() -> str | None:
    executable = shutil.which("ffmpeg")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return executable if result.returncode == 0 and "libx264" in result.stdout else None


def _create_h264_writer(path: Path, *, fps: int, size: tuple[int, int]):
    try:
        import imageio_ffmpeg
    except ImportError:
        imageio_ffmpeg = None
    if imageio_ffmpeg is not None:
        return _ImageioFfmpegWriter(
            imageio_ffmpeg, path, fps=fps, size=size
        )
    executable = _system_ffmpeg_with_libx264()
    if executable:
        return _SystemFfmpegWriter(
            executable, path, fps=fps, size=size
        )
    raise VideoDependencyError(
        "H.264 encoding requires imageio-ffmpeg or a system ffmpeg with libx264."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def finalize_mp4(
    frame_paths: Sequence[Path],
    destination: Path,
    *,
    fps: int,
    poster_path: Path,
) -> dict[str, object]:
    """Stream verified, ordered PNGs into one atomically published MP4."""
    if not frame_paths:
        raise ValueError("at least one frame is required")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60")

    paths = [Path(path) for path in frame_paths]
    destination = Path(destination)
    poster_path = Path(poster_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = destination.with_name(f".{destination.stem}.tmp.mp4")
    temporary_poster = poster_path.with_name(f".{poster_path.name}.tmp")
    writer = None
    expected_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    poster_written = False
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                if source.format != "PNG":
                    raise ValueError(f"frame {index} is not a PNG")
                if expected_size is None:
                    expected_size = source.size
                    output_size = (
                        source.width - source.width % 2,
                        source.height - source.height % 2,
                    )
                    if min(output_size) < 2:
                        raise ValueError("frame dimensions are too small")
                elif source.size != expected_size:
                    raise ValueError("frame dimensions do not match")
        assert output_size is not None
        writer = _create_h264_writer(
            temporary_video, fps=fps, size=output_size
        )
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                source.load()
                rgb = source.convert("RGB")
                if rgb.size != output_size:
                    rgb = rgb.crop((0, 0, output_size[0], output_size[1]))
                if index == 0:
                    poster = rgb.copy()
                    poster.thumbnail((1280, 1280))
                    poster.save(temporary_poster, format="JPEG", quality=88)
                    _fsync_file(temporary_poster)
                    poster_written = True
                writer.append_data(np.asarray(rgb))
        writer.close()
        writer = None
        _fsync_file(temporary_video)
        os.replace(temporary_video, destination)
        os.replace(temporary_poster, poster_path)
        poster_written = False
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        temporary_video.unlink(missing_ok=True)
        temporary_poster.unlink(missing_ok=True)
        if poster_written:
            poster_path.unlink(missing_ok=True)
        raise

    assert output_size is not None
    return {
        "width": output_size[0],
        "height": output_size[1],
        "duration": len(paths) / fps,
        "frame_count": len(paths),
        "fps": fps,
        "codec": "h264",
        "byte_size": destination.stat().st_size,
        "sha256": _sha256(destination),
        "poster_byte_size": poster_path.stat().st_size,
        "poster_sha256": _sha256(poster_path),
    }
