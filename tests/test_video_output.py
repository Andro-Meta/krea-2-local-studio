from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class _Writer:
    def __init__(self, path: str):
        self.path = Path(path)
        self.frames: list[tuple[int, int, int]] = []

    def append_data(self, frame):
        self.frames.append(tuple(int(value) for value in frame[0, 0]))

    def close(self):
        self.path.write_bytes(b"mp4:" + repr(self.frames).encode())


class VideoOutputTests(unittest.TestCase):
    def _frames(self, root: Path) -> list[Path]:
        paths = []
        for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
            path = root / f"frame_{index:06d}.png"
            Image.new("RGB", (18, 17), color).save(path)
            paths.append(path)
        return paths

    def test_finalize_streams_in_order_and_publishes_atomically(self):
        from video_output import finalize_mp4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = self._frames(root)
            destination = root / "animation.mp4"
            poster = root / "preview.jpg"
            writer = _Writer(str(destination.with_name(".animation.tmp.mp4")))

            with patch("video_output._create_h264_writer", return_value=writer):
                metadata = finalize_mp4(frames, destination, fps=12, poster_path=poster)

            self.assertEqual(writer.frames, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
            self.assertTrue(destination.is_file())
            self.assertTrue(poster.is_file())
            self.assertEqual(metadata["frame_count"], 3)
            self.assertEqual(metadata["width"] % 2, 0)
            self.assertEqual(metadata["height"] % 2, 0)
            self.assertEqual(metadata["sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())

    def test_finalize_failure_removes_temporary_media_only(self):
        from video_output import finalize_mp4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = self._frames(root)
            destination = root / "animation.mp4"
            poster = root / "preview.jpg"
            writer = _Writer(str(destination.with_name(".animation.tmp.mp4")))
            writer.append_data = lambda _frame: (_ for _ in ()).throw(RuntimeError("encode failed"))

            with (
                patch("video_output._create_h264_writer", return_value=writer),
                self.assertRaisesRegex(RuntimeError, "encode failed"),
            ):
                finalize_mp4(frames, destination, fps=12, poster_path=poster)

            self.assertFalse(destination.exists())
            self.assertFalse(poster.exists())
            self.assertEqual(len(list(root.glob("*.tmp.mp4"))), 0)
            self.assertTrue(all(path.exists() for path in frames))

    def test_rejects_mixed_frame_dimensions(self):
        from video_output import finalize_mp4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = self._frames(root)
            Image.new("RGB", (20, 20), "white").save(frames[-1])
            with self.assertRaisesRegex(ValueError, "dimensions"):
                finalize_mp4(
                    frames,
                    root / "animation.mp4",
                    fps=12,
                    poster_path=root / "preview.jpg",
                )

    def test_backend_module_imports_without_optional_ffmpeg_packages(self):
        code = """
import builtins, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("imageio"):
        raise ImportError("blocked optional dependency")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, r"%s")
import video_output
print("imported")
""" % BACKEND
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("imported", result.stdout)

    def test_missing_h264_dependency_fails_without_publishing(self):
        from video_output import VideoDependencyError, finalize_mp4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "video_output._create_h264_writer",
                    side_effect=VideoDependencyError(
                        "H.264 encoding requires imageio-ffmpeg or system ffmpeg with libx264."
                    ),
                ),
                self.assertRaisesRegex(VideoDependencyError, "imageio-ffmpeg"),
            ):
                finalize_mp4(
                    self._frames(root),
                    root / "animation.mp4",
                    fps=12,
                    poster_path=root / "preview.jpg",
                )
            self.assertFalse((root / "animation.mp4").exists())

    def test_real_output_is_h264_yuv420p_and_decodable(self):
        import imageio_ffmpeg

        from video_output import finalize_mp4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "animation.mp4"
            metadata = finalize_mp4(
                self._frames(root),
                destination,
                fps=12,
                poster_path=root / "preview.jpg",
            )
            reader = imageio_ffmpeg.read_frames(str(destination), pix_fmt="rgb24")
            stream = next(reader)
            first = next(reader)
            reader.close()

            self.assertIn(stream["codec"].lower(), {"h264", "avc1"})
            self.assertTrue(stream["pix_fmt"].startswith("yuv420p"))
            self.assertGreater(len(first), 0)
            self.assertEqual(metadata["codec"], "h264")
            payload = destination.read_bytes()
            self.assertLess(payload.find(b"moov"), payload.find(b"mdat"))


if __name__ == "__main__":
    unittest.main()
