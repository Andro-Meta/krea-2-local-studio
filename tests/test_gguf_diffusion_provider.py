from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class GgufDiffusionProviderTests(unittest.TestCase):
    def test_builds_stable_diffusion_cpp_command_without_path_lookup(self) -> None:
        from gguf_diffusion_provider import GgufRuntimeSettings, build_gguf_command

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exe = root / "sd-cli.exe"
            turbo = root / "krea2_turbo_q4.gguf"
            llm = root / "qwen3vl_q4.gguf"
            vae = root / "qwen_image_vae.safetensors"
            for path in (exe, turbo, llm, vae):
                path.write_bytes(b"x")
            req = SimpleNamespace(
                prompt="a red fox",
                negative_prompt="blur",
                checkpoint="turbo",
                width=768,
                height=768,
                steps=8,
                cfg=0.0,
                seed=123,
                mode="txt2img",
            )

            cmd, output = build_gguf_command(
                req,
                GgufRuntimeSettings(sd_cli_path=str(exe), turbo_path=str(turbo), raw_path="", llm_path=str(llm), vae_path=str(vae)),
                output_dir=root / "out",
            )

        self.assertEqual(cmd[0], str(exe))
        self.assertIn("--diffusion-model", cmd)
        self.assertIn(str(turbo), cmd)
        self.assertIn("--llm", cmd)
        self.assertIn(str(llm), cmd)
        self.assertIn("--vae", cmd)
        self.assertIn(str(vae), cmd)
        self.assertIn("--cfg-scale", cmd)
        self.assertEqual(output.suffix, ".png")

    def test_rejects_missing_runtime_paths(self) -> None:
        from gguf_diffusion_provider import GgufRuntimeSettings, build_gguf_command

        req = SimpleNamespace(prompt="x", negative_prompt="", checkpoint="turbo", width=512, height=512, steps=8, cfg=0, seed=1, mode="txt2img")

        with self.assertRaisesRegex(ValueError, "sd-cli"):
            build_gguf_command(req, GgufRuntimeSettings(), output_dir=Path("out"))

    def test_installer_extracts_sd_cli_and_cudart(self) -> None:
        import zipfile
        import gguf_runtime_installer

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            downloads = root / "downloads"
            cuda_zip = downloads / "sd-master-test-bin-win-cuda12-x64.zip"
            cudart_zip = downloads / "cudart-sd-bin-win-cu12-x64.zip"
            downloads.mkdir()
            with zipfile.ZipFile(cuda_zip, "w") as zf:
                zf.writestr("bin/sd-cli.exe", "exe")
            with zipfile.ZipFile(cudart_zip, "w") as zf:
                zf.writestr("bin/cudart64_12.dll", "dll")

            def fake_download(url: str, dest: Path) -> Path:
                source = cudart_zip if "cudart" in url else cuda_zip
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read_bytes())
                return dest

            with (
                patch.object(gguf_runtime_installer, "_latest_release_assets", return_value={
                    "sd-master-test-bin-win-cuda12-x64.zip": "https://example.test/sd.zip",
                    "cudart-sd-bin-win-cu12-x64.zip": "https://example.test/cudart.zip",
                }),
                patch.object(gguf_runtime_installer, "_download", side_effect=fake_download),
            ):
                result = gguf_runtime_installer.install_stable_diffusion_cpp(root / "runtime")

            self.assertTrue(Path(result["sd_cli_path"]).exists())
            self.assertTrue((root / "runtime" / "cudart64_12.dll").exists())


if __name__ == "__main__":
    unittest.main()
