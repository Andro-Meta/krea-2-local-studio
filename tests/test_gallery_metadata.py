from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class GalleryMetadataTests(unittest.TestCase):
    def test_gallery_persists_and_returns_metadata_json(self) -> None:
        import gallery

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            out_dir = Path(tmp) / "outputs"
            out_dir.mkdir()

            async def run() -> None:
                with (
                    patch.object(gallery, "DB_PATH", db_path),
                    patch.object(gallery, "OUTPUTS_DIR", out_dir),
                ):
                    await gallery.init_db()
                    image_id = await gallery.save_image(
                        "example.png",
                        prompt="a glass forest",
                        seed=99,
                        metadata={"prompt": "a glass forest", "seed": 99, "steps": 8},
                    )
                    data = await gallery.get_gallery()

                self.assertEqual(image_id, 1)
                self.assertEqual(data["items"][0]["metadata"]["prompt"], "a glass forest")
                self.assertEqual(data["items"][0]["metadata"]["seed"], 99)

            asyncio.run(run())

    def test_v2_metadata_shape_has_replay_engine_fields(self) -> None:
        from generation_metadata import build_generation_metadata
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="a silver robot",
            mode="redraw",
            diffusion_engine="gguf_external",
            checkpoint="turbo",
            quantization="fp8",
            sampler="euler",
            scheduler="simple",
            width=1024,
            height=1024,
        )

        metadata = build_generation_metadata(
            req,
            base_seed=7,
            resolved_provider="gguf_external",
            runtime={"provider": "gguf_external", "sd_cli_path": "tools/sd-cli.exe"},
        )

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["diffusion_engine"], "gguf_external")
        self.assertEqual(metadata["engine"]["id"], "gguf_external")
        self.assertEqual(metadata["engine"]["resolved_provider"], "gguf_external")
        self.assertEqual(metadata["runtime"]["provider"], "gguf_external")
        self.assertEqual(metadata["source"]["mode"], "redraw")

    def test_gallery_scopes_rows_by_owner(self) -> None:
        import gallery

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            out_dir = Path(tmp) / "outputs"
            out_dir.mkdir()

            async def run() -> None:
                with (
                    patch.object(gallery, "DB_PATH", db_path),
                    patch.object(gallery, "OUTPUTS_DIR", out_dir),
                ):
                    await gallery.init_db()
                    await gallery.save_image("alice.png", prompt="a", owner_username="alice")
                    await gallery.save_image("bob.png", prompt="b", owner_username="bob")
                    await gallery.save_image("legacy.png", prompt="legacy")

                    alice = await gallery.get_gallery(owner_username="alice", is_admin=False)
                    bob = await gallery.get_gallery(owner_username="bob", is_admin=False)
                    admin = await gallery.get_gallery(is_admin=True)

                self.assertEqual([item["filename"] for item in alice["items"]], ["alice.png"])
                self.assertEqual([item["filename"] for item in bob["items"]], ["bob.png"])
                self.assertEqual(admin["total"], 3)

            asyncio.run(run())

    def test_gallery_mutations_are_owner_scoped(self) -> None:
        import gallery

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            out_dir = Path(tmp) / "outputs"
            out_dir.mkdir()
            (out_dir / "alice.png").write_bytes(b"fake")
            (out_dir / "bob.png").write_bytes(b"fake")

            async def run() -> None:
                with (
                    patch.object(gallery, "DB_PATH", db_path),
                    patch.object(gallery, "OUTPUTS_DIR", out_dir),
                ):
                    await gallery.init_db()
                    alice_id = await gallery.save_image("alice.png", owner_username="alice")
                    bob_id = await gallery.save_image("bob.png", owner_username="bob")

                    self.assertFalse(await gallery.set_favorite(bob_id, True, owner_username="alice", is_admin=False))
                    self.assertTrue(await gallery.set_favorite(bob_id, True, owner_username="bob", is_admin=False))
                    self.assertIsNone(await gallery.delete_image(bob_id, owner_username="alice", is_admin=False))
                    self.assertEqual(await gallery.delete_image(alice_id, owner_username="admin", is_admin=True), "alice.png")

                self.assertFalse((out_dir / "alice.png").exists())
                self.assertTrue((out_dir / "bob.png").exists())

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
