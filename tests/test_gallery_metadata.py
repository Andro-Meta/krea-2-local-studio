from __future__ import annotations

import asyncio
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


class GalleryMetadataTests(unittest.TestCase):
    def _write_png(self, path: Path, metadata: dict | None = None) -> None:
        from PIL.PngImagePlugin import PngInfo

        info = PngInfo()
        if metadata:
            import json

            info.add_text("krea2_metadata", json.dumps(metadata))
        Image.new("RGB", (32, 32), (20, 30, 40)).save(path, pnginfo=info)

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
                    self._write_png(out_dir / "example.png", {"prompt": "a glass forest", "seed": 99, "steps": 8})
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
            diffusion_engine="native_gguf",
            checkpoint="turbo",
            quantization="gguf",
            sampler="euler",
            scheduler="simple",
            width=1024,
            height=1024,
        )

        metadata = build_generation_metadata(
            req,
            base_seed=7,
            resolved_provider="krea_native",
            runtime={"provider": "krea_native"},
        )

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["diffusion_engine"], "native_gguf")
        self.assertEqual(metadata["engine"]["id"], "native_gguf")
        self.assertEqual(metadata["engine"]["resolved_provider"], "krea_native")
        self.assertEqual(metadata["runtime"]["provider"], "krea_native")
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
                    self._write_png(out_dir / "alice.png")
                    self._write_png(out_dir / "bob.png")
                    self._write_png(out_dir / "legacy.png")
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

    def test_gallery_discovers_files_moved_into_owner_folder(self) -> None:
        import gallery

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            out_dir = Path(tmp) / "outputs"
            alice_dir = out_dir / "alice"
            alice_dir.mkdir(parents=True)
            self._write_png(alice_dir / "manual.png", {"prompt": "manual import", "seed": 123, "width": 32, "height": 32})

            async def run() -> None:
                with (
                    patch.object(gallery, "DB_PATH", db_path),
                    patch.object(gallery, "OUTPUTS_DIR", out_dir),
                ):
                    await gallery.init_db()
                    data = await gallery.get_gallery(owner_username="alice", is_admin=False)

                self.assertEqual(data["total"], 1)
                self.assertEqual(data["items"][0]["filename"], "alice/manual.png")
                # Filesystem orphans are imported into the DB on list (no longer ephemeral).
                self.assertFalse(data["items"][0]["filesystem_only"])
                self.assertEqual(data["items"][0]["metadata"]["prompt"], "manual import")
                self.assertIsNotNone(data["items"][0].get("thumbnail_b64"))

            asyncio.run(run())

    def test_gallery_prunes_db_rows_when_files_deleted_outside_app(self) -> None:
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
                    await gallery.save_image("deleted.png", prompt="gone")
                    data = await gallery.get_gallery()

                self.assertEqual(data["total"], 0)

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
