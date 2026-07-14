from __future__ import annotations

import os
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "kreadeforum_install.ps1"
INSTALLER = ROOT / "scripts" / "install_comfyui.ps1"
CHUNK_PATCH = ROOT / "patches" / "kreadeforum-krea2-chunking.patch"
PATCHED_ANIMATOR_SHA256 = (
    "2dd533428c84809c5768951d414b7edac451c4c9ba09e1ab6ced132f713f4461"
)
REPOSITORY_GIT_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in REPOSITORY_GIT_ENV_VARS:
        env.pop(name, None)
    return env


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
        env=_subprocess_env(),
    )


def _git(path: Path, *args: str) -> str:
    return _run(["git", "-C", str(path), *args]).stdout.strip()


def _ps_quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


class KreaDeforumInstallTests(unittest.TestCase):
    def test_subprocess_environment_scrubs_repository_git_overrides(self) -> None:
        overrides = {
            "GIT_DIR": "foreign.git",
            "GIT_WORK_TREE": "foreign-tree",
            "GIT_INDEX_FILE": "foreign-index",
            "GIT_OBJECT_DIRECTORY": "foreign-objects",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": "foreign-alternates",
            "GIT_COMMON_DIR": "foreign-common",
            "GIT_NAMESPACE": "foreign-namespace",
        }

        with patch.dict(os.environ, overrides, clear=False):
            child_env = _subprocess_env()

        for name in overrides:
            self.assertNotIn(name, child_env)

    def _source_repo(self, root: Path) -> tuple[Path, str, str]:
        source = root / "source"
        source.mkdir()
        _run(["git", "init", str(source)])
        tracked = source / "node.txt"
        tracked.write_text("first\n", encoding="utf-8")
        _run(["git", "-C", str(source), "add", "node.txt"])
        _run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Krea Test",
                "-c",
                "user.email=krea@example.invalid",
                "commit",
                "-m",
                "first",
            ]
        )
        first = _git(source, "rev-parse", "HEAD")
        tracked.write_text("second\n", encoding="utf-8")
        _run(["git", "-C", str(source), "add", "node.txt"])
        _run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Krea Test",
                "-c",
                "user.email=krea@example.invalid",
                "commit",
                "-m",
                "second",
            ]
        )
        return source, first, _git(source, "rev-parse", "HEAD")

    def _invoke(
        self,
        repo: Path,
        revision: str,
        destination: Path,
        *,
        patch_path: Path | None = None,
        patched_file: str = "node.txt",
        patched_sha256: str = "",
        patch_sha256: str = "",
    ) -> subprocess.CompletedProcess[str]:
        patch_args = ""
        if patch_path is not None:
            patch_args = (
                f" -PatchPath {_ps_quote(patch_path)}"
                f" -PatchedFile '{patched_file}'"
                f" -PatchedSha256 '{patched_sha256}'"
                f" -PatchSha256 '{patch_sha256}'"
            )
        body = "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f". {_ps_quote(HELPER)}",
                (
                    "Install-KreaDeforumCheckout "
                    f"-Repository {_ps_quote(repo)} "
                    f"-Revision '{revision}' "
                    f"-Destination {_ps_quote(destination)}"
                    f"{patch_args}"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "invoke.ps1"
            script.write_text(body, encoding="utf-8")
            return subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                env=_subprocess_env(),
            )

    @staticmethod
    def _simple_patch(root: Path) -> tuple[Path, str, str]:
        patch = root / "owned.patch"
        patch.write_text(
            "--- a/node.txt\n+++ b/node.txt\n"
            "@@ -1 +1 @@\n-first\n+patched\n",
            encoding="utf-8",
            newline="\n",
        )
        digest = hashlib.sha256(b"patched\n").hexdigest()
        patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        return patch, digest, patch_digest

    def test_missing_destination_checks_out_exact_revision_detached(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "missing"

            result = self._invoke(source, requested, destination)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(_git(destination, "rev-parse", "HEAD"), requested)
            self.assertEqual(_git(destination, "rev-parse", "--abbrev-ref", "HEAD"), "HEAD")

    def test_clean_checkout_moves_to_exact_revision_detached(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, other = self._source_repo(root)
            destination = root / "checkout"
            _run(["git", "clone", str(source), str(destination)])
            self.assertEqual(_git(destination, "rev-parse", "HEAD"), other)

            result = self._invoke(source, requested, destination)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(_git(destination, "rev-parse", "HEAD"), requested)
            self.assertEqual(_git(destination, "rev-parse", "--abbrev-ref", "HEAD"), "HEAD")

    def test_existing_checkout_fetches_requested_repository_not_origin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_a_root = root / "a"
            source_b_root = root / "b"
            source_a_root.mkdir()
            source_b_root.mkdir()
            source_a, _, _ = self._source_repo(source_a_root)
            source_b, _, _ = self._source_repo(source_b_root)
            (source_b / "node.txt").write_text("only source b\n", encoding="utf-8")
            _run(["git", "-C", str(source_b), "add", "node.txt"])
            _run(
                [
                    "git",
                    "-C",
                    str(source_b),
                    "-c",
                    "user.name=Krea Test",
                    "-c",
                    "user.email=krea@example.invalid",
                    "commit",
                    "-m",
                    "source b only",
                ]
            )
            requested = _git(source_b, "rev-parse", "HEAD")
            destination = root / "checkout"
            _run(["git", "clone", str(source_a), str(destination)])

            result = self._invoke(source_b, requested, destination)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(_git(destination, "rev-parse", "HEAD"), requested)
            self.assertEqual(_git(destination, "rev-parse", "--abbrev-ref", "HEAD"), "HEAD")
            self.assertEqual(_git(destination, "remote", "get-url", "origin"), str(source_a))

    def test_dirty_checkout_fails_without_changing_head_or_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            _run(["git", "clone", str(source), str(destination)])
            original_head = _git(destination, "rev-parse", "HEAD")
            dirty_content = "local work must survive\n"
            (destination / "node.txt").write_text(dirty_content, encoding="utf-8")

            result = self._invoke(source, requested, destination)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("local changes", result.stdout + result.stderr)
            self.assertIn("commit, stash, or move", (result.stdout + result.stderr).lower())
            self.assertEqual(_git(destination, "rev-parse", "HEAD"), original_head)
            self.assertEqual((destination / "node.txt").read_text(encoding="utf-8"), dirty_content)

    def test_non_git_destination_fails_without_changing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")

            result = self._invoke(source, requested, destination)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a git checkout", (result.stdout + result.stderr).lower())
            self.assertIn("move or remove", (result.stdout + result.stderr).lower())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

    def test_owned_patch_first_apply_and_idempotent_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            patch, digest, patch_digest = self._simple_patch(root)

            first = self._invoke(
                source, requested, destination,
                patch_path=patch, patched_sha256=digest,
                patch_sha256=patch_digest,
            )
            second = self._invoke(
                source, requested, destination,
                patch_path=patch, patched_sha256=digest,
                patch_sha256=patch_digest,
            )

            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual((destination / "node.txt").read_text(), "patched\n")
            self.assertEqual(_git(destination, "status", "--porcelain"), "M node.txt")

    def test_owned_patch_rejects_tampered_patched_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            patch, digest, patch_digest = self._simple_patch(root)
            self.assertEqual(
                self._invoke(
                    source, requested, destination,
                    patch_path=patch, patched_sha256=digest,
                    patch_sha256=patch_digest,
                ).returncode,
                0,
            )
            (destination / "node.txt").write_text("tampered\n")

            result = self._invoke(
                source, requested, destination,
                patch_path=patch, patched_sha256=digest,
                patch_sha256=patch_digest,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("patched hash", (result.stdout + result.stderr).lower())

    def test_owned_patch_rejects_wrong_upstream_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, _, requested = self._source_repo(root)
            destination = root / "checkout"
            patch, digest, patch_digest = self._simple_patch(root)

            result = self._invoke(
                source, requested, destination,
                patch_path=patch, patched_sha256=digest,
                patch_sha256=patch_digest,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("patch", (result.stdout + result.stderr).lower())

    def test_owned_patch_rejects_tampered_patch_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            patch, digest, patch_digest = self._simple_patch(root)
            patch.write_text(patch.read_text() + "\n# tampered\n")

            result = self._invoke(
                source, requested, destination,
                patch_path=patch,
                patched_sha256=digest,
                patch_sha256=patch_digest,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("patch artifact hash", (
                result.stdout + result.stderr
            ).lower())

    def test_owned_patch_rejects_patch_touching_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            patch, digest, _ = self._simple_patch(root)
            patch.write_text(
                patch.read_text()
                + "--- /dev/null\n+++ b/extra.txt\n"
                + "@@ -0,0 +1 @@\n+extra\n",
                newline="\n",
            )
            patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()

            result = self._invoke(
                source, requested, destination,
                patch_path=patch,
                patched_sha256=digest,
                patch_sha256=patch_digest,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one modified file", (
                result.stdout + result.stderr
            ).lower())

    def test_owned_patch_rerun_rejects_extra_untracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, requested, _ = self._source_repo(root)
            destination = root / "checkout"
            patch, digest, patch_digest = self._simple_patch(root)
            self.assertEqual(
                self._invoke(
                    source, requested, destination,
                    patch_path=patch,
                    patched_sha256=digest,
                    patch_sha256=patch_digest,
                ).returncode,
                0,
            )
            (destination / "extra.txt").write_text("extra")

            result = self._invoke(
                source, requested, destination,
                patch_path=patch,
                patched_sha256=digest,
                patch_sha256=patch_digest,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("local changes", (
                result.stdout + result.stderr
            ).lower())

    def test_provisioner_pins_repository_patch_and_exact_hash(self) -> None:
        installer = INSTALLER.read_text(encoding="utf-8")
        patch = CHUNK_PATCH.read_text(encoding="utf-8")

        self.assertIn("kreadeforum-krea2-chunking.patch", installer)
        self.assertIn(PATCHED_ANIMATOR_SHA256, installer)
        self.assertIn(hashlib.sha256(CHUNK_PATCH.read_bytes()).hexdigest(), installer)
        self.assertIn("frame_offset", patch)
        self.assertIn("seed_plan", patch)

    def test_provisioner_prewarm_isolated_midas_and_writes_marker_on_success(self) -> None:
        installer = INSTALLER.read_text(encoding="utf-8")
        prewarm = (ROOT / "scripts" / "prewarm_midas.py").read_text(encoding="utf-8")

        self.assertIn("prewarm_midas.py", installer)
        self.assertIn("MiDaS 3D setup warning", installer)
        self.assertIn("torch.hub.load", prewarm)
        self.assertIn("MiDaS_small", prewarm)
        self.assertIn("krea-midas-small-ready.json", prewarm)
        self.assertIn("os.replace", prewarm)
        dependency = prewarm.index("rwightman/gen-efficientnet-pytorch")
        midas = prewarm.index('"intel-isl/MiDaS", "MiDaS_small"')
        self.assertLess(dependency, midas)
        self.assertIn("trust_repo=True", prewarm[dependency:midas])


if __name__ == "__main__":
    unittest.main()
