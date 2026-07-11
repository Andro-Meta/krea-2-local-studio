from __future__ import annotations

import contextlib
import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import share_auth  # noqa: E402
import main  # noqa: E402


class FunnelHealthStateTests(unittest.TestCase):
    def test_requires_three_failed_intervals_and_success_resets_count(self) -> None:
        from runtime_hardening import FunnelHealthState

        state = FunnelHealthState()
        self.assertFalse(state.record_probe(False, now=0))
        self.assertFalse(state.record_probe(False, now=300))
        state.record_probe(True, now=600)
        self.assertEqual(state.failed_intervals, 0)
        self.assertFalse(state.record_probe(False, now=900))
        self.assertFalse(state.record_probe(False, now=1200))
        self.assertTrue(state.record_probe(False, now=1500))

    def test_repair_backoff_increases_and_is_bounded(self) -> None:
        from runtime_hardening import FunnelHealthState

        state = FunnelHealthState()
        self.assertEqual(state.record_repair(now=0), 300)
        self.assertEqual(state.record_repair(now=300), 900)
        self.assertEqual(state.record_repair(now=1200), 1800)
        self.assertEqual(state.record_repair(now=3000), 1800)

        state.failed_intervals = 3
        self.assertFalse(state.repair_due(now=4799))
        self.assertTrue(state.repair_due(now=4800))

    def test_manual_stop_disables_repairs_but_disconnect_repairs_after_three_intervals(self) -> None:
        from runtime_hardening import FunnelHealthMonitor

        monitor = FunnelHealthMonitor(enabled=True)
        monitor.disable()
        self.assertFalse(monitor.observe(False, now=0))
        self.assertFalse(monitor.observe(False, now=300))
        self.assertFalse(monitor.observe(False, now=600))
        self.assertEqual(monitor.state.failed_intervals, 0)

        monitor.enable()
        self.assertFalse(monitor.observe(False, now=900))
        self.assertFalse(monitor.observe(False, now=1200))
        self.assertTrue(monitor.observe(False, now=1500))

    def test_missing_funnel_url_is_an_unhealthy_interval(self) -> None:
        from runtime_hardening import auto_repair_configured, funnel_interval_healthy

        self.assertFalse(funnel_interval_healthy({"running": False, "url": ""}, None))
        self.assertFalse(funnel_interval_healthy({"running": True, "url": ""}, None))
        self.assertFalse(funnel_interval_healthy({"running": True, "url": "https://x"}, False))
        self.assertTrue(funnel_interval_healthy({"running": True, "url": "https://x"}, True))
        self.assertTrue(auto_repair_configured("true"))
        self.assertTrue(auto_repair_configured("1"))
        self.assertFalse(auto_repair_configured("false"))
        self.assertFalse(auto_repair_configured(None))


class BootstrapCredentialTests(unittest.TestCase):
    def test_bootstrap_writes_credential_without_printing_password(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "data" / "private" / "first-admin.json"
            password = "never-print-this-password"
            output = io.StringIO()

            with (
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(output),
                patch("share_auth._restrict_credential_acl"),
            ):
                result = share_auth.bootstrap_first_admin(
                    auth_path,
                    credential_path,
                    username="admin",
                    password=password,
                    generated_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
                )

            self.assertEqual(result, credential_path)
            self.assertTrue(share_auth.verify_user(auth_path, "admin", password))
            credential = json.loads(credential_path.read_text(encoding="utf-8"))
            self.assertEqual(credential["username"], "admin")
            self.assertEqual(credential["password"], password)
            self.assertEqual(credential["generated_at"], "2026-07-11T00:00:00+00:00")
            self.assertNotIn(password, output.getvalue())

    def test_bootstrap_secures_temporary_file_before_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            secured_paths: list[Path] = []

            def secure_before_publish(path: Path) -> None:
                self.assertFalse(credential_path.exists())
                secured_paths.append(path)

            with patch("share_auth._restrict_credential_acl", side_effect=secure_before_publish):
                share_auth.bootstrap_first_admin(
                    auth_path,
                    credential_path,
                    password="correct horse",
                )

            self.assertEqual(len(secured_paths), 1)
            self.assertNotEqual(secured_paths[0], credential_path)
            self.assertTrue(credential_path.exists())
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_bootstrap_does_not_destroy_preexisting_credential_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            credential_path.write_text("existing marker", encoding="utf-8")

            with patch("share_auth._restrict_credential_acl"):
                with self.assertRaises(FileExistsError):
                    share_auth.bootstrap_first_admin(
                        auth_path,
                        credential_path,
                        password="correct horse",
                    )

            self.assertEqual(credential_path.read_text(encoding="utf-8"), "existing marker")
            self.assertEqual(share_auth.load_users(auth_path), {})

    def test_bootstrap_file_failure_does_not_create_admin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            blocker = Path(td) / "not-a-directory"
            blocker.write_text("block", encoding="utf-8")

            with self.assertRaises(OSError):
                with patch("share_auth._restrict_credential_acl"):
                    share_auth.bootstrap_first_admin(
                        auth_path,
                        blocker / "first-admin.json",
                        password="never-print-this-password",
                    )

            self.assertEqual(share_auth.load_users(auth_path), {})

    def test_existing_users_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            share_auth.add_user(auth_path, "existing", "correct horse", role="admin")

            with patch("share_auth._restrict_credential_acl"):
                result = share_auth.bootstrap_first_admin(auth_path, credential_path)

            self.assertIsNone(result)
            self.assertFalse(credential_path.exists())
            self.assertEqual(share_auth.list_users(auth_path), ["existing"])

    def test_successful_bootstrap_login_removes_one_time_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            with patch("share_auth._restrict_credential_acl"):
                share_auth.bootstrap_first_admin(
                    auth_path,
                    credential_path,
                    username="admin",
                    password="correct horse",
                )

            self.assertFalse(
                share_auth.verify_login(
                    auth_path,
                    "admin",
                    "wrong horse",
                    bootstrap_credential_path=credential_path,
                )
            )
            self.assertTrue(credential_path.exists())
            self.assertTrue(
                share_auth.verify_login(
                    auth_path,
                    "admin",
                    "correct horse",
                    bootstrap_credential_path=credential_path,
                )
            )
            self.assertFalse(credential_path.exists())

    def test_windows_acl_grants_only_current_user_and_system(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "credential.json"
            path.write_text("secret", encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(share_auth.os, "name", "nt"),
                patch.dict(share_auth.os.environ, {"USERNAME": "Owner"}, clear=False),
                patch("share_auth.subprocess.run", return_value=completed) as run,
            ):
                share_auth._restrict_credential_acl(path)

            args = run.call_args.args[0]
            self.assertEqual(args[:1], ["icacls"])
            self.assertEqual(Path(args[1]), path)
            self.assertEqual(args[2:4], ["/inheritance:r", "/grant:r"])
            self.assertEqual(args[4:], ["Owner:F", "SYSTEM:F"])

    def test_windows_acl_failure_removes_credential_and_admin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            completed = SimpleNamespace(returncode=5, stdout="", stderr="access denied")

            with (
                patch.object(share_auth.os, "name", "nt"),
                patch.dict(share_auth.os.environ, {"USERNAME": "Owner"}, clear=False),
                patch("share_auth.subprocess.run", return_value=completed),
            ):
                with self.assertRaisesRegex(RuntimeError, "ACL"):
                    share_auth.bootstrap_first_admin(
                        auth_path,
                        credential_path,
                        password="never-print-this-password",
                    )

            self.assertFalse(credential_path.exists())
            self.assertEqual(share_auth.load_users(auth_path), {})

    def test_posix_acl_sets_and_verifies_mode(self) -> None:
        path = Path("credential.json")
        with (
            patch.object(share_auth.os, "name", "posix"),
            patch.object(Path, "chmod") as chmod,
            patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=0o100600)),
        ):
            share_auth._restrict_credential_acl(path)
        chmod.assert_called_once_with(0o600)

    def test_posix_acl_verification_failure_raises(self) -> None:
        path = Path("credential.json")
        with (
            patch.object(share_auth.os, "name", "posix"),
            patch.object(Path, "chmod"),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=0o100644)),
        ):
            with self.assertRaisesRegex(RuntimeError, "0600"):
                share_auth._restrict_credential_acl(path)

    def test_bootstrap_credential_unlink_retries_before_login_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            with patch("share_auth._restrict_credential_acl"):
                share_auth.bootstrap_first_admin(
                    auth_path,
                    credential_path,
                    password="correct horse",
                )

            with patch.object(Path, "unlink", side_effect=[OSError("busy"), None]) as unlink:
                self.assertTrue(
                    share_auth.verify_login(
                        auth_path,
                        "admin",
                        "correct horse",
                        bootstrap_credential_path=credential_path,
                    )
                )
            self.assertEqual(unlink.call_count, 2)

    def test_bootstrap_credential_unlink_failure_denies_login(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            credential_path = Path(td) / "first-admin.json"
            with patch("share_auth._restrict_credential_acl"):
                share_auth.bootstrap_first_admin(
                    auth_path,
                    credential_path,
                    password="correct horse",
                )

            with patch.object(Path, "unlink", side_effect=OSError("busy")):
                with self.assertRaises(share_auth.BootstrapCredentialDeletionError):
                    share_auth.verify_login(
                        auth_path,
                        "admin",
                        "correct horse",
                        bootstrap_credential_path=credential_path,
                    )

    def test_existing_user_login_without_bootstrap_marker_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            auth_path = Path(td) / "share_auth.json"
            share_auth.add_user(auth_path, "existing", "correct horse", role="admin")
            self.assertTrue(
                share_auth.verify_login(
                    auth_path,
                    "existing",
                    "correct horse",
                    bootstrap_credential_path=Path(td) / "missing.json",
                )
            )

    def test_shared_bootstrap_path_override_is_used_for_write_and_login_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            override = root / "elsewhere" / "bootstrap.json"
            default = root / "data" / "private" / "first-admin-credential.json"
            env = {"KREA_BOOTSTRAP_CREDENTIAL_FILE": str(override)}
            resolved = share_auth.resolve_bootstrap_credential_path(root, environ=env)
            self.assertEqual(resolved, override)
            self.assertEqual(
                share_auth.resolve_bootstrap_credential_path(
                    root,
                    marker_override=root / "explicit.json",
                    environ=env,
                ),
                root / "explicit.json",
            )

            with patch("share_auth._restrict_credential_acl"):
                share_auth.bootstrap_first_admin(
                    root / "share_auth.json",
                    resolved,
                    password="correct horse",
                )
            self.assertTrue(override.exists())
            self.assertFalse(default.exists())
            self.assertTrue(
                share_auth.verify_login(
                    root / "share_auth.json",
                    "admin",
                    "correct horse",
                    bootstrap_credential_path=share_auth.resolve_bootstrap_credential_path(root, environ=env),
                )
            )
            self.assertFalse(override.exists())


class BackendComfyCapabilityStartupTests(unittest.IsolatedAsyncioTestCase):
    async def _startup(self, *, reachable: bool, capable: bool):
        queue = Mock()

        def discard(coro):
            coro.close()
            return Mock()

        with (
            patch.object(main, "generation_queue", queue),
            patch.object(main, "init_db", new=AsyncMock()),
            patch.object(main, "init_moderation_db", new=AsyncMock()),
            patch.object(main, "init_moodboard_db", new=AsyncMock()),
            patch.object(
                main, "reconcile_custom_moodboard_storage", new=AsyncMock()
            ),
            patch.object(main, "should_sync_moodboards", new=AsyncMock(return_value=False)),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=reachable),
            patch.object(
                main,
                "comfy_atomic_cancel_available",
                return_value=capable,
                create=True,
            ) as capability,
            patch.object(main.asyncio, "create_task", side_effect=discard),
        ):
            await main.startup()
        return capability

    async def test_reachable_supported_comfy_starts(self) -> None:
        capability = await self._startup(reachable=True, capable=True)
        capability.assert_called_once()

    async def test_reachable_unsupported_comfy_fails_startup(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "too old or mismatched"):
            await self._startup(reachable=True, capable=False)

    async def test_unreachable_comfy_keeps_unavailable_behavior(self) -> None:
        capability = await self._startup(reachable=False, capable=False)
        capability.assert_not_called()

    async def test_startup_unavailable_then_first_recovered_task_succeeds(self):
        await self._startup(reachable=False, capable=False)
        jobs = {
            "warm": {
                "status": "queued",
                "task_kind": main.MODEL_WARMUP,
                "result": None,
                "role": "admin",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(
                main, "comfy_atomic_cancel_available", return_value=True
            ) as capability,
            patch.object(main, "_execute_model_warmup", new=AsyncMock()) as execute,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("warm", {})
        capability.assert_called_once()
        execute.assert_awaited_once()
        self.assertEqual(jobs["warm"]["status"], "done")


class RuntimeScriptContractTests(unittest.TestCase):
    def _run_powershell(self, body: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "test.ps1"
            script.write_text(body, encoding="utf-8")
            return subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                capture_output=True,
                text=True,
                timeout=20,
                env=os.environ.copy(),
            )

    def test_comfy_listener_requires_health_version_and_owned_command_line(self) -> None:
        text = (ROOT / "scripts" / "start_comfyui.ps1").read_text(encoding="utf-8")
        self.assertIn("comfyui_version", text)
        self.assertIn("Win32_Process", text)
        self.assertIn("$main", text)
        self.assertIn("$py", text)
        self.assertIn("unrelated process", text)
        self.assertIn("exit 1", text)

    def test_started_comfy_listener_requires_started_pid_or_descendant(self) -> None:
        text = (ROOT / "scripts" / "start_comfyui.ps1").read_text(encoding="utf-8")
        helper = (ROOT / "scripts" / "comfy_process_validation.ps1").read_text(encoding="utf-8")
        self.assertIn("[int]$StartedPid = 0", helper)
        self.assertIn("ParentProcessId", helper)
        self.assertIn("-StartedPid $process.Id", text)
        self.assertGreaterEqual(text.count("Get-OwnedComfyListener"), 3)
        self.assertGreaterEqual(text.count("Test-StableListenerPid"), 2)
        self.assertIn("started process or its descendant", text)
        self.assertIn("will not kill", text)

    def test_comfy_ownership_helpers_reject_toctou_and_substring_spoof(self) -> None:
        helper = ROOT / "scripts" / "comfy_process_validation.ps1"
        result = self._run_powershell(
            f"""
$ErrorActionPreference = 'Stop'
. '{helper}'
if (-not (Test-StableListenerPid -BeforePid 22 -AfterPid 22)) {{ exit 10 }}
if (Test-StableListenerPid -BeforePid 22 -AfterPid 23) {{ exit 11 }}
$parent = [pscustomobject]@{{ ProcessId=11; ParentProcessId=1; ExecutablePath='C:\\launcher.exe'; CommandLine='launcher' }}
$good = [pscustomobject]@{{ ProcessId=22; ParentProcessId=11; ExecutablePath='C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe'; CommandLine='"C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe" "C:\\repo\\ComfyUI\\main.py" --port 8188' }}
$spoof = [pscustomobject]@{{ ProcessId=23; ParentProcessId=11; ExecutablePath='C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe'; CommandLine='"C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe" --note=C:\\repo\\ComfyUI\\main.py --port 8188' }}
$evil = [pscustomobject]@{{ ProcessId=24; ParentProcessId=11; ExecutablePath='C:\\repo\\ComfyUI\\venv\\Scripts\\evil.exe'; CommandLine='"C:\\repo\\ComfyUI\\venv\\Scripts\\evil.exe" "C:\\repo\\ComfyUI\\main.py" --port 8188' }}
$bare = [pscustomobject]@{{ ProcessId=25; ParentProcessId=11; ExecutablePath='C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe'; CommandLine='"C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe" main.py --port 8188' }}
$lookup = {{ param($id) if ($id -eq 11) {{ return $parent }}; return $null }}
if (-not (Test-ComfyProcessOwnership -Process $good -ExpectedPython 'C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe' -ExpectedMain 'C:\\repo\\ComfyUI\\main.py' -StartedPid 11 -ProcessLookup $lookup)) {{ exit 12 }}
if (Test-ComfyProcessOwnership -Process $spoof -ExpectedPython 'C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe' -ExpectedMain 'C:\\repo\\ComfyUI\\main.py' -StartedPid 11 -ProcessLookup $lookup) {{ exit 13 }}
if (Test-ComfyProcessOwnership -Process $evil -ExpectedPython 'C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe' -ExpectedMain 'C:\\repo\\ComfyUI\\main.py' -StartedPid 11 -ProcessLookup $lookup) {{ exit 14 }}
if (Test-ComfyProcessOwnership -Process $bare -ExpectedPython 'C:\\repo\\ComfyUI\\venv\\Scripts\\python.exe' -ExpectedMain 'C:\\repo\\ComfyUI\\main.py' -StartedPid 11 -ProcessLookup $lookup) {{ exit 15 }}
exit 0
"""
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_comfy_cancel_capability_requires_boolean_contract(self) -> None:
        helper = ROOT / "scripts" / "comfy_process_validation.ps1"
        result = self._run_powershell(
            f"""
$ErrorActionPreference = 'Stop'
. '{helper}'
$good = {{ param($uri, $body) [pscustomobject]@{{ StatusCode=200; Content='{{"cancelled":false}}' }} }}
$badStatus = {{ param($uri, $body) [pscustomobject]@{{ StatusCode=404; Content='{{"cancelled":false}}' }} }}
$badShape = {{ param($uri, $body) [pscustomobject]@{{ StatusCode=200; Content='{{"ok":true}}' }} }}
if (-not (Test-ComfyCancelCapability -BaseUrl 'http://127.0.0.1:8188' -RequestInvoker $good)) {{ exit 20 }}
if (Test-ComfyCancelCapability -BaseUrl 'http://127.0.0.1:8188' -RequestInvoker $badStatus) {{ exit 21 }}
if (Test-ComfyCancelCapability -BaseUrl 'http://127.0.0.1:8188' -RequestInvoker $badShape) {{ exit 22 }}
exit 0
"""
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_startup_and_installer_validate_atomic_cancel_capability(self) -> None:
        start = (ROOT / "scripts" / "start_comfyui.ps1").read_text(encoding="utf-8")
        install = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")
        self.assertGreaterEqual(start.count("Assert-ComfyCancelCapability"), 2)
        self.assertIn("installed ComfyUI is too old or mismatched", start)
        self.assertIn("Test-ComfyCancelRouteSource", install)
        self.assertIn("update/reinstall", install.lower())

    def test_external_comfy_startup_validates_cancel_capability(self) -> None:
        start = (ROOT / "scripts" / "start_comfyui.ps1").read_text(encoding="utf-8")
        external = start[start.index("if ($comfyUrl -and"):start.index("if ($comfyUrl -match")]
        self.assertIn("Get-ComfyHealth -BaseUrl $comfyUrl", external)
        self.assertIn("Test-ComfyCancelCapability", external)
        self.assertIn("exit 1", external)
        self.assertLess(
            external.index("Get-ComfyHealth"),
            external.index("Test-ComfyCancelCapability"),
        )
        self.assertLess(
            external.index("Test-ComfyCancelCapability"),
            external.rindex("exit 0"),
        )

    def test_comfy_start_uses_fully_resolved_main_argument(self) -> None:
        text = (ROOT / "scripts" / "start_comfyui.ps1").read_text(encoding="utf-8")
        self.assertIn('$main = [IO.Path]::GetFullPath((Join-Path $comfyDir "main.py"))', text)
        self.assertIn('@("`"$main`"", "--enable-manager"', text)

    def test_mrflow_helper_archives_outer_clone_with_nested_files(self) -> None:
        helper = ROOT / "scripts" / "mrflow_layout.ps1"
        with tempfile.TemporaryDirectory() as td:
            comfy = Path(td) / "ComfyUI"
            outer = comfy / "custom_nodes" / "Rebels_MrFlow"
            inner = outer / "ComfyUI-Rebels-MrFlow"
            (inner / "nested").mkdir(parents=True)
            (inner / "__init__.py").write_text("# node", encoding="utf-8")
            (outer / "workflow.json").write_text('{"root": true}', encoding="utf-8")
            (outer / "local" / "notes").mkdir(parents=True)
            (outer / "local" / "notes" / "keep.txt").write_text("keep", encoding="utf-8")
            (inner / "nested" / "nested-workflow.json").write_text("{}", encoding="utf-8")
            occupied = comfy / "user" / "__sources" / "Rebels_MrFlow"
            occupied.mkdir(parents=True)
            (occupied / "older.txt").write_text("older", encoding="utf-8")
            script = f"""
$ErrorActionPreference = 'Stop'
. '{helper}'
Finalize-MrFlowLayout -ComfyDir '{comfy}' -CustomNodes '{comfy / "custom_nodes"}' | Out-Null
Finalize-MrFlowLayout -ComfyDir '{comfy}' -CustomNodes '{comfy / "custom_nodes"}' | Out-Null
"""
            result = self._run_powershell(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((comfy / "custom_nodes" / "ComfyUI-Rebels-MrFlow" / "__init__.py").exists())
            self.assertTrue((comfy / "user" / "default" / "workflows" / "Rebels_MrFlow" / "workflow.json").exists())
            self.assertFalse(outer.exists())
            archives = list((comfy / "user" / "__sources").glob("Rebels_MrFlow*"))
            self.assertEqual(len(archives), 2)
            archived_source = comfy / "user" / "__sources" / "Rebels_MrFlow-1"
            self.assertTrue((archived_source / "local" / "notes" / "keep.txt").exists())
            self.assertTrue((archived_source / "ComfyUI-Rebels-MrFlow" / "nested" / "nested-workflow.json").exists())

    def test_login_cleanup_happens_before_session_issuance(self) -> None:
        text = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
        login = text[text.index("async def share_login"):text.index("async def share_logout")]
        self.assertIn("BootstrapCredentialDeletionError", text)
        self.assertIn("status_code=500", login)
        self.assertLess(login.index("verify_login("), login.index("token = secrets.token_urlsafe"))

    def test_sharing_endpoints_control_auto_repair_state(self) -> None:
        text = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
        self.assertIn("KREA_SHARE_AUTO_FUNNEL_ENABLED", text)
        start = text[text.index("async def sharing_funnel_start"):text.index("async def sharing_funnel_repair")]
        repair = text[text.index("async def sharing_funnel_repair"):text.index("async def sharing_funnel_stop")]
        stop = text[text.index("async def sharing_funnel_stop"):text.index("# ---------------------------------------------------------------------------", text.index("async def sharing_funnel_stop"))]
        self.assertIn("_funnel_health_monitor.enable()", start)
        self.assertIn("_funnel_health_monitor.enable()", repair)
        self.assertIn("_funnel_health_monitor.disable()", stop)

    def test_mrflow_workflows_are_preserved_and_outer_clone_removed(self) -> None:
        text = (ROOT / "scripts" / "install_comfyui.ps1").read_text(encoding="utf-8")
        helper = (ROOT / "scripts" / "mrflow_layout.ps1").read_text(encoding="utf-8")
        self.assertIn("Finalize-MrFlowLayout", text)
        self.assertIn("user\\default\\workflows\\Rebels_MrFlow", helper)
        self.assertIn("user\\__sources", helper)
        self.assertIn("Move-Item -Path $outer", helper)
        self.assertNotIn("Remove-Item -Recurse", text + helper)

    def test_bootstrap_secret_is_not_echoed_and_private_path_is_ignored(self) -> None:
        run_bat = (ROOT / "run.bat").read_text(encoding="utf-8")
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertNotIn("FIRST_ADMIN_PASSWORD=", run_bat)
        self.assertNotIn("echo %BOOTSTRAP_LOGIN%", run_bat)
        self.assertIn("bootstrap_share_admin.py", run_bat)
        self.assertIn("data/private/", ignore)
        bootstrap = (ROOT / "scripts" / "bootstrap_share_admin.py").read_text(encoding="utf-8")
        self.assertIn("resolve_bootstrap_credential_path", bootstrap)

    def test_random_sharing_port_does_not_create_firewall_rule(self) -> None:
        run_bat = (ROOT / "run.bat").read_text(encoding="utf-8")
        sharing, local = run_bat.split("\n:local", 1)
        self.assertNotIn("New-NetFirewallRule", sharing)
        self.assertIn("Funnel targets localhost", sharing)
        self.assertIn("-LocalPort 8200", local)


if __name__ == "__main__":
    unittest.main()
