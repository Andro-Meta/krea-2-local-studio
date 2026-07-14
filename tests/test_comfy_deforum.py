from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from support import mock_atomic_cancel_capability

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def patched_object_info(comfy_deforum):
    result = {name: {} for name in comfy_deforum.REQUIRED_NODES}
    result["KreaDeforumAnimator"] = {
        "input": {
            "required": {},
            "optional": {
                name: ["STRING", {}]
                for name in (
                    "frame_offset",
                    "init_image_is_previous",
                    "reference_image",
                    "seed_plan",
                    "hybrid_video_has_context",
                    "prompt_blend_frames",
                )
            },
        }
    }
    result["KreaDeforumChunkAdapterVersion"] = {
        "input": {
            "required": {
                "version": [
                    ["krea2-chunking-v2"],
                    {"default": "krea2-chunking-v2"},
                ]
            }
        }
    }
    return result


class KreaDeforumStatusTests(unittest.TestCase):
    def test_midas_readiness_requires_controlled_marker_and_cache_paths(self) -> None:
        import comfy_deforum

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "krea-midas-small-ready.json"
            with patch.object(comfy_deforum, "MIDAS_READINESS_MARKER", marker):
                missing = comfy_deforum.midas_readiness()
                self.assertFalse(missing["midas_ready"])
                self.assertIn("install", missing["midas_reason"].lower())

                weights = root / "midas_v21_small_256.pt"
                repo = root / "intelisl_midas_main"
                weights.write_bytes(b"weights")
                repo.mkdir()
                marker.write_text(json.dumps({
                    "version": 1,
                    "model": "MiDaS_small",
                    "weights_path": str(weights),
                    "hub_repo_path": str(repo),
                }), encoding="utf-8")
                ready = comfy_deforum.midas_readiness()

            self.assertTrue(ready["midas_ready"])
            self.assertIn("verified", ready["midas_reason"].lower())

    def test_status_reports_midas_readiness_without_loading_model(self) -> None:
        import comfy_deforum

        object_info = patched_object_info(comfy_deforum)
        with (
            patch.object(comfy_deforum.comfy_client, "object_info", return_value=object_info),
            patch.object(comfy_deforum, "midas_readiness", return_value={
                "midas_ready": False,
                "midas_reason": "MiDaS setup marker is missing.",
            }),
        ):
            result = comfy_deforum.status(force_refresh=True, stale_ttl=-1.0)

        self.assertFalse(result["midas_ready"])
        self.assertEqual(result["midas_reason"], "MiDaS setup marker is missing.")

    def test_status_available_when_all_required_classes_are_registered(self) -> None:
        import comfy_deforum

        object_info = patched_object_info(comfy_deforum)
        with (
            patch.object(
                comfy_deforum.comfy_client,
                "object_info",
                return_value=object_info,
            ),
            patch.object(
                comfy_deforum,
                "midas_readiness",
                return_value={
                    "midas_ready": False,
                    "midas_reason": (
                        "MiDaS 3D setup is incomplete. Run install.bat, "
                        "then restart ComfyUI."
                    ),
                },
            ),
        ):
            result = comfy_deforum.status(force_refresh=True, stale_ttl=-1.0)

        self.assertEqual(
            comfy_deforum.REQUIRED_NODES,
            (
                "KreaDeforumAnimator",
                "KreaDeforumSaveVideo",
                "KreaDeforumSchedulePreview",
                "KreaDeforumChunkAdapterVersion",
            ),
        )
        self.assertEqual(
            result,
            {
                "available": True,
                "missing_nodes": [],
                "revision": "49bb6752ab045fac25652f3e9207d4706bf5c646",
                "external": True,
                "license": "unspecified",
                "patch_version": "krea2-chunking-v2",
                "patched_animator_sha256": "2dd533428c84809c5768951d414b7edac451c4c9ba09e1ab6ced132f713f4461",
                "patch_sha256": "2ef30ed45db588cad4472ac8edffce00f9a89bf249b9c4460e19e213df7f0978",
                "probe_failed": False,
                "stale": False,
                "incompatible_capabilities": [],
                "midas_ready": False,
                "midas_reason": "MiDaS 3D setup is incomplete. Run install.bat, then restart ComfyUI.",
            },
        )

    def test_status_probes_only_required_node_classes(self) -> None:
        import comfy_deforum

        object_info = patched_object_info(comfy_deforum)
        calls: list[str] = []

        def probe(class_type: str, timeout: float) -> dict:
            calls.append(class_type)
            return {class_type: object_info[class_type]}

        with patch.object(comfy_deforum.comfy_client, "object_info", side_effect=probe):
            result = comfy_deforum.status(
                timeout=1.0,
                force_refresh=True,
                stale_ttl=-1.0,
            )

        self.assertTrue(result["available"])
        self.assertEqual(calls, list(comfy_deforum.REQUIRED_NODES))

    def test_unpatched_class_names_are_incompatible(self) -> None:
        import comfy_deforum

        object_info = {name: {} for name in comfy_deforum.REQUIRED_NODES}
        with patch.object(
            comfy_deforum.comfy_client,
            "object_info",
            return_value=object_info,
        ):
            result = comfy_deforum.status(force_refresh=True)

        self.assertFalse(result["available"])
        self.assertEqual(result["missing_nodes"], [])
        self.assertIn("patched animator input contract", " ".join(
            result["incompatible_capabilities"]
        ))

    def test_status_reports_exact_missing_classes(self) -> None:
        import comfy_deforum

        with patch.object(
            comfy_deforum.comfy_client,
            "object_info",
            return_value={"KreaDeforumSaveVideo": {}},
        ):
            result = comfy_deforum.status(timeout=1.25, force_refresh=True)

        self.assertFalse(result["available"])
        self.assertEqual(
            result["missing_nodes"],
            [
                "KreaDeforumAnimator",
                "KreaDeforumSchedulePreview",
                "KreaDeforumChunkAdapterVersion",
            ],
        )

    def test_status_treats_comfy_errors_as_all_nodes_missing(self) -> None:
        import comfy_deforum

        with patch.object(
            comfy_deforum.comfy_client,
            "object_info",
            side_effect=ConnectionError("ComfyUI is offline"),
        ):
            result = comfy_deforum.status(force_refresh=True, stale_ttl=-1.0)

        self.assertFalse(result["available"])
        self.assertEqual(result["missing_nodes"], [])
        self.assertTrue(result["probe_failed"])

    def test_transient_probe_failure_retains_recent_success(self) -> None:
        import comfy_deforum

        all_nodes = patched_object_info(comfy_deforum)
        with patch.object(
            comfy_deforum.comfy_client,
            "object_info",
            side_effect=lambda class_type, timeout: {
                class_type: all_nodes[class_type]
            },
        ):
            healthy = comfy_deforum.status(force_refresh=True)
        with patch.object(
            comfy_deforum.comfy_client,
            "object_info",
            side_effect=ConnectionError("offline"),
        ):
            stale = comfy_deforum.status(
                force_refresh=True, stale_ttl=60.0
            )

        self.assertTrue(healthy["available"])
        self.assertTrue(stale["available"])
        self.assertTrue(stale["probe_failed"])
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["missing_nodes"], [])

    def test_status_probe_does_not_hold_cache_lock_during_network(self) -> None:
        import comfy_deforum

        entered = 0
        entered_lock = threading.Lock()
        both_entered = threading.Event()

        nodes = patched_object_info(comfy_deforum)

        def probe(class_type: str, **_kwargs):
            nonlocal entered
            with entered_lock:
                entered += 1
                if entered == 2:
                    both_entered.set()
            assert both_entered.wait(2.0)
            return {class_type: nodes[class_type]}

        with (
            patch.object(comfy_deforum.comfy_client, "object_info", side_effect=probe),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(comfy_deforum.status, force_refresh=True)
                for _ in range(2)
            ]
            results = [future.result(timeout=3) for future in futures]

        self.assertEqual(entered, 2 * len(comfy_deforum.REQUIRED_NODES))
        self.assertTrue(all(result["available"] for result in results))

    def test_status_caches_probes_until_forced_refresh(self) -> None:
        import comfy_deforum

        all_nodes = patched_object_info(comfy_deforum)
        probe_round = 0

        def probe(class_type: str, timeout: float) -> dict:
            nonlocal probe_round
            probe_round += 1
            if probe_round <= len(comfy_deforum.REQUIRED_NODES):
                return {class_type: all_nodes[class_type]}
            return {}

        with patch.object(
            comfy_deforum.comfy_client, "object_info", side_effect=probe
        ) as object_info:
            first = comfy_deforum.status(force_refresh=True, cache_ttl=60.0)
            cached = comfy_deforum.status(cache_ttl=60.0)
            refreshed = comfy_deforum.status(force_refresh=True, cache_ttl=60.0)

        self.assertEqual(cached, first)
        self.assertTrue(first["available"])
        self.assertFalse(refreshed["available"])
        self.assertEqual(
            object_info.call_count,
            2 * len(comfy_deforum.REQUIRED_NODES),
        )

    def test_settings_endpoint_survives_absent_deforum_nodes(self) -> None:
        from fastapi.testclient import TestClient
        import comfy_deforum
        import main

        mock_atomic_cancel_capability(main)
        with (
            patch.object(comfy_deforum.comfy_client, "object_info", return_value={}),
            TestClient(main.app) as client,
        ):
            comfy_deforum.status(force_refresh=True)
            response = client.get("/api/settings")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["krea_deforum"]["missing_nodes"],
            list(comfy_deforum.REQUIRED_NODES),
        )
        self.assertFalse(response.json()["krea_deforum"]["available"])


class SettingsConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_deforum_probe_does_not_block_concurrent_coroutine(self) -> None:
        import main

        heartbeat_at = 0.0

        def slow_status(timeout: float = 5.0) -> dict:
            time.sleep(0.25)
            return {"available": False, "missing_nodes": []}

        async def heartbeat() -> None:
            nonlocal heartbeat_at
            await asyncio.sleep(0.02)
            heartbeat_at = time.monotonic()

        started = time.monotonic()
        with patch.object(main, "krea_deforum_status", side_effect=slow_status) as probe:
            await asyncio.gather(main.get_settings(), heartbeat())

        self.assertLess(heartbeat_at - started, 0.15)
        probe.assert_called_once_with(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
