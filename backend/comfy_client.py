"""Low-level transport for driving a local ComfyUI server.

The Krea 2 backend used to run native PyTorch inference. It now acts as an
adapter: it builds a ComfyUI prompt graph (see comfy_workflows.py), submits it
to ComfyUI over HTTP, listens on the ComfyUI websocket for live progress, and
collects the rendered PNG bytes (via the SaveImageWebsocket node, with a
/history + /view fallback).

Only this module talks HTTP/WS to ComfyUI. Everything above it works with the
existing GenerationRequest shape and the (results, seed, filenames,
lora_reports, metadata) contract expected by main.py.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Callable, Optional

import requests

try:  # websocket-client (sync). Optional: we degrade to HTTP polling without it.
    import websocket  # type: ignore
except Exception:  # pragma: no cover - import guard
    websocket = None  # type: ignore

logger = logging.getLogger("krea2.comfy")

# Node id used for the SaveImageWebsocket sink in every graph we build.
WS_IMAGE_NODE = "save_ws"

ProgressCb = Optional[Callable[[int, int], None]]


def comfy_base_url() -> str:
    return os.environ.get("KREA_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


def _ws_base(http_base: str) -> str:
    return http_base.replace("https://", "wss://").replace("http://", "ws://")


class ComfyUnavailable(RuntimeError):
    """Raised when the ComfyUI server cannot be reached."""


class ComfyExecutionError(RuntimeError):
    """Raised when ComfyUI rejects a prompt or fails while executing it."""


def comfy_available(timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f"{comfy_base_url()}/system_stats", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def comfy_system_stats(timeout: float = 5.0) -> dict:
    r = requests.get(f"{comfy_base_url()}/system_stats", timeout=timeout)
    r.raise_for_status()
    return r.json()


def free_comfy_vram(unload_models: bool = True, free_memory: bool = True, timeout: float = 30.0) -> bool:
    """Ask ComfyUI to unload its models and free VRAM (POST /free).

    Used before running a heavy standalone runtime (PiD, SeedVR2) so it never
    has to share the GPU with ComfyUI's loaded Krea model.
    """
    try:
        r = requests.post(
            f"{comfy_base_url()}/free",
            json={"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        logger.warning("ComfyUI /free request failed", exc_info=True)
        return False


def interrupt_comfy(timeout: float = 10.0) -> bool:
    """Interrupt whatever prompt ComfyUI is currently running (best-effort).

    ComfyUI's /interrupt is global: it stops the in-flight prompt, which makes the
    active run's websocket loop raise, so the queued worker can finish and move on.
    """
    try:
        r = requests.post(f"{comfy_base_url()}/interrupt", timeout=timeout)
        return r.status_code == 200
    except Exception:
        logger.debug("ComfyUI interrupt request failed", exc_info=True)
        return False


def object_info(class_type: str | None = None, timeout: float = 30.0) -> dict:
    """Return ComfyUI node signatures (all nodes, or a single class)."""
    url = f"{comfy_base_url()}/object_info"
    if class_type:
        url += f"/{class_type}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


class ComfyClient:
    """A single ComfyUI session (one client_id) that can run prompt graphs."""

    def __init__(self, base_url: str | None = None):
        self.base = (base_url or comfy_base_url()).rstrip("/")
        self.client_id = uuid.uuid4().hex

    # -- HTTP helpers ----------------------------------------------------
    def _post_prompt(self, graph: dict) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        try:
            r = requests.post(f"{self.base}/prompt", json=payload, timeout=60)
        except requests.RequestException as exc:
            raise ComfyUnavailable(f"Could not reach ComfyUI at {self.base}: {exc}") from exc
        if r.status_code != 200:
            # ComfyUI returns 400 with a structured node_errors payload.
            raise ComfyExecutionError(f"ComfyUI rejected the prompt ({r.status_code}): {r.text[:4000]}")
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyExecutionError(f"ComfyUI did not return a prompt_id: {data}")
        return prompt_id

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.base}/interrupt", timeout=10)
        except Exception:
            logger.debug("ComfyUI interrupt request failed", exc_info=True)

    def get_history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.base}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def view_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(f"{self.base}/view", params=params, timeout=120)
        r.raise_for_status()
        return r.content

    # -- Execution -------------------------------------------------------
    def run(
        self,
        graph: dict,
        progress_cb: ProgressCb = None,
        image_node_id: str = WS_IMAGE_NODE,
        timeout: int = 1800,
    ) -> list[bytes]:
        """Submit a graph, stream progress, and return a list of PNG byte blobs."""
        if not comfy_available():
            raise ComfyUnavailable(
                f"ComfyUI is not responding at {self.base}. Start it with ComfyUI/run_comfyui.bat."
            )
        if websocket is None:
            logger.warning("websocket-client not installed; falling back to HTTP polling (no live progress).")
            return self._run_polling(graph, progress_cb, timeout)
        return self._run_ws(graph, progress_cb, image_node_id, timeout)

    def _run_ws(self, graph, progress_cb, image_node_id, timeout) -> list[bytes]:
        ws = websocket.WebSocket()
        try:
            ws.connect(f"{_ws_base(self.base)}/ws?clientId={self.client_id}", timeout=30)
        except Exception as exc:
            logger.warning("ComfyUI websocket connect failed (%s); polling instead.", exc)
            return self._run_polling(graph, progress_cb, timeout)
        ws.settimeout(timeout)
        try:
            prompt_id = self._post_prompt(graph)
            images: list[bytes] = []
            current_node = ""
            deadline = time.time() + timeout
            while True:
                if time.time() > deadline:
                    self.interrupt()
                    raise ComfyExecutionError("ComfyUI generation timed out.")
                out = ws.recv()
                if isinstance(out, (bytes, bytearray)):
                    # Binary frame: 4-byte event type + 4-byte image format, then PNG.
                    if current_node == image_node_id:
                        images.append(bytes(out[8:]))
                    continue
                try:
                    msg = json.loads(out)
                except (ValueError, TypeError):
                    continue
                mtype = msg.get("type")
                data = msg.get("data", {}) or {}
                if mtype == "executing":
                    if data.get("prompt_id") == prompt_id:
                        node = data.get("node")
                        if node is None:
                            break  # execution complete
                        current_node = node
                elif mtype == "progress":
                    if progress_cb:
                        value = int(data.get("value", 0) or 0)
                        maximum = int(data.get("max", 1) or 1)
                        try:
                            progress_cb(value, maximum)
                        except Exception:
                            logger.debug("progress_cb raised", exc_info=True)
                elif mtype == "execution_error":
                    raise ComfyExecutionError(
                        "ComfyUI execution error: " + json.dumps(data)[:4000]
                    )
                elif mtype == "execution_interrupted":
                    raise ComfyExecutionError("ComfyUI execution was interrupted.")
            if not images:
                images = self._collect_from_history(prompt_id)
            return images
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _run_polling(self, graph, progress_cb, timeout) -> list[bytes]:
        prompt_id = self._post_prompt(graph)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                hist = self.get_history(prompt_id)
            except Exception:
                hist = {}
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyExecutionError(f"ComfyUI execution error: {json.dumps(status)[:4000]}")
                return self._collect_from_history(prompt_id)
            time.sleep(1.0)
        self.interrupt()
        raise ComfyExecutionError("ComfyUI generation timed out (polling).")

    def _collect_from_history(self, prompt_id: str) -> list[bytes]:
        hist = self.get_history(prompt_id).get(prompt_id, {})
        outputs = hist.get("outputs", {}) or {}
        images: list[bytes] = []
        for node_out in outputs.values():
            for im in node_out.get("images", []) or []:
                if im.get("type") == "temp" and im.get("filename", "").startswith("."):
                    continue
                try:
                    images.append(
                        self.view_image(im["filename"], im.get("subfolder", ""), im.get("type", "output"))
                    )
                except Exception:
                    logger.debug("view_image failed for %s", im, exc_info=True)
        return images
