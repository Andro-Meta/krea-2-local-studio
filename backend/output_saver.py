from __future__ import annotations

import base64
import io
import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def safe_dir_name(name: str | None) -> str:
    """Sanitize a username into a safe single-level folder name (or '' if unusable)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "").strip())
    cleaned = cleaned.strip("._")
    if not cleaned or cleaned in (".", ".."):
        return ""
    return cleaned[:64]


def _pnginfo(metadata: dict | None, comfy_graph: dict | None = None) -> PngInfo | None:
    if not metadata and not comfy_graph:
        return None
    info = PngInfo()
    if metadata:
        payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        info.add_text("krea2_metadata", payload)
        info.add_text("parameters", payload)
    # ComfyUI reconstructs the node graph on drag-drop. It tries the "workflow"
    # (UI litegraph) chunk first and only falls back to the "prompt" (API-format)
    # chunk via loadApiJson. We only have the API graph, so we MUST write it under
    # "prompt" only -- writing API JSON under "workflow" would make ComfyUI take
    # the UI branch and fail without falling back.
    if comfy_graph:
        try:
            info.add_text("prompt", json.dumps(comfy_graph, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
    return info


def encode_images(
    images: Iterable[Image.Image],
    outputs_dir: Path,
    *,
    save_outputs: bool = True,
    metadata: list[dict] | dict | None = None,
    comfy_graphs: list[dict] | dict | None = None,
    subdir: str | None = None,
    output_file_cb: Callable[[str], None] | None = None,
) -> tuple[list[str], list[str]]:
    """Encode (and optionally save) images. When ``subdir`` (a username) is given,
    files are written into ``outputs_dir/<username>/`` (created on demand) and the
    returned filename is the relative path ``<username>/<name>``. Names are
    ``YYYYMMDD-HHMMSS_<username>_<hash>.png`` so they sort by recency and carry the
    date + owner. Returns (base64_results, relative_filenames)."""
    results: list[str] = []
    filenames: list[str] = []
    metadata_list = metadata if isinstance(metadata, list) else None
    graph_list = comfy_graphs if isinstance(comfy_graphs, list) else None
    user_tag = safe_dir_name(subdir)
    target_dir = outputs_dir / user_tag if user_tag else outputs_dir
    if save_outputs:
        target_dir.mkdir(parents=True, exist_ok=True)
    for index, img in enumerate(images):
        item_metadata = metadata_list[index] if metadata_list and index < len(metadata_list) else metadata
        item_metadata = dict(item_metadata) if isinstance(item_metadata, dict) else None
        item_graph = graph_list[index] if graph_list and index < len(graph_list) else (comfy_graphs if isinstance(comfy_graphs, dict) else None)
        if save_outputs:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            tag = f"{user_tag}_" if user_tag else ""
            name = f"{stamp}_{tag}{uuid.uuid4().hex[:8]}.png"
            rel = f"{user_tag}/{name}" if user_tag else name
            if item_metadata is not None:
                item_metadata["filename"] = rel
            pnginfo = _pnginfo(item_metadata, item_graph)
            final_path = target_dir / name
            tmp_path = target_dir / f".{name}.tmp"
            if output_file_cb is not None:
                tmp_rel = (
                    f"{user_tag}/{tmp_path.name}" if user_tag else tmp_path.name
                )
                output_file_cb(tmp_rel)
            img.save(str(tmp_path), format="PNG", pnginfo=pnginfo)
            tmp_path.replace(final_path)
            filenames.append(rel)
            if output_file_cb is not None:
                output_file_cb(rel)
        else:
            pnginfo = _pnginfo(item_metadata, item_graph)
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=pnginfo)
        results.append(base64.b64encode(buf.getvalue()).decode())
    return results, filenames
