from __future__ import annotations

import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import Iterable

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _pnginfo(metadata: dict | None) -> PngInfo | None:
    if not metadata:
        return None
    info = PngInfo()
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    info.add_text("krea2_metadata", payload)
    info.add_text("parameters", payload)
    return info


def encode_images(
    images: Iterable[Image.Image],
    outputs_dir: Path,
    *,
    save_outputs: bool = True,
    metadata: list[dict] | dict | None = None,
) -> tuple[list[str], list[str]]:
    results: list[str] = []
    filenames: list[str] = []
    metadata_list = metadata if isinstance(metadata, list) else None
    for index, img in enumerate(images):
        item_metadata = metadata_list[index] if metadata_list and index < len(metadata_list) else metadata
        item_metadata = dict(item_metadata) if isinstance(item_metadata, dict) else None
        if save_outputs:
            fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
            if item_metadata is not None:
                item_metadata["filename"] = fname
            pnginfo = _pnginfo(item_metadata)
            final_path = outputs_dir / fname
            tmp_path = outputs_dir / f".{fname}.tmp"
            img.save(str(tmp_path), format="PNG", pnginfo=pnginfo)
            tmp_path.replace(final_path)
            filenames.append(fname)
        else:
            pnginfo = _pnginfo(item_metadata)
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=pnginfo)
        results.append(base64.b64encode(buf.getvalue()).decode())
    return results, filenames
