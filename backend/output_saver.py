from __future__ import annotations

import base64
import io
import time
import uuid
from pathlib import Path
from typing import Iterable

from PIL import Image


def encode_images(
    images: Iterable[Image.Image],
    outputs_dir: Path,
    *,
    save_outputs: bool = True,
) -> tuple[list[str], list[str]]:
    results: list[str] = []
    filenames: list[str] = []
    for img in images:
        if save_outputs:
            fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
            img.save(str(outputs_dir / fname))
            filenames.append(fname)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results.append(base64.b64encode(buf.getvalue()).decode())
    return results, filenames
