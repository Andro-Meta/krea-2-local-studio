from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path
from typing import Callable

from PIL import Image

from generation_metadata import build_generation_metadata
from output_saver import encode_images
from quality_assets import asset_by_id, asset_installed
from settings import OUTPUTS_DIR

logger = logging.getLogger(__name__)

FLUX_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
_PIPE_CACHE = {"pipe": None}
_lock = threading.Lock()


def _b64_to_pil(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def _mask_b64_to_pil(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("L")


def _align16(value: int) -> int:
    return max(16, ((int(value) + 15) // 16) * 16)


def flux_fill_call_kwargs(
    *,
    prompt: str,
    image: Image.Image,
    mask: Image.Image,
    width: int,
    height: int,
    seed: int,
    steps: int,
) -> dict:
    # FLUX Fill does not use the img2img "strength" concept. The public
    # diffusers examples use high guidance and 50 steps for strict fill work.
    return {
        "prompt": prompt,
        "image": image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
        "mask_image": mask.convert("L").resize((width, height), Image.Resampling.BILINEAR),
        "height": height,
        "width": width,
        "guidance_scale": 30.0,
        "num_inference_steps": max(50, int(steps or 0)),
        "max_sequence_length": 512,
        "generator_seed": int(seed),
    }


def flux_fill_installed() -> bool:
    return asset_installed(asset_by_id("flux_fill"))


def unload_flux_fill() -> None:
    with _lock:
        _PIPE_CACHE["pipe"] = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Could not clear CUDA cache after unloading FLUX Fill", exc_info=True)


def _load_pipeline():
    if _PIPE_CACHE["pipe"] is not None:
        return _PIPE_CACHE["pipe"]

    spec = asset_by_id("flux_fill")
    source = str(spec.local_path) if asset_installed(spec) else FLUX_FILL_REPO

    import torch
    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(source, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
    _PIPE_CACHE["pipe"] = pipe
    return pipe


def generate_flux_fill(
    request,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    save_outputs: bool = True,
) -> tuple[list[str], int, list[str], list[dict]]:
    if not request.init_image_b64 or not request.mask_b64:
        raise ValueError("FLUX Fill requires init_image_b64 and mask_b64.")

    import torch

    seed = int(request.seed) if int(request.seed) >= 0 else int(torch.randint(0, 2**31, (1,)).item())
    width = _align16(int(request.width))
    height = _align16(int(request.height))
    image = _b64_to_pil(request.init_image_b64)
    mask = _mask_b64_to_pil(request.mask_b64)

    kwargs = flux_fill_call_kwargs(
        prompt=request.prompt,
        image=image,
        mask=mask,
        width=width,
        height=height,
        seed=seed,
        steps=int(request.steps),
    )
    generator_seed = kwargs.pop("generator_seed")
    kwargs["generator"] = torch.Generator("cpu").manual_seed(generator_seed)

    def callback(_pipe, step: int, timestep, callback_kwargs: dict):
        if progress_cb:
            progress_cb(step + 1, kwargs["num_inference_steps"])
        return callback_kwargs

    with _lock:
        pipe = _load_pipeline()
        result = pipe(
            **kwargs,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=[],
        )

    images = [img.convert("RGB") for img in result.images]
    metadata = [
        build_generation_metadata(
            request,
            base_seed=seed,
            image_index=i,
            filename="",
            resolved_provider="flux_fill",
            runtime={"provider": "flux_fill"},
            model_runtime={"loaded_checkpoint_path": "black-forest-labs/FLUX.1-Fill-dev"},
        )
        for i in range(len(images))
    ]
    results, filenames = encode_images(images, OUTPUTS_DIR, save_outputs=save_outputs, metadata=metadata)
    metadata = [
        {**item, "filename": filenames[i] if i < len(filenames) else item.get("filename", "")}
        for i, item in enumerate(metadata)
    ]
    return results, seed, filenames, [], metadata
