# Performance Notes

These notes capture measured behavior on a Windows RTX 4090 test system using
`torch 2.11.0+cu128`, `transformers 5.12.1`, and the fp8 Turbo checkpoint.

## What Helps

- Use the fp8 Turbo checkpoint for the default workflow. It fits the 24 GB 4090
  path with Qwen encoder offload and is what `install.bat` downloads by default.
- Keep `cfg=0` for Turbo unless you have a specific reason. CFG doubles DiT
  forwards per sampling step.
- Reuse the same prompt, moodboard, and reference settings when iterating seeds.
  Krea caches final conditioning tensors on CPU, so repeated runs avoid the
  expensive Qwen CPU/GPU transfer.
- Reuse reference-image moodboards when possible. The Qwen vision processor is
  cached after first use.

Measured repeated-prompt result:

- First 1024 Turbo run: about 20.6 seconds.
- Repeated same prompt/settings with a new seed: about 9.1 seconds.
- Speedup: about 2.3x.

## What Was Tested And Not Adopted

- Updating system CUDA/cuDNN separately: not useful for pip PyTorch wheels. The
  CUDA/cuDNN runtime comes from the installed Torch wheel.
- Switching away from cuDNN SDPA on Windows: not useful here. Flash attention was
  unavailable in the tested PyTorch Windows wheel, while math attention was much
  slower.
- Enabling TF32 matmul globally: did not improve 1024 Turbo timing in the tested
  path.
- Batched VAE decode: measured about the same as one-image-at-a-time decode for
  the Qwen-Image VAE path.

## Remaining High-Impact Ideas

- Avoid duplicate full Qwen3-VL loads between local prompt expansion and Krea
  conditioning.
- Reduce fp8 on-the-fly dequant overhead in the DiT.
- Avoid full decode to PIL and re-encode for the optional detail refine pass.
- Cache or merge LoRA state for repeated generations with the same adapters.
