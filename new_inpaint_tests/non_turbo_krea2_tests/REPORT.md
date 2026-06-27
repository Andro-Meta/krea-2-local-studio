# Non-Turbo Krea 2 Investigation

## Installed assets
- RAW BF16 checkpoint exists: `models/krea2/diffusion_models/krea2_raw_bf16.safetensors`
- Size: about 24.5 GiB / 26.3 GB on disk

## Live test result
A live RAW image generation was not run because the local machine is not safe for RAW BF16 with the current loader.

Real preflight result:
`RAW/BF16 variants need at least ~48GB system RAM with this loader; system has 31.7GB. Use Turbo FP8 here, or run RAW/BF16 on a higher-RAM/offloaded loader.`

The failed RAW load did not unload or break the currently loaded Turbo FP8 model.

## Defaults
Backend direct-API RAW defaults now normalize to:
- steps: 52
- cfg: 3.5
- mu: auto / None
- quantization: bf16

Frontend RAW selection already used:
- steps: 52
- cfg: 3.5
- mu: 0, which backend treats as auto

## Recommendation
- Keep Turbo FP8 as the default for this machine.
- RAW BF16 should be offered as a benchmark/high-RAM mode only.
- Do not attempt RAW BF16 on 32GB RAM with this loader.
- Future improvement: add an offloaded/streaming RAW loader before making RAW user-facing on 32GB machines.
