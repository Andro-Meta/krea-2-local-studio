# Inpainting Visual Review

## Result
Native inpainting works and produces coherent masked fills. The best visual result in this batch is `object_replace_native_steps12_output.png`; `object_replace_native_output.png` is also coherent at 6 steps.

LanPaint experimental is not production-ready in this implementation. Across object replacement, texture repair, poster detail, and lower-strength variants, it produces high-frequency speckled/noisy patches instead of coherent fills.

## Recommended defaults
- Inpaint method: `native`
- Sampler: `euler_flow`
- Steps: `8` for fast preview, `12` for better quality
- CFG: `0` for Turbo FP8 unless later testing shows CFG helps a specific checkpoint
- Denoise: `1.0` for replacing masked content
- LanPaint: keep experimental/off by default; do not recommend as default
- If exposing LanPaint, warn that it is research/diagnostic only

## Setup assessment
The inpaint path is wired correctly: jobs complete, masks are applied, outputs save, and metadata records the method/sampler. The issue is quality of the independent LanPaint-style inner update, not API wiring.
