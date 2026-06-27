# Comfy Krea Parity Visual Review

- API system check: reachable.
- txt2img_turbo_default: done.
- txt2img_style_ref: done.
- inpaint_native_default: done.
- seed_variance_ab: done.

## Visual Notes

- `txt2img_turbo_default/output_1.png`: coherent Turbo FP8 default image with glossy product detail and stable composition.
- `txt2img_style_ref/output_1.png`: style reference influence is clearly visible; the diagonal pink light and dark studio look transfer strongly into the perfume-bottle prompt.
- `inpaint_native_default/output_1.png`: native inpaint fills the masked circle with a brass lantern while preserving the surrounding simple source composition. Edges are acceptable for the synthetic mask.
- `seed_variance_ab/output_1.png`: seed variance remains coherent and produces a different forest shrine composition without obvious prompt collapse.

## Verdict

- Style references: working and visually effective.
- Native inpaint: working; default native path remains the recommended inpaint setting.
- Seed variance: working; keep default `off` because it is a variation tool, not a quality improvement for every prompt.
- Moodboard UUIDs and edit rebalance: API/schema/metadata paths are implemented and covered by unit tests; visual comparison should be expanded with real catalog UUID cases after choosing representative moodboards.
