import { metadataToGenerateParams } from './galleryMetadataImport'

const imported = metadataToGenerateParams({
  prompt: 'a silver tree',
  negative_prompt: 'blur',
  seed: 44,
  mode: 'txt2img',
  checkpoint: 'turbo',
  quantization: 'fp8',
  steps: 8,
  cfg: 0,
  width: 1024,
  height: 768,
  sampler: 'euler_flow',
  scheduler: 'simple',
  inpaint: {
    method: 'lanpaint_experimental',
    lanpaint_inner_steps: 5,
    lanpaint_strength: 0.75,
    lanpaint_lambda: 16,
    lanpaint_step_size: 0.2,
    lanpaint_beta: 1,
    lanpaint_friction: 15,
    lanpaint_early_stop: 1,
    lanpaint_prompt_mode: 'Image First',
  },
  loras: [{ name: 'krea2_darkbrush', filename: 'krea2_darkbrush.safetensors', strength: 0.7, enabled: true }],
  moodboard_ids: [12],
  moodboard_uuids: ['abc'],
  creativity: 'medium',
  image_references: {
    style_fusion_mode: 'preserve_structure',
    style_references: [{ image_b64: 'STYLE_B64', strength: -0.5, token_size: 'high', vision_position: 'after_prompt' }],
  },
  seed_variance: { preset: 'off', strength: 0, protection: 'first_half' },
  rebalance: { enabled: true, mode: 'rms_renorm', preset: 'detail', renormalize: true, multiplier: 5, weights: '1,2' },
  krea_enhancer: { enabled: true, variant: 'capped_delta', strength: 0.8, delta_cap: 0.65 },
  refine: { enabled: true, denoise: 0.2, steps: 4 },
}, 'redraw', 'IMAGE_B64')

imported.mode satisfies 'redraw'
imported.init_image_b64 satisfies string | undefined
imported.loras?.[0].name satisfies string | undefined
imported.sampler satisfies string | undefined
imported.scheduler satisfies string | undefined
imported.inpaint_method satisfies string | undefined
imported.lanpaint_inner_steps satisfies number | undefined
imported.lanpaint_strength satisfies number | undefined
imported.lanpaint_lambda satisfies number | undefined
imported.lanpaint_prompt_mode satisfies string | undefined
imported.creativity satisfies string | undefined
imported.style_references?.[0].strength satisfies number | undefined
imported.style_references?.[0].vision_position satisfies string | undefined
imported.style_fusion_mode satisfies string | undefined
imported.moodboard_uuids?.[0] satisfies string | undefined
imported.seed_variance_protection satisfies string | undefined
imported.rebalance_mode satisfies string | undefined
imported.rebalance_preset satisfies string | undefined
imported.rebalance_renormalize satisfies boolean | undefined
imported.krea_enhancer_delta_cap satisfies number | undefined

const importedT2i = metadataToGenerateParams({ prompt: 'x', mode: 'txt2img' }, 'txt2img', 'IMAGE_B64')
const importedImg2Img = metadataToGenerateParams({ prompt: 'x', mode: 'txt2img' }, 'img2img', 'IMAGE_B64')
const importedInpaint = metadataToGenerateParams({ prompt: 'x', mode: 'txt2img' }, 'inpaint', 'IMAGE_B64')
const importedOutpaint = metadataToGenerateParams({ prompt: 'x', mode: 'txt2img' }, 'outpaint', 'IMAGE_B64')

importedT2i.mode satisfies 'txt2img'
importedImg2Img.mode satisfies 'img2img'
importedInpaint.mode satisfies 'inpaint'
importedOutpaint.mode satisfies 'outpaint'
