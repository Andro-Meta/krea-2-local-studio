import type { GenerationRequest } from '../../api'

export type LabWorkflowId = 'default-star' | 'k2q' | 'seed-variance' | 'edit' | 'upscale' | 'moodboard-style' | 'turbo-4x'
export type SourceInjection = 'none' | 'redraw-reference' | 'img2img-init' | 'inpaint-init-mask'

export interface LabCase {
  id: string
  label: string
  notes: string
  request: GenerationRequest
  sourceInjection?: SourceInjection
  disabledReason?: string
}

export interface LabWorkflow {
  id: LabWorkflowId
  label: string
  description: string
  defaultPrompt: string
  defaultSeed: number
  warnings: string[]
  cases: LabCase[]
}

const bypassLora = {
  name: 'krea2filterbypass3',
  filename: 'krea2filterbypass3.safetensors',
  strength: 6850,
  enabled: true,
  block_filter: 'style_safe' as const,
}

const baseRequest: GenerationRequest = {
  prompt: '',
  negative_prompt: '',
  mode: 'txt2img',
  model_profile: 'krea_turbo',
  diffusion_engine: 'native_int8_convrot',
  checkpoint: 'turbo',
  quantization: 'int8',
  width: 1024,
  height: 1024,
  num_images: 1,
  batch_mode: 'safe_queue',
  seed: 4242,
  sampler: 'er_sde',
  scheduler: 'beta57',
  steps: 8,
  cfg: 1.0,
  mu: 1.15,
  cfg_zero_star: true,
  cfg_zero_init_steps: 1,
  loras: [bypassLora],
  use_prompt_expander: false,
  use_prompt_planner: false,
  seed_variance_preset: 'off',
  god_mode: false,
  mrflow: false,
  turbo_int8_variant: 'redcraft',
}

function req(patch: Partial<GenerationRequest>): GenerationRequest {
  return { ...baseRequest, ...patch, loras: patch.loras ?? baseRequest.loras }
}

const facePrompt = "Extreme close-up cinematic still frame of a woman's face, shot on ARRI Alexa with an 85mm anamorphic lens, realistic skin texture, natural imperfections, dramatic low-key lighting, tack-sharp eyes."
const mallPrompt = "A candid 1990s film photo of three teenagers at a mall food court, paper soda cups and pizza slices, neon signs, denim jackets, natural expressions, documentary realism."
const editPrompt = "Preserve the person and composition, but change the lighting to a cinematic rainy night market scene."

export const labWorkflows: LabWorkflow[] = [
  {
    id: 'default-star',
    label: 'Default Star',
    description: 'Compare the current Turbo star recipe against the RAW recipes we keep coming back to.',
    defaultPrompt: facePrompt,
    defaultSeed: 4242,
    warnings: ['RAW cases are slower than Turbo. Keep this to one prompt while dialing in.'],
    cases: [
      {
        id: 'turbo-star',
        label: 'Turbo default star',
        notes: 'Current fast default: RedCraft INT8, er_sde/beta57, 8 steps, CFG 1, loose hard-lock variance.',
        request: req({
          prompt: facePrompt,
          checkpoint: 'turbo',
          model_profile: 'krea_turbo',
          sampler: 'er_sde',
          scheduler: 'beta57',
          steps: 8,
          cfg: 1.0,
          mu: 1.15,
          cfg_zero_star: true,
          turbo_int8_variant: 'redcraft',
          seed_variance_preset: 'wild',
          seed_variance_algorithm: 'rbg',
          seed_variance_model_type: 'krea2',
          seed_variance_schedule: 'hard_lock',
          seed_variance_cutoff_step: 2,
          seed_variance_cutoff_strength: 1.0,
        }),
      },
      {
        id: 'raw-crisp',
        label: 'RAW crisp',
        notes: 'RAW default from Quick Presets: euler_flow/beta, 28 steps, CFG 4.',
        request: req({
          prompt: facePrompt,
          checkpoint: 'raw',
          model_profile: 'krea_raw',
          sampler: 'euler_flow',
          scheduler: 'beta',
          steps: 28,
          cfg: 4.0,
          mu: null,
          cfg_zero_star: false,
          seed_variance_preset: 'off',
        }),
      },
      {
        id: 'raw-ersde',
        label: 'RAW ER-SDE fast',
        notes: 'Fastest RAW recipe: er_sde/beta, 10 steps, CFG 4.',
        request: req({
          prompt: facePrompt,
          checkpoint: 'raw',
          model_profile: 'krea_raw',
          sampler: 'er_sde',
          scheduler: 'beta',
          steps: 10,
          cfg: 4.0,
          mu: null,
          cfg_zero_star: false,
          seed_variance_preset: 'off',
        }),
      },
      {
        id: 'raw-res2s',
        label: 'RAW RES2S max detail',
        notes: 'Slow RAW detail recipe: res_2s/bong_tangent, 24 steps, CFG 4.',
        request: req({
          prompt: facePrompt,
          checkpoint: 'raw',
          model_profile: 'krea_raw',
          sampler: 'res_2s',
          scheduler: 'bong_tangent',
          steps: 24,
          cfg: 4.0,
          mu: null,
          cfg_zero_star: false,
          seed_variance_preset: 'off',
        }),
      },
    ],
  },
  {
    id: 'k2q',
    label: 'K2Q Lab',
    description: 'Check whether K2Q Turbo LoRAs improve RAW speed or quality versus our defaults.',
    defaultPrompt: mallPrompt,
    defaultSeed: 4242,
    warnings: ['Experimental: prior local A/B did not show a clear speed or quality win.'],
    cases: [
      {
        id: 'turbo-star',
        label: 'Turbo default star',
        notes: 'Fast Turbo reference.',
        request: req({ prompt: mallPrompt, seed: 4242 }),
      },
      {
        id: 'raw-crisp',
        label: 'RAW crisp',
        notes: 'RAW baseline before K2Q LoRA.',
        request: req({ prompt: mallPrompt, checkpoint: 'raw', model_profile: 'krea_raw', sampler: 'euler_flow', scheduler: 'beta', steps: 28, cfg: 4.0, mu: null, cfg_zero_star: false, seed_variance_preset: 'off' }),
      },
      {
        id: 'raw-k2q-r64',
        label: 'RAW + K2Q r64',
        notes: 'Rank 64 Turbo LoRA at 0.6.',
        request: req({
          prompt: mallPrompt,
          checkpoint: 'raw',
          model_profile: 'krea_raw',
          sampler: 'euler_flow',
          scheduler: 'beta',
          steps: 28,
          cfg: 4.0,
          mu: null,
          cfg_zero_star: false,
          seed_variance_preset: 'off',
          loras: [bypassLora, { name: 'k2q_turbo_lora_rank64', filename: 'k2q_turbo_lora_rank64.safetensors', strength: 0.6, enabled: true, block_filter: 'all' }],
        }),
      },
      {
        id: 'raw-k2q-r128',
        label: 'RAW + K2Q r128',
        notes: 'Rank 128 Turbo LoRA at 0.6.',
        request: req({
          prompt: mallPrompt,
          checkpoint: 'raw',
          model_profile: 'krea_raw',
          sampler: 'euler_flow',
          scheduler: 'beta',
          steps: 28,
          cfg: 4.0,
          mu: null,
          cfg_zero_star: false,
          seed_variance_preset: 'off',
          loras: [bypassLora, { name: 'k2q_turbo_lora_rank128', filename: 'k2q_turbo_lora_rank128.safetensors', strength: 0.6, enabled: true, block_filter: 'all' }],
        }),
      },
    ],
  },
  {
    id: 'seed-variance',
    label: 'Seed Variance',
    description: 'Dial the RBG hard-lock cutoff while keeping the same prompt and seed.',
    defaultPrompt: facePrompt,
    defaultSeed: 111,
    warnings: ['Lower cutoff means looser composition and more variation.'],
    cases: [undefined, 4, 3, 2].map((cutoff) => ({
      id: cutoff == null ? 'variance-off' : `cutoff-${cutoff}`,
      label: cutoff == null ? 'Variance off' : `Hard lock ${cutoff}/8`,
      notes: cutoff == null ? 'Anchor image with no variance injection.' : `Wild variance after step ${cutoff}.`,
      request: req({
        prompt: facePrompt,
        seed: 111,
        seed_variance_preset: cutoff == null ? 'off' : 'wild',
        seed_variance_algorithm: 'rbg',
        seed_variance_model_type: 'krea2',
        seed_variance_schedule: 'hard_lock',
        seed_variance_cutoff_step: cutoff ?? 0,
        seed_variance_cutoff_strength: cutoff == null ? 0 : 1,
      }),
    })),
  },
  {
    id: 'edit',
    label: 'Edit Lab',
    description: 'Compare current Redraw/img2img-style edit behavior. Upload a source image before running.',
    defaultPrompt: editPrompt,
    defaultSeed: 260626,
    warnings: ['Current edit modes are not pixel-precise identity editors. Character Edit/NK2E work remains separate.'],
    cases: [
      {
        id: 'redraw-reference',
        label: 'Redraw reference',
        notes: 'Uses uploaded image as a Redraw reference and reinterprets the whole frame.',
        sourceInjection: 'redraw-reference',
        request: req({ prompt: editPrompt, mode: 'redraw', seed: 260626, denoise: 1.0, steps: 8, cfg: 1.0 }),
      },
      {
        id: 'img2img-preserve',
        label: 'Img2img preserve',
        notes: 'Uses uploaded image as init image with moderate denoise.',
        sourceInjection: 'img2img-init',
        request: req({ prompt: editPrompt, mode: 'img2img', seed: 260626, denoise: 0.45, steps: 8, cfg: 1.0 }),
      },
    ],
  },
  {
    id: 'upscale',
    label: 'Upscale Lab',
    description: 'Run experimental upscale workflows from generation outputs.',
    defaultPrompt: 'A gritty cinematic street portrait, natural light, realistic skin texture, sharp eyes, 35mm film look.',
    defaultSeed: 333,
    warnings: ['Upscaler quality is not dialed in yet. Treat this as a scratch lab.'],
    cases: [
      {
        id: 'mrflow-1k',
        label: 'Mr.Flow 1K',
        notes: 'Base render at 512, x2 SR, one-step refine to 1024.',
        request: req({ prompt: 'A gritty cinematic street portrait, natural light, realistic skin texture, sharp eyes, 35mm film look.', seed: 333, mrflow: true, mrflow_upscaler: 'esrgan_x2', width: 1024, height: 1024 }),
      },
      {
        id: 'mrflow-2k',
        label: 'Mr.Flow 2K',
        notes: 'Base render at 1024, x2 SR, one-step refine to 2048.',
        request: req({ prompt: 'A gritty cinematic street portrait, natural light, realistic skin texture, sharp eyes, 35mm film look.', seed: 333, mrflow: true, mrflow_upscaler: 'esrgan_x2', width: 2048, height: 2048 }),
      },
    ],
  },
  {
    id: 'moodboard-style',
    label: 'Moodboard Style',
    description: 'Scratch space for style-reference and moodboard conditioning recipes.',
    defaultPrompt: 'A lone figure in an evocative environment, editorial composition, strong atmosphere.',
    defaultSeed: 5150,
    warnings: ['MVP uses text-only placeholders until specific board/image fixtures are selected.'],
    cases: [
      {
        id: 'text-guidance',
        label: 'Text guidance only',
        notes: 'Baseline text-only prompt with rebalance on.',
        request: req({ prompt: 'A lone figure in an evocative environment, editorial composition, strong atmosphere.', seed: 5150, use_rebalance: true, rebalance_multiplier: 1.0 }),
      },
      {
        id: 'style-locked',
        label: 'Style locked prompt',
        notes: 'Text-only approximation of the style-lock prompt family.',
        request: req({ prompt: 'A lone figure in an evocative environment, editorial composition, strong atmosphere, preserve subject structure, strong cohesive style language, tactile material detail.', seed: 5150, use_rebalance: true, rebalance_multiplier: 1.8 }),
      },
    ],
  },
]

const turbo4xPrompt = "A calm young man around 25-30 years old in a black suit and tie sitting on a red folding chair in the middle of a flat desert, reading a newspaper with a completely unbothered expression. Behind him, a graffiti-covered vintage sedan fully engulfed in massive flames and black smoke. Flat sandy terrain, overcast pale sky. Editorial deadpan humor. Analog film grain, desaturated warm palette."

labWorkflows.push({
  id: 'turbo-4x',
  label: 'Turbo 4X',
  description: 'The exact community "Krea 2 Turbo 4X" workflow (OTU W8A8 + ClownsharK + Impact refine ladder + LatentPixelScale 4X + VAEDeGrid). Runs verbatim; prompt/seed are the editable knobs.',
  defaultPrompt: turbo4xPrompt,
  defaultSeed: 42,
  warnings: [
    'Multi-stage 4X pipeline — expect ~3–4 minutes per run (not seconds). The progress bar advances per stage, so it resets a few times; that is normal.',
    'Runs on int8-convrot Turbo + fp8 encoder so it fits in high-VRAM mode.',
  ],
  cases: [
    {
      id: 'turbo-4x-full',
      label: 'Turbo 4X (verbatim)',
      notes: 'OTU W8A8 Turbo bf16 + Wan VAE. 1st stage ClownsharK res_2s/linear_quadratic, 2nd-stage + refiner Impact denoise-tail passes, 4X LatentPixelScale (4xNomos8kDAT) + VAEDeGrid.',
      request: {
        prompt: turbo4xPrompt,
        negative_prompt: 'low quality, blurry, pixelated, distorted, deformed, bad anatomy, watermark, text, jpeg artifacts',
        mode: 'turbo_4x',
        seed: 42,
        num_images: 1,
        use_prompt_expander: false,
        use_prompt_planner: false,
      },
    },
  ],
})

export function workflowById(id: LabWorkflowId): LabWorkflow {
  return labWorkflows.find(workflow => workflow.id === id) ?? labWorkflows[0]
}
