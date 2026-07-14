import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ANIMATE_DEFAULTS,
  ANIMATE_PRESETS,
  MOTION_PRESETS,
  applyMotionPreset,
  buildEndpointSchedule,
  calculateRenderedFrames,
  validateAnimateRequest,
} from './presets.ts'

test('defaults mirror the backend AnimateRequest contract', () => {
  assert.deepEqual(ANIMATE_DEFAULTS, {
    prompt_schedule: '0: a scenic landscape, cinematic lighting',
    negative_prompt: '',
    duration_seconds: 4,
    fps: 12,
    render_frames: null,
    width: 768,
    height: 768,
    steps: 8,
    sampler_name: 'er_sde',
    scheduler: 'simple',
    seed: -1,
    seed_behavior: 'iter',
    animation_mode: '2D',
    border_mode: 'replicate',
    cfg_schedule: '0:(1.0)',
    strength_schedule: '0:(0.65)',
    zoom_schedule: '0:(1.0)',
    angle_schedule: '0:(0)',
    translation_x_schedule: '0:(0)',
    translation_y_schedule: '0:(0)',
    translation_z_schedule: '0:(0)',
    rotation_3d_x_schedule: '0:(0)',
    rotation_3d_y_schedule: '0:(0)',
    rotation_3d_z_schedule: '0:(0)',
    color_coherence: 'Match Frame 0 LAB',
    diffusion_cadence: 1,
    prompt_blend_frames: 0,
    prompt_strength_boost: 0,
    prompt_strength_boost_frames: 4,
    hybrid_strength_schedule: '0:(0.5)',
    hybrid_mode: 'optical_flow',
    init_image_b64: '',
    source_video_upload_id: '',
  })
})

test('validation mirrors integer, finite, and backend boundary rules', () => {
  const valid = {
    ...ANIMATE_DEFAULTS,
    duration_seconds: 0.5,
    fps: 1,
    render_frames: 1,
    width: 256,
    height: 1536,
    steps: 3,
    seed: Number.MAX_SAFE_INTEGER,
    diffusion_cadence: 16,
  }
  assert.deepEqual(validateAnimateRequest(valid), {})

  for (const [field, value] of [
    ['duration_seconds', Number.NaN],
    ['duration_seconds', Number.POSITIVE_INFINITY],
    ['fps', 1.5],
    ['render_frames', 1.5],
    ['width', 256.5],
    ['height', Number.NaN],
    ['steps', 3.2],
    ['seed', Number.MAX_SAFE_INTEGER + 1],
    ['diffusion_cadence', 1.5],
  ]) {
    const errors = validateAnimateRequest({ ...ANIMATE_DEFAULTS, [field]: value })
    assert.ok(errors[field], `${field}=${value} should be rejected`)
  }
  assert.ok(validateAnimateRequest({ ...ANIMATE_DEFAULTS, render_frames: 721 }).render_frames)
  assert.ok(validateAnimateRequest({ ...ANIMATE_DEFAULTS, prompt_schedule: 'x'.repeat(32769) }).prompt_schedule)
  assert.ok(validateAnimateRequest({ ...ANIMATE_DEFAULTS, zoom_schedule: 'x'.repeat(32769) }).zoom_schedule)
})

test('motion presets change schedules while preserving prompt and quality', () => {
  const base = {
    ...ANIMATE_DEFAULTS,
    prompt_schedule: '0: keep this exact prompt',
    width: 1024,
    height: 576,
    steps: 12,
  }
  const pushed = applyMotionPreset(base, 'gentle_push_in')
  assert.equal(pushed.prompt_schedule, base.prompt_schedule)
  assert.equal(pushed.width, 1024)
  assert.equal(pushed.steps, 12)
  assert.notEqual(pushed.zoom_schedule, base.zoom_schedule)
  assert.equal(MOTION_PRESETS.slow_orbit.requires3d, true)
  assert.equal(applyMotionPreset(base, 'pan_left').translation_x_schedule, '0:(0), 47:(-24)')
  assert.equal(applyMotionPreset(base, 'pan_right').translation_x_schedule, '0:(0), 47:(24)')
})

test('structured endpoint schedules use the exact last rendered frame', () => {
  assert.equal(buildEndpointSchedule(1, 1.08, 48), '0:(1), 47:(1.08)')
  assert.equal(buildEndpointSchedule(-2.5, 3, 1), '0:(-2.5)')
  assert.throws(() => buildEndpointSchedule(Number.NaN, 1, 48), /finite/)
})

test('quality presets keep rendered frames bounded and reduce diffusion work with cadence', () => {
  assert.equal(ANIMATE_PRESETS.draft.fps, 12)
  assert.ok(ANIMATE_PRESETS.draft.duration_seconds < ANIMATE_DEFAULTS.duration_seconds)
  assert.equal(ANIMATE_PRESETS.cinematic.fps, 24)
  assert.ok(ANIMATE_PRESETS.cinematic.diffusion_cadence > 1)
  assert.ok(calculateRenderedFrames(ANIMATE_PRESETS.cinematic).frames <= 720)
  assert.equal(ANIMATE_PRESETS.smooth.fps, 30)
  assert.ok(ANIMATE_PRESETS.smooth.diffusion_cadence > 1)
  assert.ok(calculateRenderedFrames(ANIMATE_PRESETS.smooth).frames <= 720)
})

test('render frame calculation matches backend override and ties-to-even rounding', () => {
  assert.deepEqual(
    calculateRenderedFrames({ duration_seconds: 2.25, fps: 10, render_frames: null }),
    { frames: 22, diffusionFrames: 22, capped: false, error: '' },
  )
  assert.deepEqual(
    calculateRenderedFrames({ duration_seconds: 60, fps: 60, render_frames: 120, diffusion_cadence: 3 }),
    { frames: 120, diffusionFrames: 40, capped: false, error: '' },
  )
  assert.match(
    calculateRenderedFrames({ duration_seconds: 60, fps: 60, render_frames: null }).error,
    /720/,
  )
})

test('validation reports prompt, dimensions, video input, schedules, and frame limit errors', () => {
  const errors = validateAnimateRequest({
    ...ANIMATE_DEFAULTS,
    prompt_schedule: 'not a keyframe',
    width: 777,
    height: 1600,
    animation_mode: 'Video Input',
    source_video_upload_id: '',
    zoom_schedule: '0: nope',
    duration_seconds: 60,
    fps: 60,
  })
  assert.ok(errors.prompt_schedule)
  assert.ok(errors.width)
  assert.ok(errors.height)
  assert.ok(errors.source_video_upload_id)
  assert.ok(errors.zoom_schedule)
  assert.ok(errors.render_frames)
})
