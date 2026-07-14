import type { AnimateRequest } from '../../api'
import animateContract from '../../generated/animate-contract.json' with { type: 'json' }

export type AnimatePresetId = 'draft' | 'cinematic' | 'smooth' | 'custom'
export type MotionPresetId = 'still' | 'gentle_push_in' | 'pan_left' | 'pan_right' | 'slow_orbit' | 'handheld_subtle'

type ContractProperty = {
  default?: unknown
  minimum?: number
  maximum?: number
  maxLength?: number
}

const CONTRACT_PROPERTIES = animateContract.request_schema.properties as Record<string, ContractProperty>
const contractDefault = <K extends keyof AnimateRequest>(key: K): AnimateRequest[K] =>
  CONTRACT_PROPERTIES[key].default as AnimateRequest[K]
const contractMinimum = (key: keyof AnimateRequest, fallback: number) =>
  CONTRACT_PROPERTIES[key].minimum ?? fallback
const contractMaximum = (key: keyof AnimateRequest, fallback: number) =>
  CONTRACT_PROPERTIES[key].maximum ?? fallback

export const ANIMATE_DEFAULTS: AnimateRequest = {
  prompt_schedule: contractDefault('prompt_schedule'),
  negative_prompt: contractDefault('negative_prompt'),
  duration_seconds: contractDefault('duration_seconds'),
  fps: contractDefault('fps'),
  render_frames: contractDefault('render_frames'),
  width: contractDefault('width'),
  height: contractDefault('height'),
  steps: contractDefault('steps'),
  sampler_name: contractDefault('sampler_name'),
  scheduler: contractDefault('scheduler'),
  seed: contractDefault('seed'),
  seed_behavior: contractDefault('seed_behavior'),
  animation_mode: contractDefault('animation_mode'),
  border_mode: contractDefault('border_mode'),
  cfg_schedule: contractDefault('cfg_schedule'),
  strength_schedule: contractDefault('strength_schedule'),
  zoom_schedule: contractDefault('zoom_schedule'),
  angle_schedule: contractDefault('angle_schedule'),
  translation_x_schedule: contractDefault('translation_x_schedule'),
  translation_y_schedule: contractDefault('translation_y_schedule'),
  translation_z_schedule: contractDefault('translation_z_schedule'),
  rotation_3d_x_schedule: contractDefault('rotation_3d_x_schedule'),
  rotation_3d_y_schedule: contractDefault('rotation_3d_y_schedule'),
  rotation_3d_z_schedule: contractDefault('rotation_3d_z_schedule'),
  color_coherence: contractDefault('color_coherence'),
  diffusion_cadence: contractDefault('diffusion_cadence'),
  prompt_blend_frames: contractDefault('prompt_blend_frames'),
  prompt_strength_boost: contractDefault('prompt_strength_boost'),
  prompt_strength_boost_frames: contractDefault('prompt_strength_boost_frames'),
  hybrid_strength_schedule: contractDefault('hybrid_strength_schedule'),
  hybrid_mode: contractDefault('hybrid_mode'),
  init_image_b64: contractDefault('init_image_b64'),
  source_video_upload_id: contractDefault('source_video_upload_id'),
}

export const MOTION_PRESETS: Record<MotionPresetId, { label: string; requires3d: boolean }> = {
  still: { label: 'Still', requires3d: false },
  gentle_push_in: { label: 'Gentle Push In', requires3d: false },
  pan_left: { label: 'Pan Left', requires3d: false },
  pan_right: { label: 'Pan Right', requires3d: false },
  slow_orbit: { label: 'Slow Orbit', requires3d: true },
  handheld_subtle: { label: 'Handheld / Subtle', requires3d: false },
}

export const ANIMATE_PRESETS: Record<AnimatePresetId, AnimateRequest> = {
  draft: {
    ...ANIMATE_DEFAULTS,
    duration_seconds: 2,
    fps: 12,
    width: 512,
    height: 512,
    steps: 4,
    diffusion_cadence: 2,
  },
  cinematic: {
    ...ANIMATE_DEFAULTS,
    duration_seconds: 4,
    fps: 24,
    width: 1024,
    height: 576,
    steps: 12,
    diffusion_cadence: 4,
  },
  smooth: {
    ...ANIMATE_DEFAULTS,
    duration_seconds: 4,
    fps: 30,
    width: 768,
    height: 768,
    steps: 8,
    diffusion_cadence: 5,
  },
  custom: ANIMATE_DEFAULTS,
}

function pythonRound(value: number): number {
  const floor = Math.floor(value)
  const fraction = value - floor
  if (Math.abs(fraction - 0.5) < Number.EPSILON * Math.max(1, Math.abs(value))) {
    return floor % 2 === 0 ? floor : floor + 1
  }
  return Math.round(value)
}

export function calculateRenderedFrames(
  input: Pick<AnimateRequest, 'duration_seconds' | 'fps' | 'render_frames'> &
    Partial<Pick<AnimateRequest, 'diffusion_cadence'>>,
  maxFrames = 720,
): { frames: number; diffusionFrames: number; capped: boolean; error: string } {
  const frames = input.render_frames == null
    ? pythonRound(input.duration_seconds * input.fps)
    : input.render_frames
  const cadence = Math.max(1, input.diffusion_cadence ?? 1)
  const capped = frames > maxFrames
  return {
    frames,
    diffusionFrames: Math.ceil(frames / cadence),
    capped,
    error: capped ? `Rendered frame count exceeds the ${maxFrames}-frame limit.` : '',
  }
}

function scheduleNumber(value: number): string {
  if (!Number.isFinite(value)) throw new Error('Motion values must be finite numbers.')
  return Object.is(value, -0) ? '0' : String(value)
}

export function buildEndpointSchedule(start: number, end: number, frames: number): string {
  if (!Number.isInteger(frames) || frames < 1) throw new Error('Rendered frames must be a positive integer.')
  const first = `0:(${scheduleNumber(start)})`
  return frames === 1 ? first : `${first}, ${frames - 1}:(${scheduleNumber(end)})`
}

export function parseScheduleEndpoints(schedule: string): { start: number; end: number } | null {
  const matches = [...schedule.matchAll(/(?:^|,)\s*\d+\s*:\s*\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)/g)]
  if (!matches.length) return null
  const start = Number(matches[0][1])
  const end = Number(matches[matches.length - 1][1])
  return Number.isFinite(start) && Number.isFinite(end) ? { start, end } : null
}

export function applyMotionPreset(request: AnimateRequest, preset: MotionPresetId): AnimateRequest {
  const frames = calculateRenderedFrames(request).frames
  const still = {
    zoom_schedule: buildEndpointSchedule(1, 1, frames),
    angle_schedule: buildEndpointSchedule(0, 0, frames),
    translation_x_schedule: buildEndpointSchedule(0, 0, frames),
    translation_y_schedule: buildEndpointSchedule(0, 0, frames),
    translation_z_schedule: buildEndpointSchedule(0, 0, frames),
    rotation_3d_x_schedule: buildEndpointSchedule(0, 0, frames),
    rotation_3d_y_schedule: buildEndpointSchedule(0, 0, frames),
    rotation_3d_z_schedule: buildEndpointSchedule(0, 0, frames),
  }
  switch (preset) {
    case 'still':
      return { ...request, ...still, animation_mode: 'None' }
    case 'gentle_push_in':
      return { ...request, ...still, animation_mode: '2D', zoom_schedule: buildEndpointSchedule(1, 1.08, frames) }
    case 'pan_left':
      return { ...request, ...still, animation_mode: '2D', translation_x_schedule: buildEndpointSchedule(0, -24, frames) }
    case 'pan_right':
      return { ...request, ...still, animation_mode: '2D', translation_x_schedule: buildEndpointSchedule(0, 24, frames) }
    case 'slow_orbit':
      return {
        ...request,
        ...still,
        animation_mode: '3D',
        translation_z_schedule: buildEndpointSchedule(0, 3, frames),
        rotation_3d_y_schedule: buildEndpointSchedule(0, 6, frames),
      }
    case 'handheld_subtle':
      return {
        ...request,
        ...still,
        animation_mode: '2D',
        angle_schedule: buildEndpointSchedule(-0.35, 0.35, frames),
        translation_x_schedule: buildEndpointSchedule(-2, 3, frames),
        translation_y_schedule: buildEndpointSchedule(1, -2, frames),
      }
  }
}

const NUMERIC_SCHEDULE_FIELDS = [
  'cfg_schedule',
  'strength_schedule',
  'zoom_schedule',
  'angle_schedule',
  'translation_x_schedule',
  'translation_y_schedule',
  'translation_z_schedule',
  'rotation_3d_x_schedule',
  'rotation_3d_y_schedule',
  'rotation_3d_z_schedule',
  'hybrid_strength_schedule',
] as const

function promptScheduleError(value: string, totalFrames: number): string {
  if (typeof value !== 'string' || !value.trim()) return 'Add at least one “frame: prompt” line.'
  if (value.length > (CONTRACT_PROPERTIES.prompt_schedule.maxLength ?? 32 * 1024)) {
    return 'Prompt schedule is too long (maximum 32768 characters).'
  }
  const seen = new Set<number>()
  const lines = value.split(/\r?\n/).filter(line => line.trim())
  if (!lines.length) return 'Add at least one “frame: prompt” line.'
  for (const line of lines) {
    const match = line.match(/^\s*(-?\d+)\s*:\s*(.+?)\s*$/)
    if (!match) return 'Each prompt keyframe must use “frame: prompt”.'
    const frame = Number(match[1])
    if (frame < 0 || frame >= totalFrames) return `Prompt frame ${frame} is outside this animation.`
    if (seen.has(frame)) return `Prompt frame ${frame} is duplicated.`
    seen.add(frame)
  }
  return ''
}

function numericScheduleError(value: string, totalFrames: number): string {
  if (typeof value !== 'string' || !value.trim()) return 'Add a schedule such as 0:(1.0).'
  if (value.length > 32 * 1024) return 'Schedule is too long (maximum 32768 characters).'
  const entries = value.split(/,(?![^()]*\))/).map(entry => entry.trim())
  if (!entries.length || entries.some(entry => !entry)) return 'Add a schedule such as 0:(1.0).'
  const seen = new Set<number>()
  for (const entry of entries) {
    const match = entry.match(/^(-?\d+)\s*:\s*\((.+)\)$/)
    if (!match) return 'Use comma-separated frame:(expression) entries.'
    const frame = Number(match[1])
    if (frame < 0 || frame >= totalFrames) return `Schedule frame ${frame} is outside this animation.`
    if (seen.has(frame)) return `Schedule frame ${frame} is duplicated.`
    if (!/^[\d\s+\-*/().,_a-zA-Z]+$/.test(match[2])) return 'Schedule expression contains unsupported characters.'
    seen.add(frame)
  }
  return ''
}

export type AnimateValidationErrors = Partial<Record<keyof AnimateRequest, string>>

export function validateAnimateRequest(
  request: AnimateRequest,
  limits: { maxFrames?: number; maxDimension?: number } = {},
): AnimateValidationErrors {
  const maxFrames = limits.maxFrames ?? contractMaximum('render_frames', 720)
  const maxDimension = limits.maxDimension ?? contractMaximum('width', 1536)
  const frameInfo = calculateRenderedFrames(request, maxFrames)
  const errors: AnimateValidationErrors = {}
  const safeFrames = Number.isInteger(frameInfo.frames) && frameInfo.frames > 0 ? frameInfo.frames : 1
  const promptError = promptScheduleError(request.prompt_schedule, safeFrames)
  if (promptError) errors.prompt_schedule = promptError
  if (!Number.isFinite(request.duration_seconds) || request.duration_seconds < contractMinimum('duration_seconds', 0.5) || request.duration_seconds > contractMaximum('duration_seconds', 60)) {
    errors.duration_seconds = 'Duration must be a finite number from 0.5–60 seconds.'
  }
  if (!Number.isInteger(request.fps) || request.fps < contractMinimum('fps', 1) || request.fps > contractMaximum('fps', 60)) {
    errors.fps = 'Playback FPS must be a whole number from 1–60.'
  }
  if (request.render_frames !== null && (
    !Number.isInteger(request.render_frames)
    || request.render_frames < contractMinimum('render_frames', 1)
    || request.render_frames > maxFrames
  )) {
    errors.render_frames = `Rendered frame override must be a whole number from 1–${maxFrames}.`
  } else if (frameInfo.error) errors.render_frames = frameInfo.error
  if (!Number.isInteger(request.width) || request.width < contractMinimum('width', 256) || request.width > maxDimension || request.width % 16) {
    errors.width = `Width must be 256–${maxDimension} and divisible by 16.`
  }
  if (!Number.isInteger(request.height) || request.height < contractMinimum('height', 256) || request.height > maxDimension || request.height % 16) {
    errors.height = `Height must be 256–${maxDimension} and divisible by 16.`
  }
  if (!Number.isInteger(request.steps) || request.steps < contractMinimum('steps', 3) || request.steps > contractMaximum('steps', 52)) {
    errors.steps = 'Steps must be a whole number from 3–52.'
  }
  if (!Number.isSafeInteger(request.seed) || request.seed < contractMinimum('seed', -1)) {
    errors.seed = `Seed must be -1 or a whole number no larger than ${Number.MAX_SAFE_INTEGER}.`
  }
  if (!Number.isInteger(request.diffusion_cadence) || request.diffusion_cadence < contractMinimum('diffusion_cadence', 1) || request.diffusion_cadence > contractMaximum('diffusion_cadence', 16)) {
    errors.diffusion_cadence = 'Diffusion cadence must be a whole number from 1–16.'
  }
  if (!Number.isInteger(request.prompt_blend_frames) || request.prompt_blend_frames < contractMinimum('prompt_blend_frames', 0) || request.prompt_blend_frames > contractMaximum('prompt_blend_frames', 12)) {
    errors.prompt_blend_frames = 'Prompt blend frames must be a whole number from 0–12.'
  }
  if (!Number.isFinite(request.prompt_strength_boost) || request.prompt_strength_boost < contractMinimum('prompt_strength_boost', 0) || request.prompt_strength_boost > contractMaximum('prompt_strength_boost', 0.35)) {
    errors.prompt_strength_boost = 'Strength boost must be from 0–0.35.'
  }
  if (!Number.isInteger(request.prompt_strength_boost_frames) || request.prompt_strength_boost_frames < contractMinimum('prompt_strength_boost_frames', 0) || request.prompt_strength_boost_frames > contractMaximum('prompt_strength_boost_frames', 16)) {
    errors.prompt_strength_boost_frames = 'Boost window must be a whole number from 0–16.'
  }
  if (request.animation_mode === 'Video Input' && !request.source_video_upload_id.trim()) {
    errors.source_video_upload_id = 'Upload a source video before queueing.'
  }
  for (const field of NUMERIC_SCHEDULE_FIELDS) {
    const error = numericScheduleError(request[field], safeFrames)
    if (error) errors[field] = error
  }
  return errors
}
