import type { AnimationResult, GpuTaskResponse } from '../api'
import type { TaskStorage } from './activeTaskPersistence'

export type AnimationResultSnapshot = GpuTaskResponse<AnimationResult> & {
  task_kind: 'animation'
  status: 'done'
  result: AnimationResult
}

export function animationResultHandoffKey(username: string | null | undefined): string {
  return `krea2_animation_result_handoff:${username?.trim() || 'local'}`
}

function validResult(value: unknown): value is AnimationResult {
  if (!value || typeof value !== 'object') return false
  const result = value as Record<string, unknown>
  return typeof result.video_url === 'string'
    && result.video_url.length > 0
    && typeof result.poster_url === 'string'
    && result.poster_url.length > 0
    && typeof result.frame_count === 'number'
    && typeof result.fps === 'number'
    && typeof result.duration === 'number'
    && typeof result.gallery_id === 'number'
}

function validSnapshot(value: unknown): value is AnimationResultSnapshot {
  if (!value || typeof value !== 'object') return false
  const snapshot = value as Record<string, unknown>
  return snapshot.task_kind === 'animation'
    && snapshot.status === 'done'
    && typeof snapshot.job_id === 'string'
    && snapshot.job_id.length > 0
    && validResult(snapshot.result)
}

export function persistAnimationResultHandoff(
  storage: TaskStorage,
  username: string | null | undefined,
  snapshot: GpuTaskResponse<unknown>,
): boolean {
  if (!validSnapshot(snapshot)) return false
  storage.setItem(animationResultHandoffKey(username), JSON.stringify(snapshot))
  return true
}

export function consumeAnimationResultHandoff(
  storage: TaskStorage,
  username: string | null | undefined,
): AnimationResultSnapshot | null {
  const key = animationResultHandoffKey(username)
  const raw = storage.getItem(key)
  if (!raw) return null
  try {
    const value: unknown = JSON.parse(raw)
    if (!validSnapshot(value)) {
      storage.removeItem(key)
      return null
    }
    storage.removeItem(key)
    return value
  } catch {
    storage.removeItem(key)
    return null
  }
}
