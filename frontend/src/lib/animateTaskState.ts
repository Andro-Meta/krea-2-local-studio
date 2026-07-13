import type {
  AnimationResult,
  GpuTaskResponse,
  GpuTaskStatus,
} from '../api'

export interface AnimateSubmission {
  jobId: string
  formRevision: number
  videoTransferred: boolean
}

export type AnimationTaskSnapshot = GpuTaskResponse<AnimationResult> & {
  completed_frames?: number
  total_frames?: number
  chunk_index?: number
}

export interface AnimateTaskState {
  active: {
    submission: AnimateSubmission
    snapshot: AnimationTaskSnapshot
    cancelPending: boolean
    restored: boolean
  } | null
  status: GpuTaskStatus | 'idle'
  error: string
  result: AnimationResult | null
  deliveryReady: boolean
}

export const initialAnimateTaskState: AnimateTaskState = {
  active: null,
  status: 'idle',
  error: '',
  result: null,
  deliveryReady: false,
}

function activeFromSubmission(submission: AnimateSubmission, restored: boolean): AnimateTaskState['active'] {
  return {
    submission,
    snapshot: {
      job_id: submission.jobId,
      status: 'queued',
      progress: 0,
      images: [],
      task_kind: 'animation',
    },
    cancelPending: false,
    restored,
  }
}

export function serializePersistedAnimateTask(submission: AnimateSubmission): string {
  return JSON.stringify({ version: 1, ...submission })
}

export function parsePersistedAnimateTask(raw: string | null): AnimateSubmission | null {
  if (!raw) return null
  try {
    const value = JSON.parse(raw) as Record<string, unknown>
    if (
      value.version !== 1
      || typeof value.jobId !== 'string'
      || !value.jobId
      || typeof value.formRevision !== 'number'
      || !Number.isSafeInteger(value.formRevision)
      || value.formRevision < 0
    ) return null
    return {
      jobId: value.jobId,
      formRevision: value.formRevision,
      videoTransferred: value.videoTransferred === true,
    }
  } catch {
    return null
  }
}

export type AnimateTaskAction =
  | { type: 'enqueued'; submission: AnimateSubmission }
  | { type: 'restored'; submission: AnimateSubmission }
  | { type: 'snapshot'; snapshot: AnimationTaskSnapshot }
  | { type: 'cancel-requested' }
  | { type: 'cancel-failed' }
  | {
      type: 'delivered'
      jobId: string
      formRevision: number
      result: AnimationResult
    }
  | {
      type: 'terminal'
      jobId: string
      status: Extract<GpuTaskStatus, 'error' | 'blocked' | 'cancelled'>
      error?: string
    }
  | { type: 'identity-changed' }
  | { type: 'hydrate-result'; result: AnimationResult }
  | { type: 'clear-error' }

export function reduceAnimateTaskState(
  state: AnimateTaskState,
  action: AnimateTaskAction,
): AnimateTaskState {
  switch (action.type) {
    case 'enqueued':
      return {
        ...state,
        active: activeFromSubmission(action.submission, false),
        status: 'queued',
        error: '',
        deliveryReady: false,
      }
    case 'restored':
      return {
        ...state,
        active: activeFromSubmission(action.submission, true),
        status: 'queued',
        error: '',
        deliveryReady: false,
      }
    case 'snapshot': {
      if (!state.active || action.snapshot.job_id !== state.active.submission.jobId) return state
      const authoritativeTerminal = ['done', 'error', 'blocked', 'cancelled'].includes(action.snapshot.status)
      return {
        ...state,
        status: action.snapshot.status,
        active: {
          ...state.active,
          snapshot: { ...state.active.snapshot, ...action.snapshot },
          cancelPending: authoritativeTerminal ? false : state.active.cancelPending,
        },
      }
    }
    case 'cancel-requested':
      return state.active
        ? { ...state, active: { ...state.active, cancelPending: true } }
        : state
    case 'cancel-failed':
      return state.active
        ? { ...state, active: { ...state.active, cancelPending: false } }
        : state
    case 'delivered':
      if (
        !state.active
        || action.jobId !== state.active.submission.jobId
        || action.formRevision !== state.active.submission.formRevision
      ) return state
      return {
        ...state,
        active: null,
        status: 'done',
        error: '',
        result: action.result,
        deliveryReady: true,
      }
    case 'terminal':
      if (!state.active || action.jobId !== state.active.submission.jobId) return state
      return {
        ...state,
        active: null,
        status: action.status,
        error: action.error || (
          action.status === 'cancelled'
            ? 'Animation cancelled.'
            : action.status === 'blocked'
              ? 'Animation was blocked by the safety policy.'
              : 'Animation failed.'
        ),
        deliveryReady: false,
      }
    case 'identity-changed':
      return initialAnimateTaskState
    case 'hydrate-result':
      return { ...state, result: action.result, deliveryReady: true }
    case 'clear-error':
      return { ...state, error: '' }
  }
}
