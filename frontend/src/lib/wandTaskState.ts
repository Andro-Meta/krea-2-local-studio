import type { GpuTaskResponse, GpuTaskStatus, MoodboardSuggestion } from '../api'

export interface WandSubmission {
  jobId: string
  submittedPrompt: string
  submittedRevision: number
}

export interface WandResult {
  expanded: string
  changed: boolean
  error?: string | null
  backend: 'local' | 'openrouter' | 'ideogram-json' | 'gguf-server' | string
  suggested_moodboards?: MoodboardSuggestion[]
  sign_copy_pass?: boolean | null
}

export interface PendingWandResult {
  originalPrompt: string
  currentPrompt: string
  result: WandResult
}

export interface WandTaskState {
  active: {
    submission: WandSubmission
    snapshot: GpuTaskResponse<WandResult>
    cancelPending: boolean
    promptLocked: boolean
  } | null
  pending: PendingWandResult | null
  autoApplied: { originalPrompt: string; result: WandResult } | null
}

export const initialWandTaskState: WandTaskState = {
  active: null,
  pending: null,
  autoApplied: null,
}

export function applyGuardedPromptMutation(
  promptLocked: boolean,
  writePrompt: (prompt: string) => void,
  prompt: string,
): boolean {
  if (promptLocked) return false
  writePrompt(prompt)
  return true
}

export function wandStatusAnnouncement(
  snapshot: Pick<GpuTaskResponse, 'status' | 'queue_position' | 'queue_length'>,
  cancelPending: boolean,
): string {
  if (cancelPending || snapshot.status === 'cancellation_requested') {
    return 'Magic Wand cancellation requested.'
  }
  if (snapshot.status === 'queued') {
    const position = snapshot.queue_position ? ` Position ${snapshot.queue_position}.` : ''
    const length = snapshot.queue_length ? ` ${snapshot.queue_length} total queued tasks.` : ''
    return `Magic Wand is queued.${position}${length}`
  }
  if (snapshot.status === 'finalizing') return 'Magic Wand is finalizing its result.'
  return 'Magic Wand is running.'
}

export function wandProgressAriaLabel(status: GpuTaskStatus): string {
  if (status === 'queued') return 'Magic Wand queue progress'
  if (status === 'finalizing') return 'Magic Wand finalizing progress'
  if (status === 'cancellation_requested') return 'Magic Wand cancellation progress'
  return 'Magic Wand running progress'
}

export function wandCancelAriaLabel(cancelPending: boolean): string {
  return cancelPending ? 'Cancelling Magic Wand' : 'Cancel Magic Wand'
}

export function shouldAutoApplyWand(
  currentPrompt: string,
  currentRevision: number,
  submission: Pick<WandSubmission, 'submittedPrompt' | 'submittedRevision'>,
): boolean {
  return currentRevision === submission.submittedRevision
    && currentPrompt === submission.submittedPrompt
}

export function serializePersistedWandTask(submission: WandSubmission): string {
  return JSON.stringify({ version: 1, ...submission })
}

export function parsePersistedWandTask(raw: string | null): WandSubmission | null {
  if (!raw) return null
  try {
    const value = JSON.parse(raw) as Record<string, unknown>
    if (
      value.version !== 1
      || typeof value.jobId !== 'string'
      || value.jobId.length === 0
      || typeof value.submittedPrompt !== 'string'
      || typeof value.submittedRevision !== 'number'
      || !Number.isSafeInteger(value.submittedRevision)
      || value.submittedRevision < 0
    ) return null
    return {
      jobId: value.jobId,
      submittedPrompt: value.submittedPrompt,
      submittedRevision: value.submittedRevision,
    }
  } catch {
    return null
  }
}

export type WandTaskAction =
  | { type: 'enqueued'; submission: WandSubmission }
  | { type: 'snapshot'; snapshot: GpuTaskResponse<WandResult> }
  | { type: 'cancel-requested' }
  | { type: 'cancel-failed' }
  | { type: 'completed'; currentPrompt: string; currentRevision: number; result: WandResult }
  | { type: 'terminal' }
  | { type: 'identity-changed' }
  | { type: 'auto-applied' }
  | { type: 'discard-pending' }

export function reduceWandTaskState(state: WandTaskState, action: WandTaskAction): WandTaskState {
  switch (action.type) {
    case 'enqueued':
      return {
        ...state,
        active: {
          submission: action.submission,
          snapshot: {
            job_id: action.submission.jobId,
            status: 'queued',
            progress: 0,
            images: [],
            task_kind: 'prompt_expand',
          },
          cancelPending: false,
          promptLocked: false,
        },
        autoApplied: null,
      }
    case 'snapshot':
      return state.active
        ? {
            ...state,
            active: {
              ...state.active,
              snapshot: action.snapshot,
              promptLocked: state.active.promptLocked
                || action.snapshot.status === 'running'
                || action.snapshot.status === 'finalizing',
            },
          }
        : state
    case 'cancel-requested':
      return state.active
        ? { ...state, active: { ...state.active, cancelPending: true } }
        : state
    case 'cancel-failed':
      return state.active
        ? { ...state, active: { ...state.active, cancelPending: false } }
        : state
    case 'completed': {
      if (!state.active) return state
      const output = action.result.changed && action.result.expanded ? action.result : null
      const autoApply = output && shouldAutoApplyWand(
        action.currentPrompt,
        action.currentRevision,
        state.active.submission,
      )
      return {
        active: null,
        pending: output && !autoApply
          ? {
              originalPrompt: state.active.submission.submittedPrompt,
              currentPrompt: action.currentPrompt,
              result: output,
            }
          : null,
        autoApplied: autoApply && output
          ? { originalPrompt: state.active.submission.submittedPrompt, result: output }
          : null,
      }
    }
    case 'terminal':
      return { ...state, active: null }
    case 'identity-changed':
      return initialWandTaskState
    case 'auto-applied':
      return { ...state, autoApplied: null }
    case 'discard-pending':
      return { ...state, pending: null }
  }
}
