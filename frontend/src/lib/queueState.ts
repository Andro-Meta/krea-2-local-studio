import type { QueueJob } from '../api'

const CANCELLATION_SETTLED = new Set([
  'cancellation_requested',
  'finalizing',
  'done',
  'error',
  'blocked',
  'cancelled',
])

export function reconcilePendingCancellations(
  pending: ReadonlySet<string>,
  jobs: readonly QueueJob[],
): Set<string> {
  const byId = new Map(jobs.map(job => [job.job_id, job]))
  return new Set([...pending].filter(jobId => {
    const job = byId.get(jobId)
    return !!job && !CANCELLATION_SETTLED.has(job.status)
  }))
}
