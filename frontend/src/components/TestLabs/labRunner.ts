import { apiFetch, type GenerationJob, type GenerationRequest } from '../../api'
import type { LabCase, LabWorkflow } from './labPresets'

export type LabCaseStatus = 'idle' | 'queued' | 'running' | 'done' | 'error' | 'blocked' | 'cancelled' | 'skipped'

export interface LabCaseRun {
  caseId: string
  label: string
  status: LabCaseStatus
  jobId?: string
  startedAt?: number
  finishedAt?: number
  elapsedSec?: number
  error?: string
  job?: GenerationJob
  request: GenerationRequest
}

export interface LabRun {
  workflowId: string
  workflowLabel: string
  startedAt: number
  finishedAt?: number
  cases: LabCaseRun[]
}

export interface RunOptions {
  sourceImageB64?: string
  seed?: number
  shouldStop?: () => boolean
  onUpdate?: (run: LabRun) => void
}

const terminalStates = new Set(['done', 'error', 'blocked', 'cancelled'])

function cloneRequest(request: GenerationRequest): GenerationRequest {
  return JSON.parse(JSON.stringify(request)) as GenerationRequest
}

export function buildCaseRequest(testCase: LabCase, opts: Pick<RunOptions, 'sourceImageB64' | 'seed'> = {}): GenerationRequest {
  const request = cloneRequest(testCase.request)
  if (opts.seed !== undefined) request.seed = opts.seed
  const source = opts.sourceImageB64?.trim()
  if (source && testCase.sourceInjection === 'redraw-reference') {
    request.mode = 'redraw'
    request.moodboard_images = [source]
    request.init_image_b64 = ''
    request.mask_b64 = ''
  }
  if (source && testCase.sourceInjection === 'img2img-init') {
    request.mode = 'img2img'
    request.init_image_b64 = source
    request.mask_b64 = ''
  }
  if (source && testCase.sourceInjection === 'inpaint-init-mask') {
    request.mode = 'inpaint'
    request.init_image_b64 = source
  }
  return request
}

function copyRun(run: LabRun): LabRun {
  return { ...run, cases: run.cases.map(item => ({ ...item })) }
}

function updateCase(run: LabRun, caseId: string, patch: Partial<LabCaseRun>) {
  run.cases = run.cases.map(item => item.caseId === caseId ? { ...item, ...patch } : item)
}

async function waitForJob(jobId: string, onSnapshot: (job: GenerationJob) => void, shouldStop?: () => boolean): Promise<GenerationJob> {
  for (;;) {
    let job: GenerationJob
    try {
      job = await apiFetch.jobStatus(jobId)
    } catch (err: any) {
      if (err?.response?.status === 404) {
        // Job evicted server-side (or not visible to this user) — terminal.
        return { status: 'error', error: 'Job is no longer available on the server.' } as GenerationJob
      }
      // Transient network error: keep polling.
      await new Promise(resolve => window.setTimeout(resolve, 2000))
      continue
    }
    onSnapshot(job)
    if (terminalStates.has(job.status)) return job
    await new Promise(resolve => window.setTimeout(resolve, 2000))
  }
}

export async function runLabWorkflow(workflow: LabWorkflow, options: RunOptions = {}): Promise<LabRun> {
  const run: LabRun = {
    workflowId: workflow.id,
    workflowLabel: workflow.label,
    startedAt: Date.now(),
    cases: workflow.cases.map(testCase => ({
      caseId: testCase.id,
      label: testCase.label,
      status: testCase.disabledReason ? 'skipped' : 'idle',
      error: testCase.disabledReason,
      request: buildCaseRequest(testCase, options),
    })),
  }
  options.onUpdate?.(copyRun(run))

  for (const testCase of workflow.cases) {
    if (options.shouldStop?.()) break
    if (testCase.disabledReason) continue
    if (testCase.sourceInjection && testCase.sourceInjection !== 'none' && !options.sourceImageB64) {
      updateCase(run, testCase.id, { status: 'skipped', error: 'Upload a source image to run this case.' })
      options.onUpdate?.(copyRun(run))
      continue
    }

    const request = buildCaseRequest(testCase, options)
    const startedAt = Date.now()
    updateCase(run, testCase.id, { status: 'queued', startedAt, request })
    options.onUpdate?.(copyRun(run))

    try {
      const submitted = await apiFetch.generate(request)
      updateCase(run, testCase.id, { status: 'running', jobId: submitted.job_id })
      options.onUpdate?.(copyRun(run))
      const job = await waitForJob(submitted.job_id, snapshot => {
        updateCase(run, testCase.id, {
          status: terminalStates.has(snapshot.status) ? snapshot.status as LabCaseStatus : 'running',
          job: snapshot,
        })
        options.onUpdate?.(copyRun(run))
      }, options.shouldStop)
      const finishedAt = Date.now()
      updateCase(run, testCase.id, {
        status: job.status as LabCaseStatus,
        finishedAt,
        elapsedSec: (finishedAt - startedAt) / 1000,
        error: job.error ?? undefined,
        job,
      })
      options.onUpdate?.(copyRun(run))
    } catch (error: any) {
      const finishedAt = Date.now()
      updateCase(run, testCase.id, {
        status: 'error',
        finishedAt,
        elapsedSec: (finishedAt - startedAt) / 1000,
        error: error?.response?.data?.detail ?? error?.message ?? 'Test case failed.',
      })
      options.onUpdate?.(copyRun(run))
    }
  }

  run.finishedAt = Date.now()
  options.onUpdate?.(copyRun(run))
  return run
}

export function exportRunJson(run: LabRun): string {
  return JSON.stringify(run, null, 2)
}
