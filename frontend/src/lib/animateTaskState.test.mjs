import assert from 'node:assert/strict'
import test from 'node:test'
import {
  initialAnimateTaskState,
  parsePersistedAnimateTask,
  reduceAnimateTaskState,
  serializePersistedAnimateTask,
} from './animateTaskState.ts'

const submission = { jobId: 'animation-1', formRevision: 4, videoTransferred: true }

const result = {
  video_url: '/api/outputs/animations/animation-1/animation.mp4',
  poster_url: '/api/outputs/animations/animation-1/preview.jpg',
  frame_count: 48,
  fps: 12,
  duration: 4,
  gallery_id: 9,
}

test('reducer tracks queue, frame progress, finalizing, and authoritative cancellation', () => {
  let state = reduceAnimateTaskState(initialAnimateTaskState, { type: 'enqueued', submission })
  state = reduceAnimateTaskState(state, {
    type: 'snapshot',
    snapshot: {
      job_id: 'animation-1',
      status: 'running',
      progress: 25,
      images: [],
      queue_position: 2,
      completed_frames: 12,
      total_frames: 48,
      child_job_ids: ['chunk-1'],
    },
  })
  state = reduceAnimateTaskState(state, { type: 'cancel-requested' })
  assert.equal(state.active?.cancelPending, true)
  assert.equal(state.active?.submission.videoTransferred, true)
  assert.equal(state.active?.snapshot.completed_frames, 12)
  assert.equal(state.active?.snapshot.queue_position, 2)

  state = reduceAnimateTaskState(state, {
    type: 'snapshot',
    snapshot: {
      job_id: 'animation-1',
      status: 'cancellation_requested',
      progress: 25,
      images: [],
    },
  })
  assert.equal(state.active?.cancelPending, true)

  state = reduceAnimateTaskState(state, {
    type: 'snapshot',
    snapshot: { job_id: 'animation-1', status: 'cancelled', progress: 25, images: [] },
  })
  assert.equal(state.active?.cancelPending, false)
})

test('reducer ignores stale snapshots and late results from superseded jobs or form revisions', () => {
  let state = reduceAnimateTaskState(initialAnimateTaskState, { type: 'enqueued', submission })
  state = reduceAnimateTaskState(state, {
    type: 'snapshot',
    snapshot: { job_id: 'old-animation', status: 'done', progress: 100, images: [], result },
  })
  assert.equal(state.active?.snapshot.status, 'queued')

  state = reduceAnimateTaskState(state, {
    type: 'delivered',
    jobId: 'animation-1',
    formRevision: 3,
    result,
  })
  assert.equal(state.result, null)
  assert.equal(state.active?.submission.jobId, 'animation-1')
})

test('delivery captures URLs before clearing active task and preserves previous result while running', () => {
  let state = { ...initialAnimateTaskState, result }
  state = reduceAnimateTaskState(state, { type: 'enqueued', submission })
  assert.deepEqual(state.result, result)
  state = reduceAnimateTaskState(state, {
    type: 'delivered',
    jobId: 'animation-1',
    formRevision: 4,
    result: { ...result, gallery_id: 10 },
  })
  assert.equal(state.active, null)
  assert.equal(state.result?.gallery_id, 10)
  assert.equal(state.deliveryReady, true)
})

test('restore payload round-trips and restored jobs remain marked', () => {
  const encoded = serializePersistedAnimateTask(submission)
  assert.deepEqual(parsePersistedAnimateTask(encoded), submission)
  assert.equal(parsePersistedAnimateTask('{"version":2,"jobId":"x","formRevision":1}'), null)
  assert.equal(parsePersistedAnimateTask('{"version":1,"jobId":"","formRevision":1}'), null)

  const state = reduceAnimateTaskState(initialAnimateTaskState, {
    type: 'restored',
    submission,
  })
  assert.equal(state.active?.restored, true)
  assert.equal(state.active?.snapshot.task_kind, 'animation')
})

test('terminal failures retain editable result context but clear active task', () => {
  let state = reduceAnimateTaskState(
    { ...initialAnimateTaskState, result },
    { type: 'enqueued', submission },
  )
  state = reduceAnimateTaskState(state, {
    type: 'terminal',
    jobId: 'animation-1',
    status: 'error',
    error: 'Video finalization failed.',
  })
  assert.equal(state.active, null)
  assert.deepEqual(state.result, result)
  assert.equal(state.error, 'Video finalization failed.')
  assert.equal(state.status, 'error')
})

test('restored completed handoff hydrates a displayable result without an active task', () => {
  const state = reduceAnimateTaskState(initialAnimateTaskState, {
    type: 'hydrate-result',
    result,
  })
  assert.deepEqual(state.result, result)
  assert.equal(state.deliveryReady, true)
  assert.equal(state.active, null)
})
