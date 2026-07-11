import assert from 'node:assert/strict'
import test from 'node:test'
import {
  applyGuardedPromptMutation,
  initialWandTaskState,
  parsePersistedWandTask,
  reduceWandTaskState,
  serializePersistedWandTask,
  shouldAutoApplyWand,
  wandCancelAriaLabel,
  wandProgressAriaLabel,
  wandStatusAnnouncement,
} from './wandTaskState.ts'

const submission = {
  jobId: 'wand-1',
  submittedPrompt: 'a fox',
  submittedRevision: 3,
}

test('auto-applies only when prompt text and revision still match', () => {
  assert.equal(shouldAutoApplyWand('a fox', 3, submission), true)
  assert.equal(shouldAutoApplyWand('a fox', 4, submission), false)
  assert.equal(shouldAutoApplyWand('a fox ', 3, submission), false)
})

test('persisted wand payload round-trips and rejects malformed or stale shapes', () => {
  const encoded = serializePersistedWandTask(submission)
  assert.deepEqual(parsePersistedWandTask(encoded), submission)
  assert.equal(parsePersistedWandTask('wand-1'), null)
  assert.equal(parsePersistedWandTask('{"version":1,"jobId":"","submittedPrompt":"fox","submittedRevision":0}'), null)
  assert.equal(parsePersistedWandTask('{"version":1,"jobId":"wand","submittedPrompt":"fox","submittedRevision":-1}'), null)
  assert.equal(parsePersistedWandTask('{"version":2,"jobId":"wand","submittedPrompt":"fox","submittedRevision":0}'), null)
})

test('reducer tracks queue snapshots and cancellation without losing submission', () => {
  let state = reduceWandTaskState(initialWandTaskState, { type: 'enqueued', submission })
  state = reduceWandTaskState(state, {
    type: 'snapshot',
    snapshot: {
      job_id: 'wand-1',
      status: 'queued',
      progress: 0,
      images: [],
      queue_position: 2,
      queue_length: 5,
      task_kind: 'prompt_expand',
    },
  })
  state = reduceWandTaskState(state, { type: 'cancel-requested' })

  assert.equal(state.active?.submission.submittedPrompt, 'a fox')
  assert.equal(state.active?.snapshot.queue_position, 2)
  assert.equal(state.active?.promptLocked, false)
  assert.equal(state.active?.cancelPending, true)

  state = reduceWandTaskState(state, { type: 'cancel-failed' })
  assert.equal(state.active?.cancelPending, false)
  state = reduceWandTaskState(state, {
    type: 'snapshot',
    snapshot: { job_id: 'wand-1', status: 'running', progress: 0, images: [] },
  })
  state = reduceWandTaskState(state, {
    type: 'snapshot',
    snapshot: { job_id: 'wand-1', status: 'cancellation_requested', progress: 0, images: [] },
  })
  assert.equal(state.active?.promptLocked, true)
})

test('late completion stores a pending result instead of applying it', () => {
  let state = reduceWandTaskState(initialWandTaskState, { type: 'enqueued', submission })
  state = reduceWandTaskState(state, {
    type: 'completed',
    currentPrompt: 'a wolf',
    currentRevision: 4,
    result: {
      expanded: 'a cinematic fox',
      changed: true,
      backend: 'local',
      suggested_moodboards: [{ id: 1, title: 'Cinema' }],
    },
  })

  assert.equal(state.active, null)
  assert.equal(state.pending?.originalPrompt, 'a fox')
  assert.equal(state.pending?.currentPrompt, 'a wolf')
  assert.equal(state.pending?.result.expanded, 'a cinematic fox')
  assert.equal(state.autoApplied, null)
})

test('matching completion emits one auto-apply and pending actions clear it', () => {
  let state = reduceWandTaskState(initialWandTaskState, { type: 'enqueued', submission })
  state = reduceWandTaskState(state, {
    type: 'completed',
    currentPrompt: 'a fox',
    currentRevision: 3,
    result: { expanded: 'a cinematic fox', changed: true, backend: 'local' },
  })
  assert.equal(state.autoApplied?.result.expanded, 'a cinematic fox')
  assert.equal(state.autoApplied?.originalPrompt, 'a fox')
  assert.equal(state.pending, null)

  state = reduceWandTaskState(state, { type: 'auto-applied' })
  assert.equal(state.autoApplied, null)
  state = reduceWandTaskState(state, { type: 'discard-pending' })
  assert.equal(state.pending, null)
})

test('identity change clears active and pending wand UI', () => {
  let state = reduceWandTaskState(initialWandTaskState, { type: 'enqueued', submission })
  state = reduceWandTaskState(state, {
    type: 'completed',
    currentPrompt: 'edited',
    currentRevision: 4,
    result: { expanded: 'expanded', changed: true, backend: 'local' },
  })
  state = reduceWandTaskState(state, { type: 'identity-changed' })
  assert.deepEqual(state, initialWandTaskState)
})

test('guarded prompt mutation blocks every write while locked', () => {
  const writes = []
  assert.equal(applyGuardedPromptMutation(true, value => writes.push(value), 'blocked'), false)
  assert.deepEqual(writes, [])
  assert.equal(applyGuardedPromptMutation(false, value => writes.push(value), 'allowed'), true)
  assert.deepEqual(writes, ['allowed'])
})

test('wand accessibility text is stable across progress-only polls', () => {
  const runningAtTen = {
    job_id: 'wand-1', status: 'running', progress: 10, images: [],
  }
  const runningAtNinety = { ...runningAtTen, progress: 90 }
  assert.equal(
    wandStatusAnnouncement(runningAtTen, false),
    wandStatusAnnouncement(runningAtNinety, false),
  )
  assert.equal(wandStatusAnnouncement(runningAtTen, false), 'Magic Wand is running.')
  assert.equal(wandStatusAnnouncement(runningAtTen, true), 'Magic Wand cancellation requested.')
  assert.equal(wandProgressAriaLabel('running'), 'Magic Wand running progress')
  assert.equal(wandCancelAriaLabel(false), 'Cancel Magic Wand')
  assert.equal(wandCancelAriaLabel(true), 'Cancelling Magic Wand')
})
