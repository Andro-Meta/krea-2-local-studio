import assert from 'node:assert/strict'
import test from 'node:test'
import {
  clearConsumedVideoUpload,
  submissionFailureKeepsUpload,
} from './animationRuntime.ts'
import {
  animationResultHandoffKey,
  consumeAnimationResultHandoff,
  persistAnimationResultHandoff,
} from './animationResultHandoff.ts'
import { createTaskWatcher } from './taskWatcher.ts'

const result = {
  video_url: '/api/outputs/animations/job/animation.mp4',
  poster_url: '/api/outputs/animations/job/preview.jpg',
  frame_count: 48,
  fps: 12,
  duration: 4,
  gallery_id: 7,
}

function memoryStorage() {
  const values = new Map()
  return {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  }
}

test('accepted upload clears editable id and records transfer summary', () => {
  const next = clearConsumedVideoUpload({
    animation_mode: 'Video Input',
    source_video_upload_id: 'upload-1',
  })
  assert.equal(next.form.source_video_upload_id, '')
  assert.equal(next.videoTransferred, true)
})

test('explicit rejected submission retains upload but ambiguous network failure clears it', () => {
  assert.equal(submissionFailureKeepsUpload({ response: { status: 422 } }), true)
  assert.equal(submissionFailureKeepsUpload({ response: { status: 503 } }), true)
  assert.equal(submissionFailureKeepsUpload(new Error('network lost')), false)
})

test('animation result handoff is user-scoped, reload-safe, and consumed once', () => {
  const storage = memoryStorage()
  const snapshot = {
    job_id: 'animation-1',
    status: 'done',
    progress: 100,
    images: [],
    task_kind: 'animation',
    result,
  }
  assert.equal(persistAnimationResultHandoff(storage, 'alice', snapshot), true)
  assert.ok(storage.getItem(animationResultHandoffKey('alice')))
  assert.equal(consumeAnimationResultHandoff(storage, 'bob'), null)
  assert.deepEqual(consumeAnimationResultHandoff(storage, 'alice'), snapshot)
  assert.equal(consumeAnimationResultHandoff(storage, 'alice'), null)
})

test('handoff refuses foreign, incomplete, or non-animation snapshots', () => {
  const storage = memoryStorage()
  assert.equal(persistAnimationResultHandoff(storage, 'alice', {
    job_id: 'x', status: 'done', progress: 100, images: [], task_kind: 'generation', result,
  }), false)
  assert.equal(persistAnimationResultHandoff(storage, 'alice', {
    job_id: 'x', status: 'done', progress: 100, images: [], task_kind: 'animation', result: null,
  }), false)
})

test('watcher persists terminal animation result before acknowledgement', async () => {
  const storage = memoryStorage()
  const snapshot = {
    job_id: 'animation-2',
    status: 'done',
    progress: 100,
    images: [],
    task_kind: 'animation',
    result,
  }
  await new Promise((resolve, reject) => {
    const watcher = createTaskWatcher({
      jobId: snapshot.job_id,
      fetchSnapshot: async () => snapshot,
      openSocket: () => ({ readyState: 0, send() {}, close() {} }),
      onSnapshot() {},
      onConnectionNote() {},
      onError: reject,
      onTerminal: terminal => {
        assert.equal(persistAnimationResultHandoff(storage, 'alice', terminal), true)
      },
      acknowledgeAfterDelivery: async () => {
        assert.ok(storage.getItem(animationResultHandoffKey('alice')))
        resolve()
      },
      subscribeToWake: () => () => {},
    })
    watcher.start()
  })
})
