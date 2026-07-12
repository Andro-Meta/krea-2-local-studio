import assert from 'node:assert/strict'
import test from 'node:test'
import {
  activeGpuTaskStorageKey,
  adoptActiveTaskPersistence,
  canDeliverTaskResult,
  reconcileActiveTaskIdentity,
  readPersistedActiveTask,
} from './activeTaskPersistence.ts'

function memoryStorage() {
  const values = new Map()
  return {
    values,
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value) },
    removeItem: key => { values.delete(key) },
  }
}

test('deferred auth persists an active watched job for reload resume', () => {
  const storage = memoryStorage()
  const key = activeGpuTaskStorageKey('alice', 'generation')

  const adoptedKey = adoptActiveTaskPersistence(storage, key, 'job-1', true)

  assert.equal(adoptedKey, key)
  assert.equal(storage.values.get(key), 'job-1')
  assert.equal(readPersistedActiveTask(storage, key), 'job-1')
})

test('terminal-before-auth does not persist or resurrect the job', () => {
  const storage = memoryStorage()
  const key = activeGpuTaskStorageKey('alice', 'generation')

  const adoptedKey = adoptActiveTaskPersistence(storage, key, null, false)

  assert.equal(adoptedKey, null)
  assert.equal(storage.values.has(key), false)
  assert.equal(readPersistedActiveTask(storage, key), null)
})

test('user switch stops old watcher, retains old key, and consults new key', () => {
  const storage = memoryStorage()
  const aliceKey = activeGpuTaskStorageKey('alice', 'generation')
  const bobKey = activeGpuTaskStorageKey('bob', 'generation')
  storage.setItem(aliceKey, 'alice-job')
  storage.setItem(bobKey, 'bob-job')

  const transition = reconcileActiveTaskIdentity({
    previousResolvedKey: aliceKey,
    nextResolvedKey: bobKey,
    watcherActive: true,
    watchedStorageKey: aliceKey,
  })

  assert.equal(transition.identityChanged, true)
  assert.equal(transition.stopWatcher, true)
  assert.equal(transition.adoptStorageKey, null)
  assert.equal(transition.consultStorageKey, bobKey)
  assert.equal(storage.getItem(aliceKey), 'alice-job')
  assert.equal(readPersistedActiveTask(storage, transition.consultStorageKey), 'bob-job')
  assert.equal(canDeliverTaskResult(bobKey, aliceKey), false)
})

test('unresolved task adopts only its first consistent resolved identity', () => {
  const aliceKey = activeGpuTaskStorageKey('alice', 'prompt_expand')
  const firstResolution = reconcileActiveTaskIdentity({
    previousResolvedKey: null,
    nextResolvedKey: aliceKey,
    watcherActive: true,
    watchedStorageKey: null,
  })
  assert.equal(firstResolution.stopWatcher, false)
  assert.equal(firstResolution.adoptStorageKey, aliceKey)
  assert.equal(firstResolution.consultStorageKey, null)
  assert.equal(canDeliverTaskResult(aliceKey, null), false)
  assert.equal(canDeliverTaskResult(aliceKey, aliceKey), true)
})
