import {
  createTaskWatcher,
  activeGpuTaskStorageKey,
  clearPersistedActiveTask,
  isGpuTaskTerminal,
  persistActiveTask,
  readPersistedActiveTask,
  type TaskStorage,
  type TaskWatcher,
  type TaskWatcherOptions,
} from './taskWatcher'
import type { GpuTaskResponse } from '../api'

type Result = { value: string }
const options: TaskWatcherOptions<Result> = {
  jobId: 'job',
  fetchSnapshot: async () => ({ job_id: 'job', status: 'queued', progress: 0, images: [] }),
  openSocket: () => ({ readyState: 0, send: () => undefined, close: () => undefined }),
  onSnapshot: (_snapshot: GpuTaskResponse<Result>) => undefined,
  onTerminal: snapshot => { void snapshot.result?.value },
  onConnectionNote: _note => undefined,
  onError: _error => undefined,
}

const watcher: TaskWatcher = createTaskWatcher(options)
void watcher
void isGpuTaskTerminal('done')

const localGenerationKey: string = activeGpuTaskStorageKey(null, 'generation')
const userGenerationKey: string = activeGpuTaskStorageKey('alice', 'generation')
const otherUserGenerationKey: string = activeGpuTaskStorageKey('bob', 'generation')
void localGenerationKey
void userGenerationKey
void otherUserGenerationKey

const memoryStorage: TaskStorage = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
}
const unresolvedWrite: false = persistActiveTask(memoryStorage, null, 'job')
const unresolvedRead: null = readPersistedActiveTask(memoryStorage, null)
const resolvedWrite: true = persistActiveTask(memoryStorage, userGenerationKey, 'job')
const resolvedRead: string | null = readPersistedActiveTask(memoryStorage, userGenerationKey)
const otherUserRead: string | null = readPersistedActiveTask(memoryStorage, otherUserGenerationKey)
const didClear: boolean = clearPersistedActiveTask(memoryStorage, userGenerationKey, 'job')
void unresolvedWrite
void unresolvedRead
void resolvedWrite
void resolvedRead
void otherUserRead
void didClear
