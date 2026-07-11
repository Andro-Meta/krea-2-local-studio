export interface TaskStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export function activeGpuTaskStorageKey(
  username: string | null | undefined,
  taskKind: string,
): string {
  return `krea2_active_gpu_task:${username?.trim() || 'local'}:${taskKind}`
}

export function persistActiveTask(storage: TaskStorage, storageKey: null, jobId: string): false
export function persistActiveTask(storage: TaskStorage, storageKey: string, jobId: string): true
export function persistActiveTask(storage: TaskStorage, storageKey: string | null, jobId: string): boolean
export function persistActiveTask(
  storage: TaskStorage,
  storageKey: string | null,
  jobId: string,
): boolean {
  if (!storageKey) return false
  storage.setItem(storageKey, jobId)
  return true
}

export function readPersistedActiveTask(storage: TaskStorage, storageKey: null): null
export function readPersistedActiveTask(storage: TaskStorage, storageKey: string): string | null
export function readPersistedActiveTask(storage: TaskStorage, storageKey: string | null): string | null
export function readPersistedActiveTask(
  storage: TaskStorage,
  storageKey: string | null,
): string | null {
  return storageKey ? storage.getItem(storageKey) : null
}

export function clearPersistedActiveTask(
  storage: TaskStorage,
  storageKey: string | null,
  expectedJobId: string,
): boolean {
  if (!storageKey || storage.getItem(storageKey) !== expectedJobId) return false
  storage.removeItem(storageKey)
  return true
}

export function adoptActiveTaskPersistence(
  storage: TaskStorage,
  resolvedStorageKey: string,
  activeJobId: string | null,
  watcherActive: boolean,
): string | null {
  if (!watcherActive || !activeJobId) return null
  persistActiveTask(storage, resolvedStorageKey, activeJobId)
  return resolvedStorageKey
}

export interface ActiveTaskIdentityTransition {
  identityChanged: boolean
  stopWatcher: boolean
  adoptStorageKey: string | null
  consultStorageKey: string | null
}

export function reconcileActiveTaskIdentity({
  previousResolvedKey,
  nextResolvedKey,
  watcherActive,
  watchedStorageKey,
}: {
  previousResolvedKey: string | null
  nextResolvedKey: string
  watcherActive: boolean
  watchedStorageKey: string | null
}): ActiveTaskIdentityTransition {
  const watcherBelongsToAnotherIdentity = watchedStorageKey !== null
    && watchedStorageKey !== nextResolvedKey
  const identityChanged = (
    previousResolvedKey !== null
    && previousResolvedKey !== nextResolvedKey
  ) || watcherBelongsToAnotherIdentity
  const stopWatcher = watcherActive && (identityChanged || watcherBelongsToAnotherIdentity)
  if (stopWatcher || !watcherActive) {
    return {
      identityChanged,
      stopWatcher,
      adoptStorageKey: null,
      consultStorageKey: nextResolvedKey,
    }
  }
  return {
    identityChanged,
    stopWatcher: false,
    adoptStorageKey: watchedStorageKey === null ? nextResolvedKey : null,
    consultStorageKey: null,
  }
}

export function canDeliverTaskResult(
  activeStorageKey: string | null,
  watchedStorageKey: string | null,
): boolean {
  return activeStorageKey === watchedStorageKey
}
