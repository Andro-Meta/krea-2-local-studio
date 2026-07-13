import {
  gpuTaskTerminalError,
  isGpuTaskTerminal,
  responseStatus,
  type GpuTaskResponse,
} from '../api.ts'
export {
  activeGpuTaskStorageKey,
  clearPersistedActiveTask,
  persistActiveTask,
  readPersistedActiveTask,
  type TaskStorage,
} from './activeTaskPersistence.ts'

export interface TaskWatcherSocket {
  readonly readyState: number
  send(data: string): void
  close(): void
}

export interface TaskWatcherOptions<TResult = unknown> {
  jobId: string
  fetchSnapshot: () => Promise<GpuTaskResponse<TResult>>
  openSocket: (
    onSnapshot: (snapshot: Partial<GpuTaskResponse<TResult>>) => void,
    onClose: (event?: { code?: number }) => void,
  ) => TaskWatcherSocket
  onSnapshot: (snapshot: GpuTaskResponse<TResult>) => void
  onTerminal: (snapshot: GpuTaskResponse<TResult>) => void | Promise<void>
  onConnectionNote: (note: string) => void
  onError: (error: Error) => void
  acknowledgeAfterDelivery?: (snapshot: GpuTaskResponse<TResult>) => void | Promise<void>
  pollIntervalMs?: number
  heartbeatMs?: number
  reconnectBaseMs?: number
  maxTransientFailures?: number
  subscribeToWake?: (wake: () => void) => () => void
}

export interface TaskWatcher {
  start(): void
  wake(): void
  stop(): void
}

function toError(value: unknown, fallback: string): Error {
  return value instanceof Error ? value : new Error(fallback)
}

export function normalizeGpuTaskSnapshot<TResult>(
  previous: GpuTaskResponse<TResult>,
  incoming: Partial<GpuTaskResponse<TResult>>,
): GpuTaskResponse<TResult> {
  const terminalType = ['done', 'error', 'blocked', 'cancelled'].includes(incoming.type ?? '')
    ? incoming.type as GpuTaskResponse['status']
    : undefined
  return {
    ...previous,
    ...incoming,
    job_id: incoming.job_id ?? previous.job_id,
    status: incoming.status ?? terminalType ?? previous.status,
    progress: incoming.progress ?? incoming.pct ?? previous.progress,
    images: incoming.images ?? previous.images,
  }
}

function browserWakeSubscription(wake: () => void): () => void {
  if (typeof window === 'undefined' || typeof document === 'undefined') return () => undefined
  const visibleWake = () => {
    if (document.visibilityState === 'visible') wake()
  }
  window.addEventListener('online', wake)
  window.addEventListener('focus', wake)
  document.addEventListener('visibilitychange', visibleWake)
  return () => {
    window.removeEventListener('online', wake)
    window.removeEventListener('focus', wake)
    document.removeEventListener('visibilitychange', visibleWake)
  }
}

export function createTaskWatcher<TResult = unknown>(
  options: TaskWatcherOptions<TResult>,
): TaskWatcher {
  const pollIntervalMs = options.pollIntervalMs ?? 2500
  const heartbeatMs = options.heartbeatMs ?? 20000
  const reconnectBaseMs = options.reconnectBaseMs ?? 1000
  const maxTransientFailures = options.maxTransientFailures ?? 8

  let stopped = true
  let terminalDelivered = false
  let polling = false
  let transientFailures = 0
  let reconnectAttempts = 0
  let socket: TaskWatcherSocket | null = null
  let pollTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  let unsubscribeWake: (() => void) | null = null
  let latestSnapshot: GpuTaskResponse<TResult> = {
    job_id: options.jobId,
    status: 'queued',
    progress: 0,
    images: [],
  }

  const clearPollTimer = () => {
    if (pollTimer !== null) clearTimeout(pollTimer)
    pollTimer = null
  }
  const clearReconnectTimer = () => {
    if (reconnectTimer !== null) clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  const clearHeartbeat = () => {
    if (heartbeatTimer !== null) clearInterval(heartbeatTimer)
    heartbeatTimer = null
  }
  const closeSocket = () => {
    const current = socket
    socket = null
    if (current) {
      try { current.close() } catch { /* already closed */ }
    }
  }
  const stop = () => {
    if (stopped) return
    stopped = true
    clearPollTimer()
    clearReconnectTimer()
    clearHeartbeat()
    closeSocket()
    unsubscribeWake?.()
    unsubscribeWake = null
  }
  const failTerminally = (error: Error) => {
    if (stopped) return
    stop()
    options.onConnectionNote('')
    options.onError(error)
  }
  const deliver = (incoming: Partial<GpuTaskResponse<TResult>>) => {
    if (stopped || terminalDelivered) return
    const snapshot = normalizeGpuTaskSnapshot(latestSnapshot, incoming)
    latestSnapshot = snapshot
    options.onConnectionNote('')
    options.onSnapshot(snapshot)
    if (!isGpuTaskTerminal(snapshot.status)) return

    terminalDelivered = true
    stop()
    void Promise.resolve(options.onTerminal(snapshot))
      .then(() => options.acknowledgeAfterDelivery?.(snapshot))
      .catch(error => options.onError(toError(error, 'Could not deliver the GPU task result.')))
  }

  const schedulePoll = (delayMs: number) => {
    if (stopped || terminalDelivered || pollTimer !== null) return
    pollTimer = setTimeout(() => {
      pollTimer = null
      void poll()
    }, delayMs)
  }
  const poll = async () => {
    if (stopped || terminalDelivered || polling) return
    polling = true
    try {
      const snapshot = await options.fetchSnapshot()
      if (stopped) return
      transientFailures = 0
      deliver(snapshot)
      if (!terminalDelivered) schedulePoll(pollIntervalMs)
    } catch (error) {
      if (stopped) return
      if (responseStatus(error) === 404) {
        failTerminally(new Error('This GPU task is no longer available on the server.'))
        return
      }
      transientFailures += 1
      if (transientFailures > maxTransientFailures) {
        failTerminally(toError(error, 'Lost connection to the GPU task.'))
        return
      }
      options.onConnectionNote('Network is spotty. Krea is still trying to reconnect to this task.')
      schedulePoll(Math.min(500 * (2 ** (transientFailures - 1)), 5000))
    } finally {
      polling = false
    }
  }

  const scheduleReconnect = () => {
    if (stopped || terminalDelivered || reconnectTimer !== null) return
    const delay = Math.min(reconnectBaseMs * (2 ** reconnectAttempts), 15000)
    reconnectAttempts += 1
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, delay)
  }
  const connect = () => {
    if (stopped || terminalDelivered || socket !== null) return
    try {
      let opened: TaskWatcherSocket | null = null
      opened = options.openSocket(
        snapshot => {
          if (socket !== opened) return
          reconnectAttempts = 0
          deliver(snapshot)
        },
        event => {
          if (socket !== opened) return
          socket = null
          clearHeartbeat()
          if (stopped || terminalDelivered) return
          if (event?.code === 1008) {
            failTerminally(new Error('Lost access to this GPU task (the session expired or the task belongs to another user).'))
            return
          }
          options.onConnectionNote('Live connection dropped. Reconnecting while polling for updates.')
          scheduleReconnect()
          schedulePoll(0)
        },
      )
      socket = opened
      heartbeatTimer = setInterval(() => {
        if (socket?.readyState === 1) {
          try { socket.send('ping') } catch { /* close handler performs recovery */ }
        }
      }, heartbeatMs)
    } catch (error) {
      socket = null
      options.onConnectionNote('Live connection is unavailable. Polling while Krea reconnects.')
      scheduleReconnect()
      schedulePoll(0)
    }
  }

  const wake = () => {
    if (stopped || terminalDelivered) return
    clearPollTimer()
    clearReconnectTimer()
    if (socket === null) connect()
    void poll()
  }
  const start = () => {
    if (!stopped || terminalDelivered) return
    stopped = false
    unsubscribeWake = (options.subscribeToWake ?? browserWakeSubscription)(wake)
    connect()
    void poll()
  }

  return { start, wake, stop }
}

export { gpuTaskTerminalError, isGpuTaskTerminal }
