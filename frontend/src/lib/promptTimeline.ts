/** Seconds-based Animate prompt timeline helpers. */

export type PromptRow = {
  id: string
  seconds: number
  prompt: string
}

export function pythonRound(value: number): number {
  if (!Number.isFinite(value)) return NaN
  return Math.sign(value) * Math.round(Math.abs(value))
}

export function totalFramesFromTiming(
  durationSeconds: number,
  fps: number,
  renderFrames: number | null,
): number {
  if (renderFrames != null) return renderFrames
  return pythonRound(durationSeconds * fps)
}

export function secondsToFrame(seconds: number, fps: number, totalFrames: number): number {
  if (!Number.isFinite(seconds) || !Number.isFinite(fps) || totalFrames < 1) return 0
  const frame = pythonRound(seconds * fps)
  return Math.min(Math.max(0, frame), totalFrames - 1)
}

export function frameToSeconds(frame: number, fps: number): number {
  if (!Number.isFinite(frame) || !Number.isFinite(fps) || fps <= 0) return 0
  return frame / fps
}

let _rowId = 0
export function newPromptRowId(): string {
  _rowId += 1
  return `pr-${_rowId}-${Date.now().toString(36)}`
}

export function parsePromptScheduleToRows(
  schedule: string,
  fps: number,
  totalFrames: number,
): PromptRow[] {
  const lines = schedule.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
  if (!lines.length) {
    return [{ id: newPromptRowId(), seconds: 0, prompt: '' }]
  }
  const rows: PromptRow[] = []
  for (const line of lines) {
    const match = line.match(/^\s*(-?\d+)\s*:\s*(.+?)\s*$/)
    if (!match) {
      rows.push({ id: newPromptRowId(), seconds: 0, prompt: line })
      continue
    }
    const frame = Number(match[1])
    const clamped = Number.isFinite(frame)
      ? Math.min(Math.max(0, frame), Math.max(0, totalFrames - 1))
      : 0
    rows.push({
      id: newPromptRowId(),
      seconds: frameToSeconds(clamped, fps),
      prompt: match[2],
    })
  }
  return rows.sort((a, b) => a.seconds - b.seconds || a.prompt.localeCompare(b.prompt))
}

export function serializePromptRows(
  rows: PromptRow[],
  fps: number,
  totalFrames: number,
): string {
  const seen = new Set<number>()
  const entries: Array<{ frame: number; prompt: string }> = []
  for (const row of rows) {
    const prompt = row.prompt.trim()
    if (!prompt) continue
    const frame = secondsToFrame(row.seconds, fps, totalFrames)
    if (seen.has(frame)) continue
    seen.add(frame)
    entries.push({ frame, prompt })
  }
  entries.sort((a, b) => a.frame - b.frame)
  if (!entries.length) return '0: '
  return entries.map(entry => `${entry.frame}: ${entry.prompt}`).join('\n')
}

/** Keep row seconds stable when FPS/duration change; clamp past the new end. */
export function rescalePromptRows(
  rows: PromptRow[],
  durationSeconds: number,
): PromptRow[] {
  const maxSeconds = Math.max(0, durationSeconds)
  return rows
    .map(row => ({
      ...row,
      seconds: Math.min(Math.max(0, row.seconds), maxSeconds),
    }))
    .sort((a, b) => a.seconds - b.seconds)
}

export function estimatedQueueTurns(totalFrames: number, chunkSize = 8): number {
  if (!Number.isFinite(totalFrames) || totalFrames < 1) return 1
  return Math.ceil(totalFrames / Math.max(1, chunkSize))
}
