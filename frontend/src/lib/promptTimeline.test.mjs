import assert from 'node:assert/strict'
import {
  estimatedQueueTurns,
  frameToSeconds,
  parsePromptScheduleToRows,
  rescalePromptRows,
  secondsToFrame,
  serializePromptRows,
  totalFramesFromTiming,
} from './promptTimeline.ts'

assert.equal(totalFramesFromTiming(4, 12, null), 48)
assert.equal(secondsToFrame(2, 12, 48), 24)
assert.equal(frameToSeconds(24, 12), 2)
assert.equal(secondsToFrame(10, 12, 48), 47)

const rows = parsePromptScheduleToRows('0: dawn\n24: dusk', 12, 48)
assert.equal(rows.length, 2)
assert.equal(rows[0].seconds, 0)
assert.equal(rows[1].seconds, 2)
assert.equal(serializePromptRows(rows, 12, 48), '0: dawn\n24: dusk')
assert.equal(serializePromptRows(rows, 24, 96), '0: dawn\n48: dusk')

const rescaled = rescalePromptRows(rows, 1.5)
assert.equal(rescaled[1].seconds, 1.5)
assert.equal(secondsToFrame(1.5, 12, 18), 17)
assert.equal(serializePromptRows(rescaled, 12, 18), '0: dawn\n17: dusk')

assert.equal(estimatedQueueTurns(48, 8), 6)
assert.equal(estimatedQueueTurns(8, 8), 1)

console.log('promptTimeline tests passed')
