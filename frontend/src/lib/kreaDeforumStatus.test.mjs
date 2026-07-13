import assert from 'node:assert/strict'
import test from 'node:test'
import { normalizeKreaDeforumStatus } from './kreaDeforumStatus.ts'

const empty = {
  available: false,
  missing_nodes: [],
  incompatible_capabilities: [],
  variants: [],
  revision: 'unknown',
  external: true,
  license: 'unspecified',
  patch_version: 'unknown',
  probe_failed: false,
  stale: false,
  midas_ready: false,
  midas_reason: 'MiDaS readiness was not reported by the backend.',
}

test('normalizes absent and malformed diagnostics without throwing', () => {
  assert.deepEqual(normalizeKreaDeforumStatus(undefined), empty)
  assert.deepEqual(normalizeKreaDeforumStatus(null), empty)
  assert.deepEqual(normalizeKreaDeforumStatus({ missing_nodes: 'bad', revision: 7 }), empty)
})

test('preserves valid diagnostic fields and filters malformed arrays', () => {
  assert.deepEqual(normalizeKreaDeforumStatus({
    available: true,
    missing_nodes: ['MissingA', 7, 'MissingB'],
    incompatible_capabilities: ['bad patch', null],
    variants: ['chunked', 1],
    revision: 'abc123',
    external: false,
    license: 'custom',
    patch_version: 'v2',
    probe_failed: true,
    stale: true,
    midas_ready: true,
    midas_reason: 'Ready marker and cache verified.',
  }), {
    available: true,
    missing_nodes: ['MissingA', 'MissingB'],
    incompatible_capabilities: ['bad patch'],
    variants: ['chunked'],
    revision: 'abc123',
    external: false,
    license: 'custom',
    patch_version: 'v2',
    probe_failed: true,
    stale: true,
    midas_ready: true,
    midas_reason: 'Ready marker and cache verified.',
  })
})
