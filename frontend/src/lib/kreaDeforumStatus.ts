import type { KreaDeforumStatus } from '../api'

const DEFAULT_STATUS: KreaDeforumStatus = {
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

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

function nonemptyString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim() ? value : fallback
}

export function normalizeKreaDeforumStatus(value: unknown): KreaDeforumStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { ...DEFAULT_STATUS }
  const input = value as Record<string, unknown>
  return {
    available: input.available === true,
    missing_nodes: stringArray(input.missing_nodes),
    incompatible_capabilities: stringArray(input.incompatible_capabilities),
    variants: stringArray(input.variants),
    revision: nonemptyString(input.revision, DEFAULT_STATUS.revision),
    external: typeof input.external === 'boolean' ? input.external : DEFAULT_STATUS.external,
    license: nonemptyString(input.license, DEFAULT_STATUS.license),
    patch_version: nonemptyString(input.patch_version, DEFAULT_STATUS.patch_version),
    probe_failed: input.probe_failed === true,
    stale: input.stale === true,
    midas_ready: input.midas_ready === true,
    midas_reason: nonemptyString(input.midas_reason, DEFAULT_STATUS.midas_reason),
    ...(typeof input.patched_animator_sha256 === 'string'
      ? { patched_animator_sha256: input.patched_animator_sha256 }
      : {}),
    ...(typeof input.patch_sha256 === 'string'
      ? { patch_sha256: input.patch_sha256 }
      : {}),
  }
}
