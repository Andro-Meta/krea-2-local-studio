import type { AnimateRequest } from '../../api'
import type { AnimateValidationErrors } from './presets'

export interface AnimateControlProps {
  value: AnimateRequest
  errors: AnimateValidationErrors
  disabled?: boolean
  update: <K extends keyof AnimateRequest>(key: K, value: AnimateRequest[K]) => void
}
