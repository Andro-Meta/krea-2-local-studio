import type { QueueJob } from '../api'
import { reconcilePendingCancellations } from './queueState'

const jobs: QueueJob[] = []
const pending = reconcilePendingCancellations(new Set(['job']), jobs)
const checked: Set<string> = pending
void checked
