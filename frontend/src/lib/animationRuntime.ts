export function clearConsumedVideoUpload<T extends {
  animation_mode: string
  source_video_upload_id: string
}>(form: T): { form: T; videoTransferred: boolean } {
  const videoTransferred = form.animation_mode === 'Video Input'
    && form.source_video_upload_id.trim().length > 0
  return {
    form: { ...form, source_video_upload_id: '' },
    videoTransferred,
  }
}

export function submissionFailureKeepsUpload(error: unknown): boolean {
  const status = (error as { response?: { status?: unknown } })?.response?.status
  return typeof status === 'number' && Number.isInteger(status)
}
