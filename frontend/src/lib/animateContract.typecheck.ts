import type {
  AnimateRequest,
  AnimationResult,
  AnimationUploadResponse,
  GpuTaskResponse,
} from '../api'

declare const request: AnimateRequest
declare const result: AnimationResult
declare const upload: AnimationUploadResponse
declare const snapshot: GpuTaskResponse<AnimationResult>

request.prompt_schedule satisfies string
request.render_frames satisfies number | null
request.animation_mode satisfies '2D' | '3D' | 'Video Input' | 'None'
result.video_url satisfies string
result.gallery_id satisfies number
upload.upload_id satisfies string
snapshot.result satisfies AnimationResult | null | undefined
