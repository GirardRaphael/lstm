import type { Analysis, AnomalyReview, VideoAnalysis, VideoMetadata } from './learningTypes'

export type VideoExample = { id: string; title: string; video_url: string; recording_id: string; source_url: string; license: string; description: string; attribution?: string; upload_fingerprint?: string }
export type ReviewFrame = { snapshot_id: string; image_url: string; video: VideoMetadata; domain: 'street'; group_id: string; reviewed_action?: string | null; anomaly_review?: AnomalyReview | null; captured_at?: string }
export type CapturedFrame = { image: string; timestamp: number }

export function formatVideoTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0'
  const rounded = Math.round(seconds * 10) / 10
  const minutes = Math.floor(rounded / 60)
  return `${minutes}:${(rounded % 60).toFixed(1).padStart(4, '0')}`
}

// Keep the last seek inside the playable range: the duration itself may have no decoded frame.
export function sampleVideoTimes(start: number, duration: number, interval: number, limit: number): number[] {
  if (![start, duration, interval, limit].every(Number.isFinite) || duration <= 0 || interval < .25 || limit < 1) return []
  const end = Math.max(0, duration - .05)
  const first = Math.min(end, Math.max(0, start))
  const times: number[] = []
  for (let index = 0; index < Math.min(60, Math.floor(limit)); index++) {
    const time = first + index * interval
    if (time > end) break
    times.push(Math.round(time * 1000) / 1000)
  }
  return times
}

export async function videoFileFingerprint(file: Blob): Promise<string> {
  const blockSize = 64 * 1024
  const starts = [0, Math.max(0, Math.floor((file.size - blockSize) / 2)), Math.max(0, file.size - blockSize)]
  const blocks = await Promise.all(starts.map(start => file.slice(start, Math.min(file.size, start + blockSize)).arrayBuffer()))
  const prefix = new TextEncoder().encode(`video-size:${file.size};`)
  const bytes = new Uint8Array(prefix.length + blocks.reduce((size, block) => size + block.byteLength, 0))
  bytes.set(prefix)
  let offset = prefix.length
  for (const block of blocks) { bytes.set(new Uint8Array(block), offset); offset += block.byteLength }
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

export function abortIfNeeded(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException('Stopped', 'AbortError')
}

export function resolveVideoRecording(fingerprint: string, examples: VideoExample[]): string {
  return examples.find(example => example.upload_fingerprint === fingerprint)?.recording_id ?? `video-${fingerprint}`
}

export function waitForVideoSeek(video: HTMLVideoElement, timestamp: number, signal: AbortSignal): Promise<void> {
  abortIfNeeded(signal)
  video.pause()
  if (Math.abs(video.currentTime - timestamp) < .025 && video.readyState >= 2 && !video.seeking) return Promise.resolve()
  return new Promise((resolve, reject) => {
    let timeout: ReturnType<typeof setTimeout>
    const cleanup = () => {
      clearTimeout(timeout)
      video.removeEventListener('seeked', ready)
      video.removeEventListener('loadeddata', ready)
      video.removeEventListener('error', failed)
      signal.removeEventListener('abort', stopped)
    }
    const ready = () => { if (video.readyState >= 2 && !video.seeking) { cleanup(); resolve() } }
    const failed = () => { cleanup(); reject(new Error('This video frame could not be decoded. Try a different time or video.')) }
    const stopped = () => { cleanup(); reject(new DOMException('Stopped', 'AbortError')) }
    video.addEventListener('seeked', ready)
    video.addEventListener('loadeddata', ready)
    video.addEventListener('error', failed)
    signal.addEventListener('abort', stopped, { once: true })
    timeout = setTimeout(failed, 20_000)
    try { video.currentTime = timestamp } catch (error) { cleanup(); reject(error) }
  })
}

export function captureVideoFrame(video: HTMLVideoElement): CapturedFrame {
  if (!video.videoWidth || !video.videoHeight || video.readyState < 2 || video.seeking) throw new Error('Wait for this video frame to finish loading.')
  const scale = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('This browser could not capture the video frame.')
  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return { image: canvas.toDataURL('image/jpeg', .93), timestamp: video.currentTime }
}

export async function runVideoSamples<T extends VideoAnalysis | Analysis>(options: {
  times: number[]; signal: AbortSignal
  seek: (time: number, signal: AbortSignal) => Promise<void>
  capture: () => CapturedFrame
  analyze: (frame: CapturedFrame, signal: AbortSignal) => Promise<T>
  onCapture: (frame: CapturedFrame) => void
  onResult: (frame: CapturedFrame, result: T, completed: number) => void
  afterResult?: (signal: AbortSignal) => Promise<void>
}): Promise<void> {
  for (let index = 0; index < options.times.length; index++) {
    abortIfNeeded(options.signal)
    await options.seek(options.times[index], options.signal)
    abortIfNeeded(options.signal)
    const frame = options.capture()
    options.onCapture(frame)
    const result = await options.analyze(frame, options.signal)
    abortIfNeeded(options.signal)
    options.onResult(frame, result, index + 1)
    if (index < options.times.length - 1 && options.afterResult) await options.afterResult(options.signal)
  }
}

export function videoObservationDelay(signal: AbortSignal): Promise<void> {
  abortIfNeeded(signal)
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { signal.removeEventListener('abort', stopped); resolve() }, 1000)
    const stopped = () => { clearTimeout(timeout); reject(new DOMException('Stopped', 'AbortError')) }
    signal.addEventListener('abort', stopped, { once: true })
  })
}
