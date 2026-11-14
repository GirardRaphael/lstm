import { describe, expect, it, vi } from 'vitest'
import type { VideoAnalysis } from './learningTypes'
import { formatVideoTime, resolveVideoRecording, runVideoSamples, sampleVideoTimes, videoFileFingerprint, waitForVideoSeek } from './videoSampling'

const analysis: VideoAnalysis = { ok: true, domain: 'street', group_id: 'recording-a', policy: { trained: false, abstain: true } }

describe('video frame sampling', () => {
  it('collects evenly spaced frames from the chosen time without seeking past the clip', () => {
    expect(sampleVideoTimes(4, 12, 2, 25)).toEqual([4, 6, 8, 10])
    expect(sampleVideoTimes(0, 60, 2, 25)).toHaveLength(25)
  })

  it('bounds collection size and safely handles the end of a video', () => {
    expect(sampleVideoTimes(0, 1000, 1, 1000)).toHaveLength(60)
    expect(sampleVideoTimes(20, 10, 2, 25)).toEqual([9.95])
    expect(sampleVideoTimes(-5, 3, 1, 10)).toEqual([0, 1, 2])
  })

  it.each([[0, Infinity, 2, 25], [0, 0, 2, 25], [0, 10, 0, 25], [0, 10, 2, 0], [NaN, 10, 2, 25]])('rejects invalid sampling input %j', (start, duration, interval, count) => {
    expect(sampleVideoTimes(start, duration, interval, count)).toEqual([])
  })

  it('formats fractional timestamps and protects against invalid media times', () => {
    expect(formatVideoTime(65.2)).toBe('1:05.2')
    expect(formatVideoTime(59.98)).toBe('1:00.0')
    expect(formatVideoTime(Infinity)).toBe('0:00.0')
    expect(formatVideoTime(-1)).toBe('0:00.0')
  })
})

describe('recording identity', () => {
  it('keeps an uploaded copy of an included clip in its original recording group', () => {
    const example = { id: 'sample', title: 'Sample', video_url: '/video.mp4', recording_id: 'source-recording', source_url: 'https://source.example', license: 'CC BY', description: '', upload_fingerprint: 'known-hash' }
    expect(resolveVideoRecording('known-hash', [example])).toBe('source-recording')
    expect(resolveVideoRecording('different-hash', [example])).toBe('video-different-hash')
    expect(resolveVideoRecording('known-hash', [])).toBe('video-known-hash')
  })

  it('uses bytes rather than a filename so renamed copies stay together', async () => {
    const bytes = new Uint8Array([3, 4, 5, 6, 7])
    const first = new File([bytes], 'intersection-original.mp4', { lastModified: 1 })
    const renamed = new File([bytes], 'renamed-copy.mp4', { lastModified: 2 })
    expect(await videoFileFingerprint(first)).toBe(await videoFileFingerprint(renamed))
    expect(await videoFileFingerprint(first)).not.toBe(await videoFileFingerprint(new Blob([new Uint8Array([3, 4, 5, 6, 8])])))
  })
})

describe('serial frame inference and cancellation', () => {
  it('waits for each exact frame result before moving to the next frame', async () => {
    const events: string[] = []
    let current = 0
    let pendingRequests = 0
    const results: number[] = []
    await runVideoSamples({ times: [1, 3, 5], signal: new AbortController().signal,
      seek: async time => { expect(pendingRequests).toBe(0); current = time; events.push(`seek:${time}`) },
      capture: () => ({ image: `frame-${current}`, timestamp: current }),
      onCapture: frame => events.push(`capture:${frame.timestamp}`),
      analyze: async frame => { pendingRequests++; await Promise.resolve(); expect(frame.image).toBe(`frame-${frame.timestamp}`); events.push(`result:${frame.timestamp}`); pendingRequests--; return analysis },
      onResult: (frame, result, completed) => { expect(result).toBe(analysis); expect(completed).toBe(results.length + 1); results.push(frame.timestamp) },
    })
    expect(results).toEqual([1, 3, 5])
    expect(events).toEqual(['seek:1', 'capture:1', 'result:1', 'seek:3', 'capture:3', 'result:3', 'seek:5', 'capture:5', 'result:5'])
  })

  it('never displays a late result after observation is stopped or the video changes', async () => {
    const controller = new AbortController()
    const onResult = vi.fn()
    const seek = vi.fn(async () => undefined)
    await expect(runVideoSamples({ times: [1, 3], signal: controller.signal, seek,
      capture: () => ({ image: 'old-frame', timestamp: 1 }), onCapture: vi.fn(),
      analyze: async () => { controller.abort(); return analysis }, onResult,
    })).rejects.toMatchObject({ name: 'AbortError' })
    expect(onResult).not.toHaveBeenCalled()
    expect(seek).toHaveBeenCalledTimes(1)
  })

  it('does not capture an undecoded frame when cancelled during seeking', async () => {
    const controller = new AbortController()
    const capture = vi.fn(() => ({ image: '', timestamp: 0 }))
    const analyze = vi.fn(async () => analysis)
    await expect(runVideoSamples({ times: [1], signal: controller.signal,
      seek: async () => { controller.abort() }, capture, analyze, onCapture: vi.fn(), onResult: vi.fn(),
    })).rejects.toMatchObject({ name: 'AbortError' })
    expect(capture).not.toHaveBeenCalled()
    expect(analyze).not.toHaveBeenCalled()
  })

  it('stops sampling after an inference failure instead of pairing old results with new frames', async () => {
    const onResult = vi.fn()
    const seek = vi.fn(async () => undefined)
    await expect(runVideoSamples({ times: [1, 3], signal: new AbortController().signal, seek,
      capture: () => ({ image: 'frame', timestamp: 1 }), onCapture: vi.fn(),
      analyze: async () => { throw new Error('Detector unavailable') }, onResult,
    })).rejects.toThrow('Detector unavailable')
    expect(seek).toHaveBeenCalledTimes(1)
    expect(onResult).not.toHaveBeenCalled()
  })
})

describe('video decoding boundary', () => {
  it('waits for seek completion before a frame becomes capturable', async () => {
    const media = Object.assign(new EventTarget(), { currentTime: 0, readyState: 2, seeking: true, pause: vi.fn() })
    let complete = false
    const waiting = waitForVideoSeek(media as unknown as HTMLVideoElement, 3, new AbortController().signal).then(() => { complete = true })
    await Promise.resolve()
    expect(complete).toBe(false)
    expect(media.currentTime).toBe(3)
    media.seeking = false
    media.dispatchEvent(new Event('seeked'))
    await waiting
    expect(complete).toBe(true)
    expect(media.pause).toHaveBeenCalled()
  })

  it('cancels a pending seek and ignores its later media event', async () => {
    const media = Object.assign(new EventTarget(), { currentTime: 0, readyState: 1, seeking: true, pause: vi.fn() })
    const controller = new AbortController()
    const waiting = waitForVideoSeek(media as unknown as HTMLVideoElement, 3, controller.signal)
    controller.abort()
    await expect(waiting).rejects.toMatchObject({ name: 'AbortError' })
    media.seeking = false; media.readyState = 2
    media.dispatchEvent(new Event('seeked'))
  })
})
