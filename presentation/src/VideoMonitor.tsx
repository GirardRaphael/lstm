import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import type { Action, Analysis, Detection, VideoAnalysis, VideoMetadata } from './learningTypes'
import { abortIfNeeded, captureVideoFrame, formatVideoTime, resolveVideoRecording, runVideoSamples, sampleVideoTimes, videoFileFingerprint, videoObservationDelay, waitForVideoSeek } from './videoSampling'
import type { CapturedFrame, ReviewFrame, VideoExample } from './videoSampling'
import './VideoMonitor.css'

type Props = { onReview: (analysis: Analysis) => void; onOpenTraining?: () => void; active?: boolean }
type Source = { url: string; title: string; recording: string; description?: string; attribution?: string; source_url?: string; license?: string; sampleId?: string }
type Preview = CapturedFrame & { result?: VideoAnalysis }
type HistoryItem = { key: number; timestamp: number; result: VideoAnalysis }
type Task = 'observe' | 'collect' | 'analyze' | 'capture' | 'review' | 'upload' | ''
const ACTIONS: { id: Action; name: string }[] = [{ id: 'ns', name: 'North / south' }, { id: 'ew', name: 'East / west' }, { id: 'walk', name: 'Pedestrian walk' }, { id: 'hold', name: 'Hold all red' }]
const actionName = (action?: string | null) => ACTIONS.find(item => item.id === action)?.name ?? 'No recommendation'
const percent = (value?: number | null) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Unavailable'
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'The request could not be completed.'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/learning/${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } })
  let body: Record<string, unknown>
  try { body = await response.json() as Record<string, unknown> } catch { throw new Error('The learning service is unavailable. Start the app service, then try again.') }
  if (!response.ok || body.ok === false) {
    const detail = [body.reason, body.error, body.detail, body.message].find(value => typeof value === 'string')
    throw new Error(typeof detail === 'string' ? detail : `The request failed (${response.status}).`)
  }
  return body as T
}

function DetectionOverlay({ detection }: { detection?: Detection }) {
  const [width, height] = detection?.image_size ?? [0, 0]
  if (!detection?.available || width <= 0 || height <= 0) return null
  const fontSize = Math.max(12, width / 65)
  return <svg className="video-detection-overlay" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Detected objects with raw model scores"><title>Estimated object boxes on this exact frozen frame</title>{detection.objects?.map((object, index) => {
    const [left, top, boxWidth, boxHeight] = object.box
    if (![left, top, boxWidth, boxHeight, object.score].every(Number.isFinite) || boxWidth <= 0 || boxHeight <= 0) return null
    const x = Math.max(0, Math.min(1, left)) * width
    const y = Math.max(0, Math.min(1, top)) * height
    const color = object.class_name === 'person' ? '#f6bd60' : '#55e3d4'
    return <g key={`${object.class_id}-${index}`}><rect x={x} y={y} width={Math.min(boxWidth * width, width - x)} height={Math.min(boxHeight * height, height - y)} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /><text x={Math.min(x + 3, Math.max(0, width - fontSize * 10))} y={Math.max(fontSize, y - 5)} fill={color} stroke="#061016" strokeWidth={fontSize * .25} paintOrder="stroke" fontSize={fontSize} fontWeight="700">{object.class_name} · {percent(object.score)}</text></g>
  })}</svg>
}

function ModelEvidence({ result }: { result?: VideoAnalysis }) {
  const policy = result?.policy
  const detection = result?.detection
  const anomaly = result?.anomaly
  const counts = Object.entries(detection?.counts ?? {}).filter(([, count]) => count > 0)
  const recommendation = policy?.trained && policy.action && !policy.abstain ? policy.action : null
  return <aside className="video-evidence" aria-label="Evidence for this video frame">
    <p className="video-kicker">Visible evidence · proposed action</p><h2>What the model reports</h2>
    {!result ? <p>Analyze a frame or start observation. Results will describe the exact image displayed beside this panel.</p> : <>
      <section><h3>1. Object recognition</h3><strong>{detection?.available ? detection.model ?? 'Pretrained visual detector' : 'Detector unavailable'}</strong>
        {detection?.available ? <><div className="video-object-counts">{counts.length ? counts.map(([name, count]) => <span key={name}><b>{count}</b> {name}{count === 1 ? '' : 's'}</span>) : <p>No objects passed the detection threshold. This does not mean the road is empty.</p>}</div><p className="video-note">Raw score threshold {percent(detection.score_threshold)}{typeof detection.latency_ms === 'number' ? ` · ${Math.round(detection.latency_ms)} ms inference` : ''}. Boxes are estimates; their scores are not measured reliability.</p><p className="video-recognition-limit">Check the boxes against the image. Small aerial vehicles and occluded objects can be missed. These are detected-object counts, not verified traffic totals or waiting queues.</p></> : <p>{detection?.reason ?? 'No detection result is available.'}</p>}
      </section>
      <section><h3>2. Anomaly review signal</h3><strong>{typeof anomaly?.anomaly_probability === 'number' ? `${percent(anomaly.anomaly_probability)} anomaly score` : 'No anomaly estimate available'}</strong><p>{typeof anomaly?.anomaly_probability === 'number' ? anomaly.decision === 'review' ? 'Flagged for human review.' : 'Not flagged at the model’s current threshold.' : anomaly?.reason ?? 'No anomaly result was returned for this frame.'}</p>{typeof anomaly?.threshold === 'number' ? <p className="video-note">Review threshold: {percent(anomaly.threshold)}.</p> : null}<p className="video-note">{anomaly?.model_version ? `${anomaly.model_version}. ` : ''}This estimate is not validated for this camera. An anomaly score does not prove a traffic violation, and a low score does not establish that the scene is safe.</p></section>
      <section><h3>3. Learned signal preference</h3><strong className="video-proposal">{!policy?.trained ? 'Needs your reviewed examples' : recommendation ? actionName(recommendation) : 'Abstained · no recommendation'}</strong>
        <p>{policy?.reason ?? 'Review examples from several signalized intersections, then train the street decision model in the teaching workspace.'}</p>
        {policy?.probabilities ? <><ul className="video-probabilities">{ACTIONS.map(action => <li key={action.id}><span>{action.name}</span><strong>{percent(policy.probabilities?.[action.id])}</strong></li>)}</ul><p className="video-note">{policy.calibrated ? 'Probabilities use the fitted calibration model.' : 'These model scores have not been calibrated.'} Displayed scores describe learned human preferences; they do not establish safe or optimal signal timing.</p></> : null}
        {policy?.trained ? <p className="video-note">Recommendation threshold: {percent(policy.confidence_threshold)}. {policy.target_met ? 'The recorded held-out agreement target was met within that evaluation scope.' : 'Reliability on this intersection has not been demonstrated.'}</p> : null}
      </section>
      <section><h3>4. Observation only</h3><p>No command is sent to this intersection or to the live simulator. A frozen image cannot establish vehicle waiting time, the full traffic-light cycle, or right of way on its own.</p></section>
      <details><summary>Model and result details</summary><dl><dt>Detector version</dt><dd>{detection?.model_version ?? 'Unavailable'}</dd><dt>Decision model version</dt><dd>{policy?.model_version ?? 'Untrained'}</dd><dt>Recording group</dt><dd>{result.group_id}</dd></dl><pre>{JSON.stringify({ detection: result.detection, anomaly: result.anomaly, policy: result.policy }, null, 2)}</pre></details>
    </>}
  </aside>
}

export function VideoMonitor({ onReview, onOpenTraining, active = true }: Props) {
  const [examples, setExamples] = useState<VideoExample[]>([])
  const [examplesLoaded, setExamplesLoaded] = useState(false)
  const [source, setSource] = useState<Source | null>(null)
  const [ready, setReady] = useState(false)
  const [duration, setDuration] = useState(0)
  const [position, setPosition] = useState(0)
  const [interval, setInterval] = useState(2)
  const [collectionSize, setCollectionSize] = useState(25)
  const [task, setTask] = useState<Task>('')
  const [preview, setPreview] = useState<Preview | null>(null)
  const [loadedPreview, setLoadedPreview] = useState('')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [frames, setFrames] = useState<ReviewFrame[]>([])
  const [frameTotal, setFrameTotal] = useState(0)
  const [galleryPage, setGalleryPage] = useState(0)
  const [galleryLoading, setGalleryLoading] = useState(false)
  const [progress, setProgress] = useState({ completed: 0, total: 0 })
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const objectUrl = useRef<string | null>(null)
  const operation = useRef<AbortController | null>(null)
  const historyKey = useRef(0)
  const sourceRef = useRef<Source | null>(null)
  const inFlight = useRef(false)

  useEffect(() => {
    if (!active || examplesLoaded) return
    const controller = new AbortController()
    void request<{ videos: VideoExample[] }>('video-examples', { signal: controller.signal }).then(body => {
      setExamples(body.videos); setExamplesLoaded(true)
    }).catch(error => { if (!controller.signal.aborted) { setError(errorMessage(error)); setExamplesLoaded(true) } })
    return () => controller.abort()
  }, [active, examplesLoaded])

  useEffect(() => () => { operation.current?.abort(); if (objectUrl.current) URL.revokeObjectURL(objectUrl.current) }, [])

  useEffect(() => {
    if (active) return
    operation.current?.abort(); videoRef.current?.pause()
  }, [active])

  useEffect(() => {
    if (!source || !active) return
    const controller = new AbortController()
    void request<{ frames: ReviewFrame[]; total: number }>(`review-frames?group_id=${encodeURIComponent(source.recording)}&offset=${galleryPage * 100}&limit=100`, { signal: controller.signal }).then(body => { setFrames(body.frames); setFrameTotal(body.total) }).catch(error => { if (!controller.signal.aborted) setError(errorMessage(error)) }).finally(() => { if (!controller.signal.aborted) setGalleryLoading(false) })
    return () => controller.abort()
  }, [source, active, galleryPage])

  const stop = () => { operation.current?.abort(); operation.current = null; videoRef.current?.pause(); setTask(''); setNotice('Stopped. Completed observations and saved training pictures are kept.') }
  const chooseSource = (next: Source) => {
    operation.current?.abort(); operation.current = null; videoRef.current?.pause()
    sourceRef.current = next; setSource(next); setReady(false); setDuration(0); setPosition(0); setPreview(null); setHistory([]); setFrames([]); setFrameTotal(0); setGalleryPage(0); setGalleryLoading(false); setTask(''); setError(''); setNotice(''); setProgress({ completed: 0, total: 0 })
  }
  const selectExample = (id: string) => {
    const item = examples.find(example => example.id === id)
    if (!item) return
    if (objectUrl.current) { URL.revokeObjectURL(objectUrl.current); objectUrl.current = null }
    chooseSource({ url: item.video_url, title: item.title, recording: item.recording_id, description: item.description, attribution: item.attribution, source_url: item.source_url, license: item.license, sampleId: item.id })
  }
  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    operation.current?.abort()
    const controller = new AbortController(); operation.current = controller
    setTask('upload'); setError('')
    try {
      const fingerprint = await videoFileFingerprint(file)
      const knownExamples = examplesLoaded ? examples : (await request<{ videos: VideoExample[] }>('video-examples', { signal: controller.signal })).videos
      abortIfNeeded(controller.signal)
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = URL.createObjectURL(file)
      const knownExample = knownExamples.find(example => example.upload_fingerprint === fingerprint)
      chooseSource({ url: objectUrl.current, title: file.name, recording: resolveVideoRecording(fingerprint, knownExamples), description: knownExample?.description ?? 'Your local video. Captured frames are processed by the local learning service.', license: knownExample?.license ?? 'User-provided recording; permissions not verified', source_url: knownExample?.source_url, attribution: knownExample?.attribution })
    } catch (error) { if (!controller.signal.aborted) setError(errorMessage(error)) }
    finally { if (operation.current === controller) { operation.current = null; setTask('') } }
    event.target.value = ''
  }

  const addSavedFrame = (analysis: Analysis) => {
    if (!analysis.image_url || !analysis.video) return
    setFrames(previous => {
      if (previous.some(frame => frame.snapshot_id === analysis.snapshot_id)) return previous
      return [{ snapshot_id: analysis.snapshot_id, image_url: analysis.image_url!, video: analysis.video!, domain: 'street' as const, group_id: analysis.group_id, reviewed_action: null, anomaly_review: analysis.anomaly_review ?? null }, ...previous].slice(0, 100)
    })
  }

  const run = async (kind: 'observe' | 'collect' | 'analyze' | 'capture') => {
    const media = videoRef.current
    const selected = source
    if (!active || !media || !selected || !ready || inFlight.current) return
    operation.current?.abort()
    const controller = new AbortController(); operation.current = controller
    inFlight.current = true; media.pause(); setTask(kind); setError(''); setNotice('')
    const persist = kind === 'collect' || kind === 'capture'
    const times = sampleVideoTimes(media.currentTime, duration, interval, kind === 'collect' ? collectionSize : kind === 'observe' ? 60 : 1)
    setProgress({ completed: 0, total: kind === 'observe' ? 0 : times.length })
    let completed = 0
    let nextTimes = times
    let latestSaved: Analysis | null = null
    try {
      while (nextTimes.length) {
        await runVideoSamples<VideoAnalysis>({
          times: nextTimes, signal: controller.signal,
          seek: (time, signal) => waitForVideoSeek(media, time, signal),
          capture: () => captureVideoFrame(media),
          onCapture: frame => { setPreview(frame); setPosition(frame.timestamp) },
          analyze: async (frame, signal) => {
            const video: VideoMetadata = { recording_id: selected.recording, name: selected.title, timestamp_seconds: frame.timestamp, ...(selected.source_url ? { source_url: selected.source_url } : {}), ...(selected.license ? { license: selected.license } : {}) }
            const result = await request<VideoAnalysis & { duplicate?: boolean }>(persist ? 'analyze' : 'monitor', { method: 'POST', signal, body: JSON.stringify({ domain: 'street', group_id: selected.recording, video, image_data_url: frame.image }) })
            if (result.group_id !== selected.recording || !result.video || (Math.abs(result.video.timestamp_seconds - frame.timestamp) > .1 && !(persist && result.duplicate))) throw new Error('The returned result does not match this video frame. Observation has stopped.')
            return result
          },
          onResult: (frame, result) => {
            completed++
            setPreview({ ...frame, result }); setProgress({ completed, total: kind === 'observe' ? 0 : times.length })
            setHistory(previous => [{ key: ++historyKey.current, timestamp: frame.timestamp, result }, ...previous].slice(0, 100))
            if (persist && result.snapshot_id) { latestSaved = result as Analysis; addSavedFrame(latestSaved) }
          },
          ...(kind === 'observe' ? { afterResult: videoObservationDelay } : {}),
        })
        if (kind !== 'observe') break
        const next = nextTimes[nextTimes.length - 1] + interval
        if (next > duration - .05) break
        nextTimes = sampleVideoTimes(next, duration, interval, 60)
      }
      abortIfNeeded(controller.signal)
      setNotice(persist ? `${completed} frame${completed === 1 ? '' : 's'} processed. Saved pictures still need your action labels; collecting pictures does not train the model.` : `${completed} frame${completed === 1 ? '' : 's'} observed. Monitoring does not save training pictures or change any light.`)
      if (persist) {
        const refreshed = await request<{ frames: ReviewFrame[]; total: number }>(`review-frames?group_id=${encodeURIComponent(selected.recording)}&limit=100`, { signal: controller.signal })
        abortIfNeeded(controller.signal); setFrames(refreshed.frames); setFrameTotal(refreshed.total); setGalleryPage(0)
      }
      if (kind === 'capture' && latestSaved) onReview(latestSaved)
    } catch (error) { if (!controller.signal.aborted) setError(errorMessage(error)) }
    finally { inFlight.current = false; if (operation.current === controller) { operation.current = null; setTask('') } }
  }

  const scrub = (time: number) => {
    stop(); setPreview(null); setPosition(time); setNotice('')
    if (videoRef.current) videoRef.current.currentTime = time
  }
  const showOriginal = () => { stop(); setPreview(null); setNotice('Original footage. Pause or choose a time, then analyze that frame.'); void videoRef.current?.play().catch(() => setNotice('Press play on the video to watch the original footage.')) }
  const openReview = async (snapshot: string) => {
    if (!active || inFlight.current) return
    operation.current?.abort()
    const controller = new AbortController(); operation.current = controller
    setTask('review'); setError('')
    try { const analysis = await request<Analysis>(`review-frame/${encodeURIComponent(snapshot)}`, { signal: controller.signal }); abortIfNeeded(controller.signal); onReview(analysis) }
    catch (error) { if (!controller.signal.aborted) setError(errorMessage(error)) }
    finally { if (operation.current === controller) { operation.current = null; setTask('') } }
  }
  const busy = Boolean(task) || galleryLoading
  const collectionTimes = sampleVideoTimes(position, duration, interval, collectionSize)
  const result = preview?.result
  const shownFrameCount = frames.length
  const savedFrameCount = Math.max(frameTotal, shownFrameCount)

  return <section className="video-monitor" aria-label="Real traffic video monitor">
    <header className="video-header"><div><p className="video-kicker">Real footage · local visual recognition</p><h1>Watch. Inspect. Teach.</h1><p>Observe a real traffic-light intersection, inspect the model’s evidence, and collect exact frames to review. This workspace never operates the lights in the video.</p></div>{onOpenTraining ? <button type="button" onClick={onOpenTraining}>Open teaching workspace</button> : null}</header>
    <section className="video-source-card" aria-label="Choose traffic footage"><div><label htmlFor="video-example">Included reusable footage</label><select id="video-example" value={source?.sampleId ?? ''} onChange={event => selectExample(event.target.value)} disabled={busy}><option value="" disabled>{examplesLoaded ? examples.length ? 'Choose a sample video' : 'No sample video is installed yet' : 'Loading sample videos…'}</option>{examples.map(example => <option key={example.id} value={example.id}>{example.title}</option>)}</select></div><span className="video-source-or">or</span><label className="video-file-label">Open your own video<input type="file" accept="video/*,.mp4,.webm,.mov,.m4v" onChange={upload} disabled={busy} /></label></section>
    {error ? <p className="video-error" role="alert">{error}</p> : null}
    {notice ? <p className="video-notice" role="status">{notice}</p> : null}
    {!source ? <section className="video-empty"><h2>Start with a real intersection</h2><p>Select the included reusable sample, or open a traffic video from your computer. Monitoring uses the pretrained detector; action recommendations require your reviewed examples and a trained street decision model.</p><ol><li>Watch the footage and inspect object boxes.</li><li>Collect spaced pictures with their recording and time.</li><li>Review each picture, then train and evaluate across separate recordings.</li></ol></section> : <>
      <div className="video-workspace"><section className="video-viewer" aria-label="Real traffic video"><div className="video-viewer-heading"><h2>{source.title}</h2><span>{preview ? `Frozen frame · ${formatVideoTime(preview.timestamp)}` : 'Original footage'}</span></div>
        <div className="video-surface">
          <video ref={videoRef} key={source.url} src={source.url} controls={!busy} muted playsInline preload="auto" hidden={Boolean(preview)} onLoadedMetadata={event => { if (sourceRef.current?.url !== source.url) return; const media = event.currentTarget; setDuration(Number.isFinite(media.duration) ? media.duration : 0); setPosition(media.currentTime) }} onLoadedData={event => { if (sourceRef.current?.url !== source.url) return; const media = event.currentTarget; setReady(media.videoWidth > 0 && Number.isFinite(media.duration) && media.duration > 0) }} onDurationChange={event => { const value = event.currentTarget.duration; if (Number.isFinite(value) && value > 0) { setDuration(value); if (event.currentTarget.readyState >= 2) setReady(true) } }} onTimeUpdate={event => setPosition(event.currentTarget.currentTime)} onError={() => { setReady(false); stop(); setError('This video could not be played. Use a supported MP4 or WebM file, or check that the included sample is installed.') }} />
          {preview ? <div className="video-frozen-surface"><img key={preview.image} src={preview.image} alt={`${result ? "Analyzed" : "Captured"} traffic frame at ${formatVideoTime(preview.timestamp)}`} onLoad={() => setLoadedPreview(preview.image)} />{loadedPreview === preview.image ? <DetectionOverlay detection={result?.detection} /> : null}</div> : null}
          {!ready ? <p className="video-loading">Loading video and reading its duration…</p> : null}
        </div>
        <label className="video-scrubber">Video position <span>{formatVideoTime(position)} / {formatVideoTime(duration)}</span><input type="range" min="0" max={Math.max(0, duration - .05)} step="0.1" value={position} disabled={!ready || task === 'upload'} onChange={event => scrub(Number(event.target.value))} /></label>
        <div className="video-toolbar"><button type="button" onClick={showOriginal} disabled={!ready || busy}>Watch original video</button><button type="button" onClick={() => void run('analyze')} disabled={!ready || busy}>Analyze this frame</button><button type="button" className="primary" onClick={() => void run('observe')} disabled={!ready || busy}>Observe video</button>{busy ? <button type="button" className="video-stop" onClick={stop}>Stop</button> : null}</div>
        <div className="video-pacing"><label>Sample every <select aria-label="Video sampling interval" value={interval} disabled={busy} onChange={event => setInterval(Number(event.target.value))}><option value={1}>1 second</option><option value={2}>2 seconds</option><option value={5}>5 seconds</option><option value={10}>10 seconds</option></select></label><p className="video-note">Observation pauses at each sampled frame until inference finishes, then shows the result before advancing. It runs at processing pace, not real time. Seeking stops observation and clears the previous overlay.</p></div>
        {busy ? <p className="video-progress" role="status">{task === 'upload' ? 'Identifying this recording…' : task === 'review' ? 'Opening saved frame…' : `${task === 'collect' || task === 'capture' ? 'Saving training pictures' : 'Analyzing video'} · ${progress.completed}${progress.total ? ` / ${progress.total}` : ''} completed`}</p> : null}
      </section><ModelEvidence result={result} /></div>
      <section className="video-collection"><div><p className="video-kicker">Exact frames · your labels</p><h2>Create training pictures</h2><p>Save images from this recording, then decide which action you would recommend. Detection boxes and model suggestions are predictions, never automatic training labels.</p></div><div className="video-toolbar"><button type="button" onClick={() => void run('capture')} disabled={!ready || busy}>Capture this frame &amp; review</button><label>Pictures <select aria-label="Number of training pictures" value={collectionSize} onChange={event => setCollectionSize(Number(event.target.value))} disabled={busy}><option value={10}>10</option><option value={25}>25</option><option value={60}>60</option></select></label><button type="button" className="primary" onClick={() => void run('collect')} disabled={!ready || busy}>Collect {collectionTimes.length || collectionSize} spaced pictures</button></div><p className="video-note">Starts at the current position, every {interval} second{interval === 1 ? '' : 's'}, stopping before the video ends. Matching saved frames are reused. All pictures from this video stay in one recording group; more pictures from one clip do not count as independent validation.</p></section>
      <section className="video-saved" aria-label="Saved training pictures"><div className="video-section-heading"><div><h2>Pictures to review</h2><p>{savedFrameCount} saved in this recording · Displayed frames: {frames.filter(frame => Boolean(frame.reviewed_action)).length} light-action reviews · {frames.filter(frame => Boolean(frame.anomaly_review)).length} anomaly reviews (including uncertain)</p></div>{shownFrameCount ? <a className="video-download" href={`/api/learning/training-package?group_id=${encodeURIComponent(source.recording)}`} download>Download images + label manifest</a> : null}</div>{frames.length ? <div className="video-frame-gallery">{frames.map(frame => <button type="button" key={frame.snapshot_id} onClick={() => void openReview(frame.snapshot_id)} disabled={busy}><img src={frame.image_url} loading="lazy" alt={`Saved frame at ${formatVideoTime(frame.video?.timestamp_seconds ?? 0)}`} /><span><b>{formatVideoTime(frame.video?.timestamp_seconds ?? 0)}</b><small>Light: {frame.reviewed_action ? actionName(frame.reviewed_action) : 'unreviewed'}</small><small>Anomaly: {frame.anomaly_review?.judgement ?? 'unreviewed'}</small></span></button>)}</div> : <p>No training pictures have been saved for this recording. Use “Capture this frame &amp; review” or collect a spaced set above.</p>}{frameTotal > shownFrameCount ? <p className="video-note">Showing {shownFrameCount} of {frameTotal} frames. The download contains the complete recording’s saved set.</p> : null}</section>
      {frameTotal > 100 ? <nav className="video-gallery-pagination" aria-label="Saved frame pages"><button type="button" disabled={busy || galleryPage === 0} onClick={() => { setGalleryLoading(true); setGalleryPage(page => page - 1) }}>Previous pictures</button><span>Page {galleryPage + 1} of {Math.ceil(frameTotal / 100)}{galleryLoading ? ' · Loading…' : ''}</span><button type="button" disabled={busy || (galleryPage + 1) * 100 >= frameTotal} onClick={() => { setGalleryLoading(true); setGalleryPage(page => page + 1) }}>Next pictures</button></nav> : null}
      <section className="video-timeline"><h2>Observation timeline</h2><p className="video-note">The most recent 100 observations in this visit. This records visible model outputs and reasons supplied by the service.</p>{history.length ? <div className="video-table-wrap"><table><thead><tr><th>Video time</th><th>Detected objects</th><th>Anomaly estimate</th><th>Learned action</th><th>Evidence</th></tr></thead><tbody>{history.map(item => <tr key={item.key}><td>{formatVideoTime(item.timestamp)}</td><td>{item.result.detection?.available ? Object.entries(item.result.detection.counts ?? {}).filter(([, count]) => count > 0).map(([name, count]) => `${count} ${name}`).join(', ') || 'None above threshold' : 'Detector unavailable'}</td><td>{typeof item.result.anomaly?.anomaly_probability === 'number' ? `${percent(item.result.anomaly.anomaly_probability)} · ${item.result.anomaly.decision === 'review' ? 'Review' : 'Not flagged'}` : 'Unavailable'}</td><td>{!item.result.policy.trained ? 'Needs reviewed examples' : item.result.policy.abstain || !item.result.policy.action ? 'Abstained' : actionName(item.result.policy.action)}</td><td>{item.result.policy.reason ?? 'No reason returned'}</td></tr>)}</tbody></table></div> : <p>Analyze a frame or observe the video to populate this timeline.</p>}</section>
      <details className="video-provenance"><summary>Recording source and training limits</summary><p>{source.description}</p>{source.attribution ? <p>{source.attribution}</p> : null}<p>License: {source.license ?? 'Not provided'}. {source.source_url ? <a href={source.source_url} target="_blank" rel="noreferrer">Original source and usage terms</a> : null}</p><p className="video-note">Recording group: <code>{source.recording}</code>. Local uploads use a fingerprint of sampled file bytes. Renamed copies keep their group; edited copies of the same recording must also stay together when evaluating a model.</p><p>The pretrained detector recognizes objects. Human-reviewed frames train the separate action model. Its performance must be evaluated using different recordings, and this footage does not provide verified optimal signal actions or establish a reliability percentage.</p></details>
    </>}
  </section>
}
