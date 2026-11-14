import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import './TrainingLab.css'

type Action = 'ns' | 'ew' | 'walk' | 'hold'
type Domain = 'street' | 'simulation'
type Source = 'dataset' | 'upload' | 'simulation'
type Metrics = Record<string, unknown>
type Detection = {
  available: boolean; model?: string; model_version?: string; pretrained_on?: string; telemetry_notice?: string
  objects?: { class_id: number; class_name: string; score: number; box: [number, number, number, number] }[]
  counts?: Record<string, number>; score_threshold?: number; scores_calibrated?: boolean
  latency_ms?: number; image_size?: [number, number]; target?: number; target_met?: boolean
  evaluation?: { precision: number; recall: number; f1: number; images: number; iou_threshold: number; scope: string } | null
  reason?: string
}
type PolicyStatus = {
  trained: boolean; count: number; group_count: number; label_counts: Partial<Record<Action, number>>
  requirements?: { min_samples: number; min_groups: number; min_per_action: number }
  reason?: string; model_version?: string | null; metrics?: Metrics | null; calibrated?: boolean; evaluation_reused?: boolean
  reliability_target?: number; confidence_threshold?: number; target_met?: boolean | null; target_reason?: string; interval_limitation?: string
}
type Status = {
  ok: boolean
  detection?: Detection
  vision: { available: boolean; model_version?: string; metrics?: Metrics | null; reason?: string; calibrated?: boolean; deployment_ready?: boolean; validation_passed?: boolean; adequate?: boolean; evaluation_reused?: boolean; validation_result?: unknown }
  policy: Partial<Record<Domain, PolicyStatus>>
  dataset: { available: boolean; total: number; counts?: Record<string, number> }
}
type Sample = { sample_id: string; source_video: string; split: string; label: number | string; class_name: string; image_url: string }
type Samples = { ok: boolean; samples: Sample[]; total: number }
type Analysis = {
  ok: boolean; snapshot_id: string; domain: Domain; group_id: string; decision_teaching_allowed?: boolean; image_url?: string; detection?: Detection
  anomaly?: { anomaly_probability?: number | null; calibrated?: boolean; decision?: string; model_version?: string; reason?: string; [key: string]: unknown }
  policy: { trained?: boolean; action?: Action | null; suggested_action?: Action | null; probabilities?: Partial<Record<Action, number>> | null; confidence?: number | null; calibrated?: boolean; abstain?: boolean; reason?: string; model_version?: string | null; confidence_threshold?: number; reliability_target?: number; target_met?: boolean | null; target_reason?: string }
}
type Candidate = { preview: string; title: string; domain: Domain; groupId: string; sampleId?: string; dataUrl?: string }
type Props = { frame: ImageData | null; simulationSessionId?: string; onApply?: (action: Action) => void }
const ACTIONS: { value: Action; label: string; detail: string }[] = [
  { value: 'ns', label: 'North / south', detail: 'Serve the vertical road' },
  { value: 'ew', label: 'East / west', detail: 'Serve the horizontal road' },
  { value: 'walk', label: 'Pedestrian walk', detail: 'Stop vehicles for pedestrians' },
  { value: 'hold', label: 'Hold all red', detail: 'Wait, unclear or occupied' },
]
const actionName = (action: Action | null | undefined) => ACTIONS.find(item => item.value === action)?.label ?? 'No recommendation'
const percent = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Not available'
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'The request could not be completed. Try again.'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/learning/${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } })
  let body: Record<string, unknown>
  try { body = await response.json() as Record<string, unknown> } catch { throw new Error('The learning service is not available yet. Start the app service, then refresh this panel.') }
  if (!response.ok || body.ok === false) {
    const message = [body.reason, body.error, body.detail, body.message].find(value => typeof value === 'string')
    throw new Error(typeof message === 'string' ? message : response.status === 503 ? 'The learning model is not ready yet. Your image has not been used for training.' : `The request could not be completed (${response.status}).`)
  }
  return body as T
}

const METRIC_NAMES: { key: string; label: string; ratio?: boolean }[] = [
  { key: 'n', label: 'Evaluation examples' },
  { key: 'accuracy', label: 'Correct classifications', ratio: true },
  { key: 'balanced_accuracy', label: 'Balanced accuracy', ratio: true },
  { key: 'precision', label: 'Anomaly precision', ratio: true },
  { key: 'recall', label: 'Anomaly recall', ratio: true },
  { key: 'f1', label: 'Anomaly F1', ratio: true },
  { key: 'macro_f1', label: 'Macro F1', ratio: true },
  { key: 'brier', label: 'Probability error · lower is better' },
  { key: 'brier_score', label: 'Probability error · lower is better' },
  { key: 'review_coverage', label: 'Frames receiving a recommendation', ratio: true },
  { key: 'review_accuracy', label: 'Agreement on recommended frames', ratio: true },
  { key: 'ece_10_bins', label: 'Calibration gap · lower is better' },
]
function MetricEvidence({ metrics, label }: { metrics?: Metrics | null; label: string }) {
  if (!metrics || Object.keys(metrics).length === 0) return <p className="lab-note">No held-out evaluation has been recorded for this model yet.</p>
  const rows = METRIC_NAMES.filter(item => typeof metrics[item.key] === 'number')
  return <details className="lab-metrics"><summary>{label}</summary><p className="lab-note">Recorded by the model service. These results apply to the stated evaluation set, not every future road or recording. Fitting a calibration method does not guarantee that its probabilities are reliable.</p>{rows.length ? <dl className="lab-metric-grid">{rows.map(item => <div key={item.key}><dt>{item.label}</dt><dd>{item.ratio ? percent(metrics[item.key] as number) : item.key === 'n' ? String(metrics[item.key]) : (metrics[item.key] as number).toFixed(3)}</dd></div>)}</dl> : null}<details><summary>Complete recorded evaluation</summary><pre>{JSON.stringify(metrics, null, 2)}</pre></details></details>
}

function ReliabilityEvidence({ policy }: { policy?: PolicyStatus }) {
  const metrics = policy?.metrics
  const interval = metrics?.review_accuracy_interval as { lower?: number; upper?: number; n?: number; correct?: number } | undefined
  const accuracy = typeof metrics?.review_accuracy === 'number' ? metrics.review_accuracy : null
  const target = policy?.reliability_target ?? .9
  const cutoff = policy?.confidence_threshold ?? .9
  return <div className="lab-reliability"><h3>{percent(target)} reliability target</h3><p>A model score of at least {percent(cutoff)} is required before suggesting a light action. That score is separate from measured agreement with human labels on held-out recordings.</p><strong>{policy?.target_met === true ? 'Held-out agreement target met for the recorded experiment' : 'Reliability target not yet demonstrated'}</strong><p>{accuracy === null ? 'There is no held-out agreement result to support a reliability claim yet.' : `Measured agreement on recommended held-out frames: ${percent(accuracy)}.`}</p>{interval && typeof interval.lower === 'number' && typeof interval.upper === 'number' ? <p>95% agreement interval: {percent(interval.lower)}–{percent(interval.upper)} · {String(metrics?.review_count ?? interval.n ?? 0)} reviewed predictions across {String(metrics?.review_group_count ?? 0)} recording groups.</p> : null}{typeof metrics?.target_reason === 'string' ? <p className="lab-note">{metrics.target_reason}</p> : null}<p className="lab-note">To report the reliability target as met, the interval’s lower bound must reach {percent(target)}, with at least 40 recommended test frames from 3 recording groups and at least 5 examples of each action. Frames from the same recording remain correlated, so this interval cannot guarantee future performance. Agreement with human preferences does not establish road safety or optimal signal timing.</p></div>
}

function AnomalyConfusion({ metrics }: { metrics?: Metrics | null }) {
  const matrix = metrics?.confusion_matrix
  if (!Array.isArray(matrix) || matrix.length !== 2 || !matrix.every(row => Array.isArray(row) && row.length === 2 && row.every(value => typeof value === 'number' && Number.isFinite(value)))) return null
  const values = matrix as number[][]
  return <div className="lab-confusion"><h3>What the anomaly model actually got right and wrong</h3><dl className="lab-metric-grid"><div><dt>Correctly recognized normal</dt><dd>{values[0][0]}</dd></div><div><dt>False alerts on normal frames</dt><dd>{values[0][1]}</dd></div><div><dt>Missed anomalies</dt><dd>{values[1][0]}</dd></div><div><dt>Detected anomalies</dt><dd>{values[1][1]}</dd></div></dl></div>
}

function FrozenPreview({ candidate, analysis }: { candidate: Candidate; analysis: Analysis | null }) {
  const [loadedUrl, setLoadedUrl] = useState('')
  const url = analysis?.image_url ?? candidate.preview
  const detection = analysis?.domain === 'street' && analysis.image_url && analysis.detection?.available ? analysis.detection : null
  const dimensions = detection?.image_size
  const width = dimensions?.[0] ?? 1
  const height = dimensions?.[1] ?? 1
  const validDimensions = Boolean(dimensions && width > 0 && height > 0)
  const drawBoxes = Boolean(detection && validDimensions && loadedUrl === url)
  const fontSize = Math.max(12, width / 55)
  return <div className={`lab-image-stage ${validDimensions ? 'with-detections' : ''}`} style={validDimensions ? { maxWidth: 560 * width / height } : undefined}>
    <img key={url} className="lab-frozen-image" src={url} alt={candidate.title} onLoad={() => setLoadedUrl(url)} />
    {drawBoxes ? <svg className="lab-detection-overlay" viewBox={`0 0 ${width} ${height}`} aria-label="Predicted object boxes on the analyzed image" role="img"><title>Object recognition estimates; labels show raw model scores</title>{detection?.objects?.map((object, index) => {
      const [left, top, boxWidth, boxHeight] = object.box
      if (![left, top, boxWidth, boxHeight, object.score].every(Number.isFinite) || boxWidth <= 0 || boxHeight <= 0) return null
      const x = Math.max(0, Math.min(1, left)) * width
      const y = Math.max(0, Math.min(1, top)) * height
      const color = object.class_name === 'person' ? '#f6bd60' : '#55e3d4'
      return <g key={`${object.class_id}-${index}`}><rect x={x} y={y} width={Math.max(0, Math.min(boxWidth * width, width - x))} height={Math.max(0, Math.min(boxHeight * height, height - y))} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /><text x={Math.min(x + 3, Math.max(0, width - fontSize * 10))} y={Math.max(fontSize, y - 5)} fill={color} stroke="#071016" strokeWidth={fontSize * .25} paintOrder="stroke" fontSize={fontSize} fontWeight="700">{object.class_name} · score {percent(object.score)}</text></g>
    })}</svg> : null}
  </div>
}

function DetectionEvidence({ detection, compact = false }: { detection?: Detection; compact?: boolean }) {
  if (!detection) return <div className="lab-result"><h3>Object recognition</h3><p className="lab-note">No object detection result is available for this image.</p></div>
  if (!detection.available) return <div className="lab-result"><h3>Object recognition</h3><p>{detection.reason ?? 'The pretrained object detector is unavailable.'}</p></div>
  const counts = Object.entries(detection.counts ?? {}).filter(([, count]) => count > 0)
  return <div className="lab-result lab-detector"><h3>Learned object recognition</h3><strong>{detection.model ?? 'EfficientDet-Lite2'}</strong><p className="lab-note">Pretrained on {detection.pretrained_on ?? 'COCO'} · {detection.model_version ?? 'version supplied by the model service'}</p>{!compact ? <>{counts.length ? <div className="lab-object-counts">{counts.map(([name, count]) => <span key={name}><b>{count}</b> {name}{count === 1 ? '' : 's'}</span>)}</div> : <p>No objects passed the model’s score threshold. This does not prove that the image contains no traffic.</p>}<p className="lab-note">Boxes show estimated locations in this frozen image. Display threshold: raw score {percent(detection.score_threshold)}{typeof detection.latency_ms === 'number' ? ` · inference ${Math.round(detection.latency_ms)} ms` : ''}.</p></> : null}<p className="lab-notice"><b>Not validated on this intersection.</b> Box scores are uncalibrated model outputs, not probabilities of correctness or measured reliability. Object recognition alone does not determine traffic violations or which light should change.</p>{detection.evaluation ? <><dl className="lab-metric-grid"><div><dt>Measured detection precision</dt><dd>{percent(detection.evaluation.precision)}</dd></div><div><dt>Measured detection recall</dt><dd>{percent(detection.evaluation.recall)}</dd></div><div><dt>Detection F1</dt><dd>{percent(detection.evaluation.f1)}</dd></div><div><dt>Evaluation images</dt><dd>{detection.evaluation.images}</dd></div></dl><p className="lab-note">Evaluation scope: {detection.evaluation.scope}. Box matching uses an overlap threshold of {detection.evaluation.iou_threshold}.</p></> : <p className="lab-note">No labeled evaluation of this detector on your intersection has been recorded.</p>}<p className="lab-note">Inference runs locally. MediaPipe may report SDK usage metrics; Google states input images are not sent.</p><p className="lab-note">{percent(detection.target ?? .9)} object-recognition reliability target: {detection.target_met ? 'met only within the recorded evaluation scope' : 'not demonstrated'}.</p></div>
}

function imageFromData(frame: ImageData): string {
  const canvas = document.createElement('canvas')
  canvas.width = frame.width; canvas.height = frame.height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('This browser could not capture the image.')
  context.putImageData(frame, 0, 0)
  return canvas.toDataURL('image/png')
}

function collectionSessionId(): string {
  const key = 'traffic-learning-collection-v1'
  try {
    const existing = window.sessionStorage.getItem(key)
    if (existing && /^collection-[a-f0-9-]{36}$/.test(existing)) return existing
    const created = `collection-${crypto.randomUUID()}`
    window.sessionStorage.setItem(key, created)
    return created
  } catch {
    // Keep all unnamed uploads together even when the browser blocks storage.
    // An explicit recording name remains the way to distinguish recordings.
    return 'collection-browser-storage-unavailable'
  }
}

async function videoFingerprint(file: File): Promise<string> {
  // Sample stable file bytes rather than its name or modification time, so renamed
  // copies of a recording cannot silently become independent evaluation groups.
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

function captureMedia(source: HTMLImageElement | HTMLVideoElement, width: number, height: number): string {
  if (!width || !height) throw new Error('Wait until the image or video frame has loaded.')
  const scale = Math.min(1, 1600 / Math.max(width, height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(width * scale)); canvas.height = Math.max(1, Math.round(height * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('This browser could not prepare the image.')
  context.drawImage(source, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/jpeg', 0.93)
}

export function TrainingLab({ frame, simulationSessionId, onApply }: Props) {
  const [source, setSource] = useState<Source>('dataset')
  const [status, setStatus] = useState<Status | null>(null)
  const [samples, setSamples] = useState<Samples | null>(null)
  const [offset, setOffset] = useState(0)
  const [candidate, setCandidate] = useState<Candidate | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [teachingAction, setTeachingAction] = useState<Action | ''>('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [video, setVideo] = useState<{ url: string; name: string; group: string } | null>(null)
  const [videoReady, setVideoReady] = useState(false)
  const [recordingName, setRecordingName] = useState('')
  const [collectionId] = useState(collectionSessionId)
  const videoRef = useRef<HTMLVideoElement>(null)
  const objectUrl = useRef<string | null>(null)
  const domain: Domain = source === 'simulation' ? 'simulation' : 'street'
  const policyStatus = status?.policy[domain]
  const missedAllAnomalies = status?.vision.metrics?.recall === 0 && typeof status.vision.metrics.positives === 'number' && status.vision.metrics.positives > 0
  const evaluationCount = typeof status?.vision.metrics?.n === 'number' ? status.vision.metrics.n : 0
  const anomalyCount = typeof status?.vision.metrics?.positives === 'number' ? status.vision.metrics.positives : 0
  const alwaysNormalAccuracy = evaluationCount > 0 ? (evaluationCount - anomalyCount) / evaluationCount : null
  const teachingAllowed = Boolean(analysis && analysis.decision_teaching_allowed !== false && !candidate?.sampleId)
  const requirements = policyStatus?.requirements ?? { min_samples: 80, min_groups: 12, min_per_action: 8 }
  const enoughLabels = Boolean(policyStatus && policyStatus.count >= requirements.min_samples && policyStatus.group_count >= requirements.min_groups && ACTIONS.every(item => (policyStatus.label_counts[item.value] ?? 0) >= requirements.min_per_action))

  useEffect(() => {
    const controller = new AbortController()
    void request<Status>('status', { signal: controller.signal }).then(setStatus).catch(error => { if (!controller.signal.aborted) setError(errorMessage(error)) })
    void request<Samples>('samples?split=train&offset=0&limit=12', { signal: controller.signal }).then(setSamples).catch(error => { if (!controller.signal.aborted) setError(errorMessage(error)) })
    return () => controller.abort()
  }, [])
  useEffect(() => () => { if (objectUrl.current) URL.revokeObjectURL(objectUrl.current) }, [])

  const refresh = async () => {
    setBusy('refresh'); setError('')
    try { setStatus(await request<Status>('status')) } catch (error) { setError(errorMessage(error)) } finally { setBusy('') }
  }
  const changeSource = (next: Source) => { setSource(next); setCandidate(null); setAnalysis(null); setTeachingAction(''); setError(''); setNotice('') }
  const chooseCandidate = (next: Candidate) => { setCandidate(next); setAnalysis(null); setTeachingAction(''); setNotice(''); setError('') }
  const loadPage = async (next: number) => {
    setBusy('samples'); setError('')
    try { const page = await request<Samples>(`samples?split=train&offset=${next}&limit=12`); setSamples(page); setOffset(next) } catch (error) { setError(errorMessage(error)) } finally { setBusy('') }
  }
  const captureSimulation = () => {
    if (!frame) return
    try { const dataUrl = imageFromData(frame); chooseCandidate({ preview: dataUrl, dataUrl, domain: 'simulation', groupId: simulationSessionId ?? collectionId, title: 'Frozen app screenshot' }) } catch (error) { setError(errorMessage(error)) }
  }
  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy('upload'); setError(''); setNotice(''); setAnalysis(null); setCandidate(null); setVideoReady(false)
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = URL.createObjectURL(file)
    const url = objectUrl.current
    try {
      if (file.type.startsWith('video/')) {
        const fingerprint = await videoFingerprint(file)
        setVideo({ url, name: file.name, group: `video-${fingerprint}` })
      } else if (file.type.startsWith('image/')) {
        setVideo(null)
        const image = new Image(); image.src = url
        await image.decode()
        const dataUrl = captureMedia(image, image.naturalWidth, image.naturalHeight)
        chooseCandidate({ preview: dataUrl, dataUrl, domain: 'street', groupId: recordingName.trim() || collectionId, title: file.name })
      } else throw new Error('Choose an image or a video supported by your browser.')
    } catch (error) { setError(errorMessage(error)) } finally { setBusy(''); event.target.value = '' }
  }
  const captureVideo = () => {
    const element = videoRef.current
    if (!element || !video) return
    try {
      element.pause()
      const dataUrl = captureMedia(element, element.videoWidth, element.videoHeight)
      chooseCandidate({ preview: dataUrl, dataUrl, domain: 'street', groupId: video.group, title: `${video.name} · ${element.currentTime.toFixed(2)} seconds` })
    } catch (error) { setError(errorMessage(error)) }
  }
  const analyze = async () => {
    if (!candidate) return
    setBusy('analyze'); setError(''); setNotice(''); setAnalysis(null); setTeachingAction('')
    const payload = candidate.sampleId ? { sample_id: candidate.sampleId } : { image_data_url: candidate.dataUrl, domain: candidate.domain, group_id: candidate.groupId }
    try { setAnalysis(await request<Analysis>('analyze', { method: 'POST', body: JSON.stringify(payload) })) } catch (error) { setError(errorMessage(error)) } finally { setBusy('') }
  }
  const teach = async () => {
    if (!analysis || !teachingAction || !teachingAllowed) return
    setBusy('feedback'); setError(''); setNotice('')
    try {
      await request('feedback', { method: 'POST', body: JSON.stringify({ snapshot_id: analysis.snapshot_id, action: teachingAction }) })
      setNotice(`Saved your preference: ${actionName(teachingAction)}. This is a human teaching label, not proof of an optimal signal decision.`)
      setStatus(await request<Status>('status'))
    } catch (error) { setError(errorMessage(error)) } finally { setBusy('') }
  }
  const train = async () => {
    setBusy('train'); setError(''); setNotice('')
    try {
      await request('train', { method: 'POST', body: JSON.stringify({ domain }) })
      setStatus(await request<Status>('status'))
      setAnalysis(null)
      setNotice('Training finished. Review the held-out decision results below, then analyze an image to use the new model.')
    } catch (error) { setError(errorMessage(error)); try { setStatus(await request<Status>('status')) } catch { /* Keep the actionable training error. */ } } finally { setBusy('') }
  }

  return <section className="training-lab" aria-label="Image recognition and decision teaching">
    <header className="lab-header"><div><p className="lab-kicker">Observe · review · teach</p><h1>Traffic learning lab</h1><p>Inspect a real image, see the model’s uncertainty, and teach a separate light-decision model with your own choices.</p></div><button type="button" className="lab-button" onClick={() => void refresh()} disabled={Boolean(busy)}>Refresh model status</button></header>
    <div className="lab-status-grid">
      <article><span className="lab-eyebrow">Street anomaly recognition</span><strong>{status ? status.vision.available ? missedAllAnomalies ? 'Baseline inadequate' : 'Research model available' : 'Model not ready' : 'Connecting…'}</strong><p>{status?.vision.model_version ?? status?.vision.reason ?? 'Recognizes patterns associated with normal or anomalous traffic. It does not choose the lights.'}</p></article>
      <article><span className="lab-eyebrow">Learned light decisions · {domain === 'street' ? 'street images' : 'app screenshots'}</span><strong>{policyStatus?.trained ? 'Trained on human choices' : 'Not trained yet'}</strong><p>{policyStatus?.reason ?? 'Needs your labeled examples before it can make learned recommendations.'}</p></article>
    </div>
    {missedAllAnomalies ? <p className="lab-error" role="status"><b>The current baseline missed all {String(status?.vision.metrics?.positives)} held-out anomalies.</b> Its {percent(status?.vision.metrics?.accuracy as number)} overall accuracy comes from predicting normal; anomaly recall is 0%. These probabilities are for inspection, not reliable anomaly warnings or signal control.</p> : status?.vision.deployment_ready === false ? <p className="lab-note">The street recognition model is experimental and has not passed deployment validation. Review its recorded error rates before interpreting a confidence value.</p> : null}
    <div className="lab-source-tabs" aria-label="Image source">
      {([{ key: 'dataset', label: 'Real street examples' }, { key: 'upload', label: 'Your image or video' }, { key: 'simulation', label: 'App screenshot' }] as const).map(item => <button type="button" key={item.key} aria-pressed={source === item.key} className={source === item.key ? 'selected' : ''} onClick={() => changeSource(item.key)} disabled={Boolean(busy)}>{item.label}</button>)}
    </div>
    {source === 'dataset' ? <div className="lab-source-panel">
      <p>Choose a real aerial frame. This gallery uses the training partition only; evaluation recordings stay separate. These roundabouts provide binary anomaly labels. Upload a signalized intersection recording to teach light choices.</p>
      {samples?.samples.length ? <div className="lab-samples">{samples.samples.map(sample => <button type="button" key={sample.sample_id} className={candidate?.sampleId === sample.sample_id ? 'selected' : ''} disabled={Boolean(busy)} onClick={() => chooseCandidate({ preview: sample.image_url, sampleId: sample.sample_id, domain: 'street', groupId: sample.source_video, title: `Recording ${sample.source_video} · ${sample.sample_id}` })}><img src={sample.image_url} alt={`Aerial traffic frame ${sample.sample_id}`} loading="lazy" /><span>{sample.class_name || (Number(sample.label) ? 'Anomaly' : 'Normal')} · publisher label</span></button>)}</div> : <p className="lab-empty">{samples ? 'No local training images are available. Upload an image or video to continue.' : 'Loading available street images…'}</p>}
      <div className="lab-pagination"><button type="button" className="lab-button" disabled={Boolean(busy) || offset === 0} onClick={() => void loadPage(Math.max(0, offset - 12))}>Previous</button><span>{samples?.total ? `${offset + 1}–${Math.min(offset + 12, samples.total)} of ${samples.total} training frames` : 'No frames loaded'}</span><button type="button" className="lab-button" disabled={Boolean(busy) || !samples || offset + 12 >= samples.total} onClick={() => void loadPage(offset + 12)}>Next</button></div>
    </div> : source === 'upload' ? <div className="lab-source-panel">
      <label className="lab-upload-label">Choose a street image or video<input type="file" accept="image/*,video/*" onChange={event => void upload(event)} disabled={Boolean(busy)} /></label>
      <p className="lab-note">Video stays in your browser. Seek to a moment, pause, then capture one frame. Only that frame is sent to the local model service. Large frames are resized to at most 1,600 pixels on their longest edge.</p>
      {video ? <div className="lab-video"><video ref={videoRef} src={video.url} controls preload="metadata" onLoadedData={() => setVideoReady(true)} onSeeking={() => setVideoReady(false)} onSeeked={() => setVideoReady(true)} onError={() => setError('This browser could not read the video. Try an MP4 video or upload a still image.')} /><button type="button" className="lab-button" onClick={captureVideo} disabled={Boolean(busy) || !videoReady}>Capture this video frame</button><p className="lab-note">All frames from this file stay in one recording group.</p></div> : <label className="lab-recording">Recording name for still images <input value={recordingName} onChange={event => { const name = event.target.value; setRecordingName(name); setCandidate(previous => previous && !previous.sampleId ? { ...previous, groupId: name.trim() || collectionId } : previous); setAnalysis(null); setTeachingAction('') }} placeholder="For example: junction-camera-1-monday" disabled={Boolean(busy)} /><span>Use the same name for every frame from the same recording. Without a name, all images in this browser session stay in one group, including after changing app tabs.</span></label>}
    </div> : <div className="lab-source-panel"><p>Freeze the current intersection screenshot. The app’s graphics are outside the street model’s training domain, so street anomaly confidence is not shown here. App and street teaching examples train separate decision models.</p><button type="button" className="lab-button primary" disabled={Boolean(busy) || !frame} onClick={captureSimulation}>Capture app screenshot</button>{!frame ? <p className="lab-note">Open the live intersection first to capture a frame, then return here.</p> : null}<p className="lab-note">Screenshots from one simulation episode share a recording group. Reset the live intersection to begin a genuinely new episode.</p></div>}
    {error ? <p className="lab-error" role="alert">{error}</p> : null}
    {notice ? <p className="lab-notice" role="status">{notice}</p> : null}
    {busy ? <p className="lab-progress" role="status">{busy === 'train' ? 'Training and evaluating the decision model. This can take a moment…' : busy === 'analyze' ? 'Analyzing the frozen image…' : busy === 'feedback' ? 'Saving your teaching example…' : 'Loading…'}</p> : null}
    <div className="lab-workspace">
      <article className="lab-preview-card"><h2>1. Inspect the image</h2>{candidate ? <><FrozenPreview candidate={candidate} analysis={analysis} /><p className="lab-caption">{candidate.title}</p><p className="lab-note">This preview stays frozen while you analyze or label it.</p><button type="button" className="lab-button primary" onClick={() => void analyze()} disabled={Boolean(busy)}>Analyze this image</button></> : <p className="lab-empty">Select a frame above to begin.</p>}</article>
      <article className="lab-results-card"><h2>2. Read the model results</h2>{analysis ? <>
        <DetectionEvidence detection={analysis.detection} /><div className="lab-result"><h3>Street anomaly estimate</h3>{analysis.domain === 'simulation' ? <p>Unavailable for simulated graphics. This model was trained on real aerial images.</p> : typeof analysis.anomaly?.anomaly_probability === 'number' ? <><strong className="lab-number">{percent(analysis.anomaly.anomaly_probability)}</strong><p>Estimated probability of the source dataset’s anomaly class.</p><span className="lab-badge">{analysis.anomaly.calibrated ? 'Probability calibration fitted on separate data' : 'Uncalibrated model score'}</span><p className="lab-note">{analysis.anomaly.decision ? `Model decision: ${analysis.anomaly.decision}. ` : ''}{analysis.anomaly.calibrated ? 'Calibration has been fitted; check its reliability in the evaluation below. ' : 'This score has not been calibrated. '}This is not the probability of a particular traffic violation, and it does not authorize a light change.</p></> : <p>{analysis.anomaly?.reason ?? 'The street model is unavailable. No confidence score has been invented.'}</p>}</div>
        <div className="lab-result"><h3>Learned light recommendation</h3><strong>{analysis.policy.action ? actionName(analysis.policy.action) : analysis.policy.abstain ? 'Model abstained' : 'No trained recommendation'}</strong><p>{analysis.policy.reason}</p>{typeof analysis.policy.confidence === 'number' ? <><p>Top-action {analysis.policy.calibrated ? 'calibrated probability' : 'uncalibrated score'}: <b>{percent(analysis.policy.confidence)}</b></p><p className="lab-note">This is a model probability, not a measured success rate. The separate 90% reliability target must be checked against held-out human choices; neither proves safety or optimal traffic flow.</p></> : null}{analysis.policy.probabilities ? <ul className="lab-probabilities">{ACTIONS.map(item => <li key={item.value}><span>{item.label}</span><b>{percent(analysis.policy.probabilities?.[item.value])}</b></li>)}</ul> : null}<p className="lab-note">{analysis.policy.target_met === true ? 'The recorded experiment met its human-label agreement target. ' : 'The 90% held-out reliability target has not been demonstrated. '}Recommendations require human review. Uploaded street images never control the simulation or real infrastructure.</p>{analysis.policy.action && !analysis.policy.abstain && teachingAllowed ? <button type="button" className="lab-button" disabled={Boolean(busy)} onClick={() => { if (analysis.policy.action) setTeachingAction(analysis.policy.action) }}>Use suggested action</button> : null}</div>
      </> : <p className="lab-empty">Analyze an image to see its actual model outputs. No prediction is generated from a label you have not supplied.</p>}</article>
    </div>
    <section className="lab-teaching"><h2>3. Teach the light-decision model</h2><p>For the exact image above, choose the action you would prefer. For street examples, north/south means the vertical road and east/west the horizontal road in the image. If the road layout or signal state cannot support a choice, use “Hold all red” or skip the frame.</p>{candidate?.sampleId ? <p className="lab-notice">These roundabout images have no verified signal layout. They can be used to inspect anomaly recognition, but cannot supply correct north/south or east/west light labels. Upload a signalized intersection recording to teach light choices.</p> : null}<fieldset disabled={!teachingAllowed || Boolean(busy)}><legend>Your preferred action</legend><div className="lab-action-options">{ACTIONS.map(item => <label key={item.value} className={teachingAction === item.value ? 'selected' : ''}><input type="radio" name="teaching-action" value={item.value} checked={teachingAction === item.value} onChange={() => setTeachingAction(item.value)} /><span><b>{item.label}</b><small>{item.detail}</small></span></label>)}</div></fieldset><div className="lab-actions"><button type="button" className="lab-button primary" disabled={!teachingAllowed || !teachingAction || Boolean(busy)} onClick={() => void teach()}>Save teaching example</button>{analysis?.domain === 'simulation' && onApply ? <button type="button" className="lab-button" disabled={!teachingAction || Boolean(busy)} onClick={() => { if (teachingAction) onApply(teachingAction) }}>Apply my reviewed choice in the simulation</button> : null}</div><p className="lab-note">Teaching labels are your preferences. The model does not create its own correct answers, and saving an example does not immediately retrain it. Missing motion, signal-state or visibility information can make a single-frame choice unreliable.</p></section>
    <section className="lab-training"><div><h2>4. Train from your examples</h2><p>{domain === 'street' ? 'Street image' : 'App screenshot'} decision model · {policyStatus?.trained ? `version ${policyStatus.model_version ?? 'available'}` : 'not trained'}</p></div><div className="lab-counts"><span><b>{policyStatus?.count ?? 0} / {requirements.min_samples}</b> human labels</span><span><b>{policyStatus?.group_count ?? 0} / {requirements.min_groups}</b> recording groups</span>{ACTIONS.map(item => <span key={item.value}><b>{policyStatus?.label_counts[item.value] ?? 0} / {requirements.min_per_action}</b> {item.label}</span>)}</div><p className="lab-note">Recordings are separated for fitting, calibration and evaluation. More frames from one recording do not replace independent recordings. These minimums make evaluation possible; they do not establish that the policy is ready for a real road.</p><ReliabilityEvidence policy={policyStatus} /><button type="button" className="lab-button primary" onClick={() => void train()} disabled={Boolean(busy) || !enoughLabels}>{policyStatus?.trained ? 'Retrain decision model' : 'Train decision model'}</button>{!enoughLabels ? <p className="lab-note">Collect at least {requirements.min_samples} human labels across {requirements.min_groups} recording groups, with {requirements.min_per_action} examples of every action, to enable training.</p> : null}{policyStatus?.evaluation_reused ? <p className="lab-note">This evaluation has been viewed during earlier training rounds. Treat repeated results as retrospective feedback, not a new untouched test.</p> : null}<MetricEvidence metrics={policyStatus?.metrics} label="View held-out light-decision evaluation" /></section>
    <section className="lab-evidence"><h2>Object detection evidence</h2><DetectionEvidence detection={status?.detection} compact /></section>
    <section className="lab-evidence"><h2>Street anomaly evidence</h2><p>Street anomaly evaluation and light-decision evaluation answer different questions. Good normal/anomaly classification does not prove that a chosen traffic light is safe or efficient. Real footage can differ from the aerial training data; high confidence can still be wrong.</p>{status?.vision.metrics ? <dl className="lab-metric-grid">{METRIC_NAMES.filter(item => ['accuracy', 'balanced_accuracy', 'precision', 'recall'].includes(item.key) && typeof status.vision.metrics?.[item.key] === 'number').map(item => <div key={item.key}><dt>{item.label}</dt><dd>{percent(status.vision.metrics?.[item.key] as number)}</dd></div>)}</dl> : null}{alwaysNormalAccuracy !== null ? <p className="lab-note"><b>Always-normal baseline: {percent(alwaysNormalAccuracy)} accuracy{anomalyCount > 0 ? ', with 0% anomaly recall' : ''}.</b> A useful anomaly model must improve on this simple baseline; overall accuracy alone can hide missed anomalies.</p> : null}{status?.vision.evaluation_reused ? <p className="lab-notice">Several candidate models have been inspected on this historical evaluation set. These results are retrospective. A new recording set must remain untouched to confirm performance.</p> : null}<AnomalyConfusion metrics={status?.vision.metrics} /><MetricEvidence metrics={status?.vision.metrics} label="View real-image anomaly evaluation" /></section>
  </section>
}
