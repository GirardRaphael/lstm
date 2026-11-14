import { useEffect, useRef, useState } from 'react'
import type { VisionFrame } from './screenshotVision'
import './VisionPanel.css'

type Props = {
  frame: ImageData | null
  vision: VisionFrame | null
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function VisionPanel({ frame, vision, enabled, onEnabledChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [downloadError, setDownloadError] = useState('')
  const stamp = vision ? vision.capturedAt.toFixed(2).replace('.', '-') : 'latest'
  const available = Boolean(frame && vision && frame.width === vision.width && frame.height === vision.height && Number.isFinite(vision.capturedAt))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !frame) return
    canvas.width = frame.width
    canvas.height = frame.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.putImageData(frame, 0, 0)
    if (!vision) return
    ctx.lineWidth = 2
    ctx.font = 'bold 12px sans-serif'
    ctx.textBaseline = 'top'
    for (const detection of vision.detections) {
      const color = detection.kind === 'pedestrian' ? '#f6bd60' : detection.kind === 'truck' ? '#c2a1ff' : '#55e3d4'
      ctx.strokeStyle = color
      ctx.strokeRect(detection.x, detection.y, detection.width, detection.height)
      const label = `${detection.kind} ${detection.id}`
      const labelWidth = ctx.measureText(label).width + 6
      const labelX = Math.max(0, Math.min(frame.width - labelWidth, detection.x))
      const labelY = Math.max(0, detection.y - 15)
      ctx.fillStyle = '#081018'
      ctx.fillRect(labelX, labelY, labelWidth, 15)
      ctx.fillStyle = color
      ctx.fillText(label, labelX + 3, labelY + 1)
    }
  }, [frame, vision])

  const downloadScreenshot = () => {
    if (!frame) return
    setDownloadError('')
    // Export the untouched source image, never the preview's detection boxes.
    const source = document.createElement('canvas')
    source.width = frame.width
    source.height = frame.height
    const ctx = source.getContext('2d')
    if (!ctx) {
      setDownloadError('This browser could not prepare the screenshot.')
      return
    }
    ctx.putImageData(frame, 0, 0)
    source.toBlob((blob) => {
      if (blob) saveBlob(blob, `intersection-${stamp}.png`)
      else setDownloadError('The screenshot could not be saved. Try again.')
    }, 'image/png')
  }

  const downloadSample = () => {
    if (!frame || !vision || !available) return
    setDownloadError('')
    // Both values belong to the same atomic camera sample supplied by App.
    // Encoding synchronously keeps this download tied to that one capture.
    const source = document.createElement('canvas')
    source.width = frame.width
    source.height = frame.height
    const ctx = source.getContext('2d')
    if (!ctx) {
      setDownloadError('This browser could not prepare the sample.')
      return
    }
    ctx.putImageData(frame, 0, 0)
    try {
      const record = {
        source: 'app-canvas-screenshot',
        scene: 'simulated-intersection',
        provenance: 'predicted annotations, not verified labels',
        timeUnit: 'simulation-seconds',
        capturedAt: vision.capturedAt,
        screenshot: {
          mimeType: 'image/png', width: frame.width, height: frame.height,
          dataUrl: source.toDataURL('image/png'),
        },
        vision,
      }
      saveBlob(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }), `intersection-sample-${stamp}.json`)
    } catch {
      setDownloadError('The paired sample could not be saved. Try again.')
    }
  }

  const downloadDetections = () => {
    if (!vision) return
    setDownloadError('')
    const record = {
      source: 'app-canvas-screenshot',
      scene: 'simulated-intersection',
      provenance: 'predicted annotations, not verified labels',
      timeUnit: 'simulation-seconds',
      screenshot: `intersection-${stamp}.png`,
      ...vision,
    }
    saveBlob(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }), `intersection-${stamp}.json`)
  }

  return <section className="vision-panel" aria-label="Screenshot image recognition">
    <header className="vision-header">
      <div><p className="vision-kicker">Screenshot input</p><h2>What the camera sees</h2></div>
      <label className="vision-toggle"><input type="checkbox" checked={enabled} onChange={event => onEnabledChange(event.target.checked)} />Camera on</label>
    </header>
    <p>Snapshots of the intersection supply the traffic estimates used by the lights. The boxes below show the recognizer’s results.</p>
    <p className={`vision-status ${enabled && vision?.valid ? 'vision-status-ready' : ''}`} role="status">
      {!enabled ? 'Camera paused. The last captured image stays here.' : !vision ? 'Waiting for the first screenshot.' : vision.valid ? `Captured at ${vision.capturedAt.toFixed(2)}s · ${vision.width} × ${vision.height} pixels` : `Image unavailable: ${vision.reason}`}
    </p>
    <div className="vision-image">
      {frame ? <canvas ref={canvasRef} className="vision-canvas" aria-label="Captured intersection screenshot with detected vehicle and pedestrian boxes" /> : <p className="vision-empty">The first captured frame will appear here.</p>}
    </div>
    {vision?.valid ? <>
      <div className="vision-legend"><span className="vision-car">Cars</span><span className="vision-truck">Trucks</span><span className="vision-pedestrian">Pedestrians</span></div>
      <p className="vision-estimates">Last frame estimates: {vision.observation.cars} cars, {vision.observation.trucks} trucks and {vision.observation.pedestrians} waiting pedestrians.</p>
      <p className="vision-estimates">Junction: {vision.boxOccupied ? 'occupied' : 'clear'} · Crossings: {vision.crossingPedestrians ? 'occupied' : 'clear'}.</p>
    </> : null}
    <div className="vision-actions">
      <button type="button" disabled={!available} onClick={downloadSample}>Download sample</button>
      <button type="button" disabled={!available} onClick={downloadScreenshot}>Download screenshot</button>
      <button type="button" disabled={!available} onClick={downloadDetections}>Download detections</button>
    </div>
    {downloadError ? <p role="alert">{downloadError}</p> : null}
    <p className="vision-note">Download sample bundles one raw image with its matching estimated boxes, counts and tracking times. Separate screenshot and detection downloads can capture different moments while live. These are predicted annotations, not verified training labels.</p>
    <p className="vision-note">This recognizer is calibrated to the app’s graphics. Traffic is simulated; recognizing real aerial footage requires a separately trained and validated detector. Turning the camera off demonstrates the controller’s response to a missing feed.</p>
  </section>
}
