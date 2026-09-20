import { useEffect, useMemo, useRef, useState } from 'react'
import { IntersectionCanvas } from './IntersectionCanvas'
import { SlideView } from './SlideView'
import { SLIDES } from './slides'
import {
  createInitialState,
  createTrafficJam,
  phaseLabel,
  spawnAgents,
  stepSimulation,
  type Approach,
  type SimState,
} from './simulation'
import './App.css'

type Mode = 'slides' | 'live'

export default function App() {
  const [mode, setMode] = useState<Mode>('slides')
  const [slideIndex, setSlideIndex] = useState(0)
  const [cars, setCars] = useState(6)
  const [trucks, setTrucks] = useState(2)
  const [pedestrians, setPedestrians] = useState(4)
  const [jamApproach, setJamApproach] = useState<Approach>('east')
  const [adaptive, setAdaptive] = useState(true)
  const [running, setRunning] = useState(true)
  const [state, setState] = useState<SimState>(() => createInitialState(true))
  const lastTs = useRef<number | null>(null)

  useEffect(() => {
    setState((prev) => ({ ...prev, adaptive }))
  }, [adaptive])

  useEffect(() => {
    if (mode !== 'live' || !running) {
      lastTs.current = null
      return
    }
    let frame = 0
    const loop = (ts: number) => {
      if (lastTs.current == null) lastTs.current = ts
      const dt = Math.min(0.05, (ts - lastTs.current) / 1000)
      lastTs.current = ts
      setState((prev) => stepSimulation(prev, dt * 1.35))
      frame = requestAnimationFrame(loop)
    }
    frame = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(frame)
  }, [mode, running])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (mode === 'slides') {
        if (e.key === 'ArrowRight' || e.key === ' ') {
          e.preventDefault()
          setSlideIndex((i) => Math.min(SLIDES.length - 1, i + 1))
        }
        if (e.key === 'ArrowLeft') {
          e.preventDefault()
          setSlideIndex((i) => Math.max(0, i - 1))
        }
      }
      if (e.key.toLowerCase() === 'l') setMode('live')
      if (e.key.toLowerCase() === 's') setMode('slides')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode])

  const queueTotal = useMemo(
    () => state.queues.north + state.queues.south + state.queues.east + state.queues.west,
    [state.queues],
  )

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">Traffic LSTM</span>
          <span className="brand-sub">Observatory demo</span>
        </div>
        <nav className="mode-switch" aria-label="Presentation mode">
          <button
            type="button"
            className={mode === 'slides' ? 'active' : ''}
            onClick={() => setMode('slides')}
          >
            Slideshow
          </button>
          <button
            type="button"
            className={mode === 'live' ? 'active' : ''}
            onClick={() => setMode('live')}
          >
            Live intersection
          </button>
        </nav>
        <div className="top-hint">← → navigate · L live · S slides</div>
      </header>

      {mode === 'slides' ? (
        <main className="slides-stage">
          <SlideView slide={SLIDES[slideIndex]} index={slideIndex} total={SLIDES.length} />
          <div className="slide-controls">
            <button
              type="button"
              onClick={() => setSlideIndex((i) => Math.max(0, i - 1))}
              disabled={slideIndex === 0}
            >
              Previous
            </button>
            <button
              type="button"
              onClick={() => setSlideIndex((i) => Math.min(SLIDES.length - 1, i + 1))}
              disabled={slideIndex === SLIDES.length - 1}
            >
              Next
            </button>
            <button type="button" className="accent" onClick={() => setMode('live')}>
              Open live intersection
            </button>
          </div>
        </main>
      ) : (
        <main className="live-stage">
          <section className="live-visual">
            <IntersectionCanvas state={state} />
            <p className="sim-disclaimer">
              Simulated 2D intersection — adaptive greens respond to queue pressure. Not connected to
              real signals or city sensors.
            </p>
          </section>

          <aside className="live-panel">
            <h2>Intersection controls</h2>
            <p className="panel-lead">
              Spawn mixed traffic, force a jam, and watch phase timing stretch toward the congested
              approach when adaptive mode is on.
            </p>

            <div className="stat-row">
              <div>
                <span className="stat-label">Phase</span>
                <strong>{phaseLabel(state.phase)}</strong>
              </div>
              <div>
                <span className="stat-label">Timer</span>
                <strong>{state.phaseTimer.toFixed(1)}s</strong>
              </div>
              <div>
                <span className="stat-label">Queue (PCU)</span>
                <strong>{queueTotal}</strong>
              </div>
              <div>
                <span className="stat-label">Throughput</span>
                <strong>{state.throughput}</strong>
              </div>
            </div>

            <label className="field">
              <span>Cars</span>
              <input
                type="range"
                min={0}
                max={30}
                value={cars}
                onChange={(e) => setCars(Number(e.target.value))}
              />
              <em>{cars}</em>
            </label>
            <label className="field">
              <span>Trucks</span>
              <input
                type="range"
                min={0}
                max={15}
                value={trucks}
                onChange={(e) => setTrucks(Number(e.target.value))}
              />
              <em>{trucks}</em>
            </label>
            <label className="field">
              <span>Pedestrians</span>
              <input
                type="range"
                min={0}
                max={20}
                value={pedestrians}
                onChange={(e) => setPedestrians(Number(e.target.value))}
              />
              <em>{pedestrians}</em>
            </label>

            <label className="field">
              <span>Jam approach</span>
              <select
                value={jamApproach}
                onChange={(e) => setJamApproach(e.target.value as Approach)}
              >
                <option value="east">East</option>
                <option value="west">West</option>
                <option value="north">North</option>
                <option value="south">South</option>
              </select>
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={adaptive}
                onChange={(e) => setAdaptive(e.target.checked)}
              />
              Adaptive green (queue-aware)
            </label>

            <div className="action-row">
              <button
                type="button"
                className="accent"
                onClick={() =>
                  setState((prev) => spawnAgents(prev, { cars, trucks, pedestrians }))
                }
              >
                Spawn traffic
              </button>
              <button
                type="button"
                className="warn"
                onClick={() => setState((prev) => createTrafficJam(prev, jamApproach))}
              >
                Create traffic jam
              </button>
              <button type="button" onClick={() => setRunning((v) => !v)}>
                {running ? 'Pause' : 'Resume'}
              </button>
              <button
                type="button"
                onClick={() => setState(createInitialState(adaptive))}
              >
                Reset
              </button>
            </div>

            <div className="queue-breakdown">
              {(['north', 'south', 'east', 'west'] as Approach[]).map((a) => (
                <div key={a}>
                  <span>{a}</span>
                  <div className="bar">
                    <i style={{ width: `${Math.min(100, state.queues[a] * 8)}%` }} />
                  </div>
                  <strong>{state.queues[a]}</strong>
                </div>
              ))}
            </div>

            <p className="forecast-note">
              In the full stack, hourly LSTM / XGBoost forecasts feed demand estimates that a
              controller could consume. Here, queue pressure stands in for that demand signal so you
              can see the response live.
            </p>
          </aside>
        </main>
      )}
    </div>
  )
}
