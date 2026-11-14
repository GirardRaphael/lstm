import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { VisionPanel } from './VisionPanel'
import { TrainingLab } from './TrainingLab'
import { analyzeScreenshot, type VisionFrame } from './screenshotVision'
import { IntersectionCanvas } from './IntersectionCanvas'
import { PresentationDeck } from './PresentationDeck'
import { createClassroomScenario, type DemoScenarioId } from './classroomScenarios'
import { initialClassroomRoute } from './classroomRoute'
import { applyReviewedActionToState } from './reviewedAction'
import { ThinkingPanel } from './ThinkingPanel'
import { ControllerPanel, AppliedDecisions } from './ControllerPanel'
import { cameraReady, observeTraffic, type Axis } from './trafficController'
import {
  congestionIntensity,
  thinkDecision,
  type HourLookupFile,
  type LiveModelOutput,
} from './decisionBrain'
import { SLIDES, type Slide } from './slides'
import {
  applyVisionFrame,
  createInitialState,
  createTrafficJam,
  phaseLabel,
  pressCrosswalkButton,
  randomApproachChoice,
  randomSpawnCounts,
  spawnAgents,
  stepSimulation,
  type Approach,
  type SimState,
} from './simulation'
import './App.css'

type Mode = 'slides' | 'live' | 'learn'

export default function App() {
  const [mode, setMode] = useState<Mode>(() => initialClassroomRoute(window.location.hash).mode)
  const [slideIndex, setSlideIndex] = useState(() => initialClassroomRoute(window.location.hash).slideIndex)
  const [teachingSource, setTeachingSource] = useState<'dataset' | 'upload' | 'simulation'>('upload')
  const [guidedDemo, setGuidedDemo] = useState<NonNullable<Slide['demo']> | null>(null)
  const [cars, setCars] = useState(12)
  const [trucks, setTrucks] = useState(3)
  const [pedestrians, setPedestrians] = useState(20)
  const [jamApproach, setJamApproach] = useState<Approach>('east')
  const [running, setRunning] = useState(true)
  const [simulationSessionId, setSimulationSessionId] = useState(() => crypto.randomUUID())
  const [cameraEnabled, setCameraEnabled] = useState(true)
  const [cameraSample, setCameraSample] = useState<{ pixels: ImageData; vision: VisionFrame } | null>(null)
  const previousVision = useRef<VisionFrame | undefined>(undefined)
  const [showHistory, setShowHistory] = useState(false)
  const [lastAction, setLastAction] = useState('Ready — randomize traffic or force a jam.')
  const [lookup, setLookup] = useState<HourLookupFile | null>(null)
  const [liveModel, setLiveModel] = useState<LiveModelOutput | null>(null)
  const [modelStatus, setModelStatus] = useState('Loading Keras hour lookup…')
  const [state, setState] = useState<SimState>(() =>
    spawnAgents(createInitialState(true, 'camera'), { cars: 12, trucks: 3, pedestrians: 20 }),
  )
  const lastTs = useRef<number | null>(null)
  const adaptive = state.adaptive
  useEffect(() => {
    const hash = mode === 'slides' ? `#slides/${SLIDES[slideIndex].id}` : `#${mode}`
    window.history.replaceState(null, '', hash)
  }, [mode, slideIndex])
  const receiveScreenshot = useCallback((frame: ImageData, capturedAt: number) => {
    if (previousVision.current && capturedAt <= previousVision.current.capturedAt) return
    // This is the perception boundary: pixels, a clock and prior pixel tracks only.
    const measured = analyzeScreenshot(frame, capturedAt, previousVision.current)
    previousVision.current = measured
    setCameraSample({ pixels: frame, vision: measured })
    setState(prev => applyVisionFrame(prev, measured))
  }, [])

  useEffect(() => {
    fetch('/hour_lookup.json')
      .then((r) => {
        if (!r.ok) throw new Error(`lookup ${r.status}`)
        return r.json()
      })
      .then((data: HourLookupFile) => {
        if (!Array.isArray(data.hours) || data.hours.length === 0) {
          throw new Error('empty lookup')
        }
        setLookup(data)
        setModelStatus('Replayed Keras output for the closest real hour (static lookup loaded).')
      })
      .catch(() => {
        setLookup(null)
        setModelStatus('Historical forecast unavailable. The scene controller continues independently.')
      })
  }, [])

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
      if (e.altKey || e.ctrlKey || e.metaKey || (e.target instanceof HTMLElement && (e.target.closest('input, select, textarea, button, a, summary') || e.target.isContentEditable))) return
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
      if (mode === 'live' && e.key.toLowerCase() === 'r') {
        const counts = randomSpawnCounts()
        setCars(counts.cars)
        setTrucks(counts.trucks)
        setPedestrians(counts.pedestrians)
        setState((prev) => spawnAgents(prev, counts))
        setLastAction(
          `Random spawn: ${counts.cars} cars, ${counts.trucks} trucks, ${counts.pedestrians} pedestrians.`,
        )
      }
      if (mode === 'live' && e.key.toLowerCase() === 'j') {
        const approach = randomApproachChoice()
        setJamApproach(approach)
        setState((prev) => createTrafficJam(prev, approach))
        setLastAction(`Random jam on the ${approach} approach — watch the screenshot detections.`)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode])

  const observation = useMemo(() => observeTraffic(state), [state])
  const queueTotal = Object.values(observation.queues).reduce((sum, count) => sum + count, 0)
  const liveCounts = observation
  const intensityBucket = Math.round(congestionIntensity(state) * 40)

  useEffect(() => {
    if (mode !== 'live' || !showHistory) return
    const intensity = intensityBucket / 40
    const controller = new AbortController()
    fetch(`/api/forecast?intensity=${intensity.toFixed(3)}`, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`api ${r.status}`))))
      .then((data) => {
        if (!data?.ok || !Number.isFinite(data.predicted_vehicles)) {
          throw new Error('invalid forecast')
        }
        setLiveModel({
          index: data.index,
          timestamp: data.timestamp,
          actual_vehicles: data.actual_vehicles,
          predicted_vehicles: data.predicted_vehicles,
          gates: data.gates,
          neurons: data.neurons,
          dense: data.dense,
          top_neurons: data.top_neurons ?? [],
          source: 'sidecar',
          demand_query: data.demand_query,
          label: data.label,
          model: data.model,
        })
        setModelStatus(
          'Live sidecar: NumPy forward pass of baseline_univariate.keras on the closest real hour.',
        )
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return
        setLiveModel(null)
        if (lookup) {
          setModelStatus(
            'Sidecar offline — using static replayed Keras output for the closest real hour.',
          )
        } else {
          setModelStatus(
            `Model unavailable${err instanceof Error ? ` (${err.message})` : ''} — forecast hidden.`,
          )
        }
      })
    return () => controller.abort()
  }, [mode, intensityBucket, lookup, showHistory])

  const thought = useMemo(() => thinkDecision(state, liveModel, lookup), [state, liveModel, lookup])
  const applyReviewedAction = (action: 'ns' | 'ew' | 'walk' | 'hold') => {
    setState(prev => applyReviewedActionToState(prev, action))
    setRunning(true)
    setMode('live')
    setLastAction(action === 'hold' ? 'Reviewed hold: finish yellow, then keep red.' :
      `Reviewed ${action.toUpperCase()} request queued${action === 'ns' || action === 'ew' ? '; automatic control enabled' : ''}. Crossing clearance and waiting priorities remain active.`)
  }

  const overrideDecisionAxis = (axis: Axis) => {
    setState(prev => ({ ...prev, operatorHold: false, controllerOverride: axis }))
    setLastAction(`Requested ${axis.toUpperCase()} at the next clear junction. Waiting-time and pedestrian priority remain active.`)
  }

  const openTraining = (source: 'dataset' | 'upload' | 'simulation' = 'upload') => {
    setTeachingSource(source)
    setMode('learn')
    window.scrollTo({ top: 0 })
  }
  const startClassroomScenario = (id: DemoScenarioId) => {
    const scenario = createClassroomScenario(id)
    previousVision.current = undefined
    setCameraSample(null)
    setState(scenario.state)
    setCameraEnabled(scenario.cameraEnabled)
    setCars(scenario.counts.cars); setTrucks(scenario.counts.trucks); setPedestrians(scenario.counts.pedestrians)
    setJamApproach(scenario.jamApproach)
    setSimulationSessionId(crypto.randomUUID())
    setRunning(true); setShowHistory(false); setMode('live')
    setLastAction('Classroom scenario loaded. The camera will measure the rendered scene.')
    window.scrollTo({ top: 0 })
  }
  const launchDemo = (demo: NonNullable<Slide['demo']>) => {
    setGuidedDemo(demo)
    if (demo.target === 'learn') openTraining('upload')
    else startClassroomScenario(demo.scenario ?? 'balanced')
  }
  const startTeachingPractice = () => {
    previousVision.current = undefined; setCameraSample(null)
    const counts = randomSpawnCounts()
    setState(spawnAgents(createInitialState(true, 'camera'), counts))
    setCars(counts.cars); setTrucks(counts.trucks); setPedestrians(counts.pedestrians)
    setSimulationSessionId(crypto.randomUUID()); setCameraEnabled(true); setRunning(true); setMode('live')
    setLastAction('Teaching practice: let the scene develop, then choose Teach a decision from this screenshot.')
    window.scrollTo({ top: 0 })
  }

  const randomizeCounts = () => {
    const counts = randomSpawnCounts()
    setCars(counts.cars)
    setTrucks(counts.trucks)
    setPedestrians(counts.pedestrians)
    setLastAction(
      `Rolled random counts: ${counts.cars} cars, ${counts.trucks} trucks, ${counts.pedestrians} pedestrians.`,
    )
    return counts
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">Traffic LSTM</span>
          <span className="brand-sub">Observatory demo</span>
        </div>
        <nav className="mode-switch" aria-label="Application mode">
          <button type="button" className={mode === 'learn' ? 'active' : ''} onClick={() => openTraining()}>
            Train &amp; review
          </button>
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
        <div className="top-hint">
          {mode === 'slides' ? '← → navigate · L live' : mode === 'learn' ? 'Real images · model evidence · your corrections' : 'R spawn · J jam · screenshot detections'}
        </div>
      </header>

      {guidedDemo && mode !== 'slides' ? <aside className="classroom-return" aria-label="Guided classroom demonstration"><div><p className="visual-eyebrow">CLASSROOM DEMONSTRATION</p><strong>{guidedDemo.label}</strong><ol>{guidedDemo.steps.map(step => <li key={step}>{step}</li>)}</ol></div><div><button className="accent" onClick={() => { setMode('slides'); window.scrollTo({ top: 0 }) }}>Return to slide {slideIndex + 1}</button>{guidedDemo.scenario ? <button onClick={() => startClassroomScenario(guidedDemo.scenario!)}>Restart this scenario</button> : null}<button onClick={() => setGuidedDemo(null)}>Close demo guide</button></div></aside> : null}
      {mode === 'learn' ? (
        <main className="learning-stage">
          <TrainingLab frame={cameraSample?.pixels ?? null} simulationSessionId={simulationSessionId} initialSource={teachingSource} onStartSimulation={startTeachingPractice} onApply={applyReviewedAction} />
        </main>
      ) : mode === 'slides' ? (
        <PresentationDeck index={slideIndex} onIndex={setSlideIndex} onDemo={launchDemo} />
      ) : (
        <main className="live-stage thinking-layout">
          <IntersectionCanvas state={state} cameraEnabled={cameraEnabled} onFrame={receiveScreenshot} />
          <div className="live-visual">
            <div className={`response-banner ${state.jamActive ? 'hot' : ''}`}>
              <strong>{state.operatorHold ? 'Reviewed hold — waiting for release' : state.adaptive ? 'Automatic traffic control' : 'Fixed signal cycle'}</strong>
              <span>{phaseLabel(state.phase)} · {state.phaseTimer.toFixed(1)}s remaining</span>
            </div>
            <VisionPanel frame={cameraSample?.pixels ?? null} vision={cameraSample?.vision ?? null} enabled={cameraEnabled} onEnabledChange={setCameraEnabled} />
            <button type="button" className="accent" disabled={!cameraSample} onClick={() => openTraining('simulation')}>Teach a decision from this screenshot</button>
          </div>

          <div className="live-windows">
          <ControllerPanel state={state} onOverride={overrideDecisionAxis} />

          <aside className="live-panel">
            <h2>Intersection controls</h2>
            <p className="panel-lead">
              Create a jam or add pedestrians and watch the controller choose a direction at the next clear junction.
              Switch automatic control off to compare with a fixed cycle.
            </p>

            <p className="action-log" role="status">
              {lastAction}
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
                <span className="stat-label">Detected queue (PCU)</span>
                <strong>{cameraReady(state) ? queueTotal : '—'}</strong>
              </div>
              <div>
                <span className="stat-label">Detected traffic</span>
                <strong>
                  {liveCounts.cars}c / {liveCounts.trucks}t / {liveCounts.pedestrians}p
                </strong>
              </div>
            </div>

            <div className="legend">
              <span>
                <i className="swatch car" /> Car
              </span>
              <span>
                <i className="swatch truck" /> Truck
              </span>
              <span>
                <i className="swatch ped" /> Pedestrian
              </span>
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
              <input
                className="count-box"
                type="number"
                min={0}
                max={30}
                value={cars}
                onChange={(e) => setCars(Math.max(0, Math.min(30, Number(e.target.value) || 0)))}
              />
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
              <input
                className="count-box"
                type="number"
                min={0}
                max={15}
                value={trucks}
                onChange={(e) => setTrucks(Math.max(0, Math.min(15, Number(e.target.value) || 0)))}
              />
            </label>
            <label className="field">
              <span>Pedestrians</span>
              <input
                type="range"
                min={0}
                max={36}
                value={pedestrians}
                onChange={(e) => setPedestrians(Number(e.target.value))}
              />
              <input
                className="count-box"
                type="number"
                min={0}
                max={36}
                value={pedestrians}
                onChange={(e) =>
                  setPedestrians(Math.max(0, Math.min(36, Number(e.target.value) || 0)))
                }
              />
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
                onChange={(e) => { const enabled = e.target.checked; setState(prev => ({ ...prev, adaptive: enabled, controllerOverride: null })) }}
              />
              Automatic traffic controller (off = fixed cycle)
            </label>

            <div className="crosswalk-row">
              <span className="crosswalk-label">Crosswalk</span>
              <button
                type="button"
                className={`crosswalk-btn ${state.crosswalkRequest.ns ? 'pressed' : ''}`}
                onClick={() => {
                  setState((prev) => pressCrosswalkButton(prev, 'ns'))
                  setLastAction('Pedestrian pressed NS crosswalk — light will change.')
                }}
              >
                NS
              </button>
              <button
                type="button"
                className={`crosswalk-btn ${state.crosswalkRequest.ew ? 'pressed' : ''}`}
                onClick={() => {
                  setState((prev) => pressCrosswalkButton(prev, 'ew'))
                  setLastAction('Pedestrian pressed EW crosswalk — light will change.')
                }}
              >
                EW
              </button>
            </div>

            <div className="action-row">
              <button
                type="button"
                onClick={() => {
                  const counts = randomizeCounts()
                  setState((prev) => spawnAgents(prev, counts))
                  setLastAction(
                    `Random spawn: ${counts.cars} cars, ${counts.trucks} trucks, ${counts.pedestrians} pedestrians.`,
                  )
                }}
              >
                Randomize &amp; spawn
              </button>
              <button
                type="button"
                className="accent"
                onClick={() => {
                  setState((prev) => spawnAgents(prev, { cars, trucks, pedestrians }))
                  setLastAction(
                    `Spawned ${cars} cars, ${trucks} trucks, ${pedestrians} pedestrians.`,
                  )
                }}
              >
                Spawn traffic
              </button>
              <button
                type="button"
                className="warn"
                onClick={() => {
                  setState((prev) => createTrafficJam(prev, jamApproach))
                  setLastAction(
                    `Traffic jam on ${jamApproach} — watch the detections and the next light decision.`,
                  )
                }}
              >
                Create traffic jam
              </button>
              <button
                type="button"
                className="warn"
                onClick={() => {
                  const approach = randomApproachChoice()
                  setJamApproach(approach)
                  setState((prev) => createTrafficJam(prev, approach))
                  setLastAction(`Random jam on the ${approach} approach.`)
                }}
              >
                Random jam
              </button>
              <button type="button" onClick={() => setRunning((v) => !v)}>
                {running ? 'Pause' : 'Resume'}
              </button>
              <button
                type="button"
                onClick={() => {
                  previousVision.current = undefined
                  setCameraSample(null)
                  setState(createInitialState(adaptive, 'camera'))
                  setSimulationSessionId(crypto.randomUUID())
                  setLastAction('Intersection reset.')
                }}
              >
                Reset
              </button>
            </div>

            <div className="queue-breakdown">
              {(['north', 'south', 'east', 'west'] as Approach[]).map((a) => (
                <div key={a}>
                  <span>{a}</span>
                  <div className="bar">
                    <i style={{ width: `${Math.min(100, observation.queues[a] * 8)}%` }} />
                  </div>
                  <strong>{observation.queues[a]}</strong>
                </div>
              ))}
            </div>
          </aside>
          </div>

          <AppliedDecisions state={state} />
          <details className="historical-model" onToggle={event => setShowHistory(event.currentTarget.open)}>
            <summary>Historical LSTM forecast — separate from light control</summary>
            <p>{modelStatus}</p>
            {showHistory && <ThinkingPanel thought={thought} />}
          </details>
        </main>
      )}
    </div>
  )
}
