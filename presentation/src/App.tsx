import { useEffect, useMemo, useRef, useState } from 'react'
import { IntersectionCanvas } from './IntersectionCanvas'
import { SlideView } from './SlideView'
import { ThinkingPanel } from './ThinkingPanel'
import { DecisionLog } from './DecisionLog'
import {
  congestionIntensity,
  thinkDecision,
  type HourLookupFile,
  type LiveModelOutput,
} from './decisionBrain'
import {
  applyUserOverride,
  emptyDecisionLog,
  ingestSnapshot,
  markRowCorrect,
  snapshotFromLive,
  type Axis,
  type LogTrigger,
} from './decisionLog'
import { SLIDES } from './slides'
import {
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

type Mode = 'slides' | 'live'

function countByKind(state: SimState) {
  let cars = 0
  let trucks = 0
  let pedestrians = 0
  for (const v of state.vehicles) {
    if (v.crossed) continue
    if (v.kind === 'car') cars += 1
    else trucks += 1
  }
  for (const p of state.pedestrians) {
    if (!p.done) pedestrians += 1
  }
  return { cars, trucks, pedestrians }
}

export default function App() {
  const [mode, setMode] = useState<Mode>('slides')
  const [slideIndex, setSlideIndex] = useState(0)
  const [cars, setCars] = useState(8)
  const [trucks, setTrucks] = useState(3)
  const [pedestrians, setPedestrians] = useState(20)
  const [jamApproach, setJamApproach] = useState<Approach>('east')
  const [adaptive, setAdaptive] = useState(true)
  const [running, setRunning] = useState(true)
  const [lastAction, setLastAction] = useState('Ready — randomize traffic or force a jam.')
  const [lookup, setLookup] = useState<HourLookupFile | null>(null)
  const [liveModel, setLiveModel] = useState<LiveModelOutput | null>(null)
  const [modelStatus, setModelStatus] = useState('Loading Keras hour lookup…')
  const [state, setState] = useState<SimState>(() =>
    spawnAgents(createInitialState(true), { cars: 12, trucks: 3, pedestrians: 20 }),
  )
  const [log, setLog] = useState(emptyDecisionLog)
  const lastTs = useRef<number | null>(null)
  const seededLive = useRef(false)

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
        setModelStatus('Model lookup failed to load — forecast hidden; green stays heuristic.')
      })
  }, [])

  useEffect(() => {
    setState((prev) => ({ ...prev, adaptive }))
  }, [adaptive])

  useEffect(() => {
    if (mode === 'live' && !seededLive.current) {
      seededLive.current = true
      const counts = randomSpawnCounts()
      setCars(counts.cars)
      setTrucks(counts.trucks)
      setPedestrians(counts.pedestrians)
      setState(() => {
        const fresh = createInitialState(adaptive)
        return spawnAgents(fresh, counts)
      })
      setLastAction(
        `Auto-seeded ${counts.cars} cars, ${counts.trucks} trucks, ${counts.pedestrians} pedestrians.`,
      )
    }
  }, [mode, adaptive])

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
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
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
        setLastAction(`Random jam on the ${approach} approach — watch the decision brain.`)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode])

  const queueTotal = useMemo(
    () => state.queues.north + state.queues.south + state.queues.east + state.queues.west,
    [state.queues],
  )
  const liveCounts = useMemo(() => countByKind(state), [state])
  const intensityBucket = Math.round(congestionIntensity(state) * 40)

  useEffect(() => {
    if (mode !== 'live') return
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
  }, [mode, intensityBucket, lookup])

  const thought = useMemo(
    () => thinkDecision(state, liveModel, lookup, log.pendingAxisOverride),
    [state, liveModel, lookup, log.pendingAxisOverride],
  )
  const latestLive = useRef({ state, thought, liveCounts })

  useEffect(() => {
    latestLive.current = { state, thought, liveCounts }
  }, [state, thought, liveCounts])

  const ingestTrigger = (trigger: LogTrigger) => {
    setLog((prev) => {
      const live = latestLive.current
      return ingestSnapshot(
        prev,
        snapshotFromLive({
          trigger,
          phase: live.state.phase,
          phaseLabel: phaseLabel(live.state.phase),
          tick: live.state.tick,
          jamActive: live.state.jamActive,
          queues: live.state.queues,
          liveCounts: live.liveCounts,
          thought: live.thought,
          adaptive: live.state.adaptive,
        }),
      )
    })
  }

  const forecastKey = `${thought.matchedHour?.index ?? 'none'}:${
    Number.isFinite(thought.forecastVehicles) ? Math.round(thought.forecastVehicles) : 'na'
  }`

  useEffect(() => {
    if (mode !== 'live') return
    ingestTrigger('phase')
  }, [mode, state.phase])

  useEffect(() => {
    if (mode !== 'live') return
    ingestTrigger('jam')
  }, [mode, state.jamActive])

  useEffect(() => {
    if (mode !== 'live') return
    ingestTrigger('forecast')
  }, [mode, forecastKey])

  const modelBanner = liveModel
    ? 'sidecar'
    : lookup
      ? 'lookup'
      : modelStatus.toLowerCase().includes('loading')
        ? 'loading'
        : 'unavailable'

  const markDecisionCorrect = (rowId: number) => {
    setLog((prev) => markRowCorrect(prev, rowId))
    setLastAction('You marked that ledger row as looking right. The model did not learn from this.')
  }

  const overrideDecisionAxis = (rowId: number, axis: Axis) => {
    setLog((prev) => {
      const live = latestLive.current
      return applyUserOverride(
        prev,
        rowId,
        axis,
        snapshotFromLive({
          trigger: 'override',
          phase: live.state.phase,
          phaseLabel: phaseLabel(live.state.phase),
          tick: live.state.tick,
          jamActive: live.state.jamActive,
          queues: live.state.queues,
          liveCounts: live.liveCounts,
          thought: live.thought,
          adaptive: live.state.adaptive,
        }),
      )
    })
    setLastAction(
      `You corrected the next heuristic: serve ${axis.toUpperCase()}. The LSTM did not learn from this.`,
    )
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
        <div className="top-hint">
          {mode === 'slides' ? '← → navigate · L live' : 'R spawn · J jam · watch the decision brain'}
        </div>
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
        <main className="live-stage thinking-layout">
          <div className="live-windows">
            <section className="live-visual">
            <IntersectionCanvas state={state} />
            <div className={`response-banner ${state.jamActive ? 'hot' : ''}`}>
              <strong>{state.jamActive ? 'Jam response active' : 'Steady traffic'}</strong>
              <span>
                Heaviest: {thought.congested} ({thought.congestedQueue} PCU) · regime {thought.regime} ·
                forecast{' '}
                {Number.isFinite(thought.forecastVehicles)
                  ? `~${Math.round(thought.forecastVehicles)} veh/h`
                  : 'unavailable'}{' '}
                ({thought.source}) · heuristic {thought.recommendedAxis.toUpperCase()} green{' '}
                {thought.recommendedGreen.toFixed(1)}s
                {thought.userOverride ? ' · user correction (not learned)' : ''}
              </span>
            </div>
            <p className="sim-disclaimer">
              Simulation only — not city signals. Cartoon lights are not a closed-loop controller.
              Forecast is a real Keras-weight replay for the closest Metro Interstate hour.{' '}
              {modelStatus}
            </p>
          </section>

          <ThinkingPanel thought={thought} />

          <aside className="live-panel">
            <h2>Intersection controls</h2>
            <p className="panel-lead">
              Force a jam. Gates and neurons come from the closest real test hour. Green time is a
              queue heuristic driven by that forecast — the LSTM did not choose the light.
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
                <span className="stat-label">Queue (PCU)</span>
                <strong>{queueTotal}</strong>
              </div>
              <div>
                <span className="stat-label">On road now</span>
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
                onChange={(e) => setAdaptive(e.target.checked)}
              />
              Adaptive green (heuristic from forecast / queue — not LSTM-chosen lights)
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
                    `Traffic jam on ${jamApproach} — watch gates and neurons before green moves.`,
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
                  setState(createInitialState(adaptive))
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
                    <i style={{ width: `${Math.min(100, state.queues[a] * 8)}%` }} />
                  </div>
                  <strong>{state.queues[a]}</strong>
                </div>
              ))}
            </div>
          </aside>
          </div>

          <DecisionLog
            log={log}
            modelBanner={modelBanner}
            modelStatus={modelStatus}
            pendingOverride={log.pendingAxisOverride}
            onMarkCorrect={markDecisionCorrect}
            onOverrideAxis={overrideDecisionAxis}
          />
        </main>
      )}
    </div>
  )
}
