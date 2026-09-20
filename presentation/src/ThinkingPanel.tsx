import { useMemo } from 'react'
import type { DecisionThought, GateKey, NeuronTraceFile } from './decisionBrain'

const GATE_META: Record<GateKey, { label: string; hint: string }> = {
  f: { label: 'Forget', hint: 'keep vs erase memory' },
  i: { label: 'Input', hint: 'write new evidence' },
  g: { label: 'Candidate', hint: 'what to write' },
  o: { label: 'Output', hint: 'expose to decision' },
}

type Props = {
  thought: DecisionThought
  traces: NeuronTraceFile | null
}

function activationColor(v: number): string {
  const t = Math.max(-1, Math.min(1, v))
  if (t >= 0) {
    const a = 0.15 + t * 0.85
    return `rgba(46, 196, 182, ${a})`
  }
  const a = 0.15 + Math.abs(t) * 0.85
  return `rgba(56, 189, 248, ${a})`
}

export function ThinkingPanel({ thought, traces }: Props) {
  const neuronCells = useMemo(() => thought.neurons.slice(0, 32), [thought.neurons])
  const denseCells = useMemo(() => thought.dense.slice(0, 16), [thought.dense])

  return (
    <section className="thinking-panel" aria-label="Neural decision thinking">
      <header className="thinking-header">
        <div>
          <p className="thinking-kicker">Decision brain · live</p>
          <h2>How the network is thinking</h2>
        </div>
        <div className={`regime-pill regime-${thought.regime}`}>
          {thought.regime === 'jam' ? 'Jam regime' : thought.regime === 'building' ? 'Building' : 'Calm'}
        </div>
      </header>

      <p className="thinking-claim">
        {traces?.claim ??
          'Gate values come from the trained LSTM weights (NumPy replay of Keras).'}
      </p>

      <div className="arch-flow" aria-hidden>
        <span>24h history</span>
        <i />
        <span>LSTM 64</span>
        <i />
        <span>LSTM 32</span>
        <i />
        <span>Dense 16</span>
        <i />
        <span className="arch-out">Forecast → green</span>
      </div>

      <div className="gate-grid">
        {(Object.keys(GATE_META) as GateKey[]).map((key) => {
          const value = thought.gates[key]
          return (
            <div key={key} className="gate-card">
              <div className="gate-top">
                <strong>{GATE_META[key].label}</strong>
                <em>{(value * 100).toFixed(0)}%</em>
              </div>
              <div className="gate-track">
                <i style={{ width: `${Math.max(4, value * 100)}%` }} />
              </div>
              <span>{GATE_META[key].hint}</span>
            </div>
          )
        })}
      </div>

      <div className="neuron-block">
        <div className="neuron-block-title">
          <strong>LSTM-2 neurons</strong>
          <span>teal = positive activation · blue = negative</span>
        </div>
        <div className="neuron-grid" style={{ gridTemplateColumns: 'repeat(8, 1fr)' }}>
          {neuronCells.map((v, idx) => (
            <div
              key={idx}
              className="neuron-cell"
              title={`u${idx}: ${v.toFixed(3)}`}
              style={{ background: activationColor(v) }}
            />
          ))}
        </div>
      </div>

      <div className="neuron-block">
        <div className="neuron-block-title">
          <strong>Dense decision head</strong>
          <span>ReLU units feeding the forecast</span>
        </div>
        <div className="neuron-grid" style={{ gridTemplateColumns: 'repeat(8, 1fr)' }}>
          {denseCells.map((v, idx) => (
            <div
              key={idx}
              className="neuron-cell tall"
              title={`d${idx}: ${v.toFixed(3)}`}
              style={{ background: activationColor(Math.max(0, v)) }}
            />
          ))}
        </div>
      </div>

      {thought.topNeurons.length > 0 ? (
        <div className="top-neurons">
          <strong>Most active units right now</strong>
          <ul>
            {thought.topNeurons.map((n) => (
              <li key={`${n.unit}-${n.final}`}>
                <span>u{n.unit}</span>
                <span>{n.role}</span>
                <span>{n.final.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="thought-steps">
        <strong>Decision trace</strong>
        <ol>
          {thought.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <p className="thought-why">{thought.why}</p>
      </div>

      <div className="forecast-chip">
        <span>Implied demand</span>
        <strong>{Math.round(thought.forecastVehicles).toLocaleString()} veh/h</strong>
        <span>
          → {thought.recommendedAxis.toUpperCase()} green {thought.recommendedGreen.toFixed(1)}s
        </span>
      </div>
    </section>
  )
}
