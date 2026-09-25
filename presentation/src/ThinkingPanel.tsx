import { useMemo } from 'react'
import type { DecisionThought, GateKey } from './decisionBrain'
import { NeuralGraph } from './NeuralGraph'

const GATE_META: Record<GateKey, { label: string; hint: string }> = {
  f: { label: 'Forget', hint: 'keep memory' },
  i: { label: 'Input', hint: 'write new' },
  g: { label: 'Candidate', hint: 'what to write' },
  o: { label: 'Output', hint: 'expose state' },
}

type Props = {
  thought: DecisionThought
}

export function ThinkingPanel({ thought }: Props) {
  const topNeurons = useMemo(() => thought.topNeurons.slice(0, 5), [thought.topNeurons])
  const forecastText = Number.isFinite(thought.forecastVehicles)
    ? `~${Math.round(thought.forecastVehicles).toLocaleString()} veh/h`
    : 'unavailable'
  const sourceLabel =
    thought.source === 'sidecar'
      ? 'sidecar · Keras replay'
      : thought.source === 'lookup'
        ? 'closest real hour'
        : 'model unavailable'

  return (
    <section className="thinking-panel" aria-label="Neural decision thinking">
      <header className="thinking-header">
        <div>
          <p className="thinking-kicker">Decision brain · simulation</p>
          <h2>Neural decision network</h2>
        </div>
        <div className={`regime-pill regime-${thought.regime}`}>
          {thought.regime === 'jam' ? 'Jam' : thought.regime === 'building' ? 'Building' : 'Calm'}
        </div>
      </header>

      <NeuralGraph thought={thought} />

      <div className="gate-strip">
        {(Object.keys(GATE_META) as GateKey[]).map((key) => {
          const value = thought.gates[key]
          const pct = Math.round(value * 100)
          return (
            <div key={key} className="gate-pill" title={GATE_META[key].hint}>
              <span className="gate-name">{GATE_META[key].label}</span>
              <div className="gate-bar">
                <i style={{ width: `${Math.max(6, pct)}%` }} />
              </div>
              <span className="gate-value">{pct}%</span>
            </div>
          )
        })}
      </div>

      {topNeurons.length > 0 ? (
        <div className="active-units">
          <span className="active-label">Active units</span>
          <div className="unit-chips">
            {topNeurons.map((n) => (
              <span
                key={`${n.unit}-${n.final}`}
                className="unit-chip"
                title={`${n.role} · forget ${(n.mean_forget * 100).toFixed(0)}%`}
              >
                u{n.unit}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="decision-flow">
        <div className="flow-step">
          <span className="flow-num">1</span>
          <span>Sense {thought.congestedQueue} PCU on {thought.congested}</span>
        </div>
        <div className="flow-step">
          <span className="flow-num">2</span>
          <span>
            Forecast {forecastText}
            {thought.matchedHour ? ` · hour ${thought.matchedHour.timestamp}` : ''}
          </span>
        </div>
        <div className="flow-step">
          <span className="flow-num">3</span>
          <span>
            Heuristic {thought.recommendedAxis.toUpperCase()} green → {thought.recommendedGreen.toFixed(1)}s
          </span>
        </div>
      </div>

      <p className="source-pill" data-source={thought.source}>
        {sourceLabel}
      </p>
      <p className="thinking-note">{thought.sourceNote}</p>
      <p className="thinking-note">
        Cartoon lights are not a closed-loop controller. The LSTM did not choose the light.
      </p>
    </section>
  )
}
