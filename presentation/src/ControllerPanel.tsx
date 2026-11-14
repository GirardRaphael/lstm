import { decideTraffic, observeTraffic, type Axis, type SignalDecision } from './trafficController'
import { phaseLabel, type SimState } from './simulation'

function actionLabel(decision: SignalDecision) {
  return decision.action === 'walk' ? 'Pedestrian crossing' : `${decision.action === 'ns' ? 'North–South' : 'East–West'} ${decision.movement === 'left' ? 'protected left turns' : 'green'}`
}
export function ControllerPanel({ state, onOverride }: { state: SimState; onOverride: (axis: Axis) => void }) {
  const observation = observeTraffic(state)
  const preview = decideTraffic(state)
  const applied = state.decisions[0]
  return (
    <section className="thinking-panel controller-panel" aria-label="Live traffic controller">
      <header className="thinking-header"><div>
        <p className="thinking-kicker">{state.adaptive ? 'Automatic decisions · active' : 'Fixed cycle · active'}</p>
        <h2>Traffic controller</h2>
      </div></header>
      <p>Reads the vehicles and pedestrians in this scene. Chooses the next green direction and its duration.</p>
      <div className="controller-current">
        <strong>Lights now: {phaseLabel(state.phase)}</strong>
        <span>{state.phaseTimer.toFixed(1)}s remaining</span>
        {state.phase === 'all-red' && <span>Waiting for the junction and crossings to clear.</span>}
      </div>
      <div className="stat-row">
        <div><span className="stat-label">Visible cars</span><strong>{observation.cars}</strong></div>
        <div><span className="stat-label">Visible trucks</span><strong>{observation.trucks}</strong></div>
        <div><span className="stat-label">Waiting pedestrians</span><strong>{observation.pedestrians}</strong></div>
      </div>
      <p>NS queue: {observation.queues.north + observation.queues.south} · EW queue: {observation.queues.east + observation.queues.west} (a truck counts as two cars).</p>
      <p>Waiting for green: NS {state.redWait.ns.toFixed(0)}s · EW {state.redWait.ew.toFixed(0)}s.</p>
      <h3>Last applied decision</h3>
      <p data-testid="applied-decision">{applied ? `${actionLabel(applied)} for ${applied.seconds.toFixed(1)}s. ${applied.reason}` : 'Starting: the first choice will be made after the junction clears.'}</p>
      <h3>If deciding now</h3>
      <p>{actionLabel(preview)} for {preview.seconds.toFixed(1)}s. {preview.reason}</p>
      <p className="thinking-note">Preview can change as traffic moves. A new choice is applied at the next clear junction, after yellow and clearance. Directions waiting 30s receive priority at that choice.</p>
      {state.controllerOverride && <p role="status">Queued request: {state.controllerOverride.toUpperCase()}. Pedestrians and waiting-time priority may go first.</p>}
      <div className="controller-actions">
        <button disabled={!state.adaptive} onClick={() => onOverride('ns')}>Request NS next</button>
        <button disabled={!state.adaptive} onClick={() => onOverride('ew')}>Request EW next</button>
      </div>
      <p className="thinking-note">This is an observation-based control algorithm, not a trained neural policy. It reads scene data, not screenshot pixels. The historical LSTM is shown separately below.</p>
    </section>
  )
}

export function AppliedDecisions({ state }: { state: SimState }) {
  return <section className="decision-ledger" aria-label="Applied signal decisions">
    <details open><summary className="ledger-header"><h2>Applied light decisions</h2><p>Actual choices sent to the simulated lights, newest first.</p></summary>
    <div className="ledger-body"><ol className="ledger-list">
      {state.decisions.map((decision) => <li className="ledger-row" key={decision.tick}>
        <strong>{actionLabel(decision)} · {decision.seconds.toFixed(1)}s · {decision.automatic ? 'automatic' : 'fixed cycle'}</strong>
        <p>{decision.reason}</p>
        <p>Observed: {decision.observation.cars} cars, {decision.observation.trucks} trucks, {decision.observation.pedestrians} waiting pedestrians. NS score {decision.scores.ns.toFixed(1)} · EW score {decision.scores.ew.toFixed(1)}.</p>
      </li>)}
    </ol>{!state.decisions.length && <p>Waiting for the first clear junction.</p>}</div></details>
  </section>
}
