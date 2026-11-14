import { cameraReady, decideTraffic, observeTraffic, type Axis, type SignalDecision } from './trafficController'
import { phaseLabel, type SimState } from './simulation'

function actionLabel(decision: SignalDecision) {
  return decision.action === 'walk' ? 'Pedestrian crossing' : `${decision.action === 'ns' ? 'North–South' : 'East–West'} ${decision.movement === 'left' ? 'protected left turns' : 'green'}`
}
function sourceLabel(decision: SignalDecision) {
  return decision.source === 'camera'
    ? `Screenshot${decision.frameCapturedAt == null ? '' : ` at ${decision.frameCapturedAt.toFixed(2)}s`}`
    : 'Scene data'
}
export function ControllerPanel({ state, onOverride }: { state: SimState; onOverride: (axis: Axis) => void }) {
  const observation = observeTraffic(state)
  const ready = state.controlSource !== 'camera' || cameraReady(state)
  const preview = ready ? decideTraffic(state) : null
  const applied = state.decisions[0]
  const camera = state.controlSource === 'camera'
  return (
    <section className="thinking-panel controller-panel" aria-label="Live traffic controller">
      <header className="thinking-header"><div>
        <p className="thinking-kicker">{state.adaptive ? 'Automatic decisions' : 'Fixed cycle'} · {ready ? 'active' : 'camera hold'}</p>
        <h2>Traffic controller</h2>
      </div></header>
      <p>{camera ? 'Uses vehicles and pedestrians detected in screenshots to choose the next green direction and its duration.' : 'Uses the scene objects to choose the next green direction and its duration.'}</p>
      <div className="controller-current">
        <strong>Lights now: {phaseLabel(state.phase)}</strong>
        <span>{Math.max(0, state.phaseTimer).toFixed(1)}s remaining</span>
        {!ready ? <span role="status">Waiting for a fresh, readable screenshot. Yellow and clearance finish before the lights hold red.</span> : state.phase === 'all-red' ? <span>Waiting for the junction and crossings to clear.</span> : null}
      </div>
      <div className="stat-row">
        <div><span className="stat-label">{camera ? 'Detected cars' : 'Visible cars'}</span><strong>{ready ? observation.cars : '—'}</strong></div>
        <div><span className="stat-label">{camera ? 'Detected trucks' : 'Visible trucks'}</span><strong>{ready ? observation.trucks : '—'}</strong></div>
        <div><span className="stat-label">Waiting pedestrians</span><strong>{ready ? observation.pedestrians : '—'}</strong></div>
      </div>
      <p>{ready ? `Estimated NS queue: ${observation.queues.north + observation.queues.south} · EW queue: ${observation.queues.east + observation.queues.west} (a truck counts as two cars).` : 'Queue estimates are unavailable until the camera is ready.'}</p>
      <p>Waiting for green: NS {state.redWait.ns.toFixed(0)}s · EW {state.redWait.ew.toFixed(0)}s.</p>
      <h3>Last applied decision</h3>
      <p data-testid="applied-decision">{applied ? `${actionLabel(applied)} for ${applied.seconds.toFixed(1)}s. ${applied.reason}` : 'Starting: the first choice will be made after the junction clears.'}</p>
      {applied ? <p className="thinking-note">Input: {sourceLabel(applied)}.</p> : null}
      <h3>If deciding now</h3>
      <p>{preview ? `${actionLabel(preview)} for ${preview.seconds.toFixed(1)}s. ${preview.reason}` : 'No new green or walk phase while the camera input is missing, unreadable or stale.'}</p>
      <p className="thinking-note">Preview can change as traffic moves. A new choice is applied at the next clear junction, after yellow and clearance. Directions waiting 30s receive priority at that choice.</p>
      {state.controllerOverride ? <p role="status">Queued request: {state.controllerOverride.toUpperCase()}. Pedestrians and waiting-time priority may go first.</p> : null}
      <div className="controller-actions">
        <button disabled={!state.adaptive || !ready} onClick={() => onOverride('ns')}>Request NS next</button>
        <button disabled={!state.adaptive || !ready} onClick={() => onOverride('ew')}>Request EW next</button>
      </div>
      <p className="thinking-note">{camera ? 'Image recognition is calibrated to this app’s graphics. It measures screenshot pixels from simulated traffic; it is not a trained model for real road footage.' : 'Scene mode reads simulation data directly.'} The light policy uses rules, and the historical LSTM is separate.</p>
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
        <p>{sourceLabel(decision)}: {decision.observation.cars} cars, {decision.observation.trucks} trucks, {decision.observation.pedestrians} waiting pedestrians. NS score {decision.scores.ns.toFixed(1)} · EW score {decision.scores.ew.toFixed(1)}.</p>
      </li>)}
    </ol>{!state.decisions.length ? <p>Waiting for the first clear junction.</p> : null}</div></details>
  </section>
}
