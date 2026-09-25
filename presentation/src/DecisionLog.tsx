import type { Axis, DecisionLogState, DecisionRow } from './decisionLog'
import {
  axisShort,
  formatCounts,
  formatQueues,
  oppositeAxis,
  statusLabel,
} from './decisionLog'

type ModelBanner = 'loading' | 'sidecar' | 'lookup' | 'unavailable'

type Props = {
  log: DecisionLogState
  modelBanner: ModelBanner
  modelStatus: string
  pendingOverride: Axis | null
  onMarkCorrect: (rowId: number) => void
  onOverrideAxis: (rowId: number, axis: Axis) => void
}

function clock(ms: number): string {
  return new Date(ms).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function triggerLabel(row: DecisionRow): string {
  if (row.trigger === 'jam') return 'Jam created'
  if (row.trigger === 'forecast') return 'Closest hour changed'
  if (row.trigger === 'override') return 'User override'
  return row.phaseLabel
}

function modelBannerText(banner: ModelBanner, status: string): string {
  if (banner === 'loading') return `Loading model lookup… ${status}`
  if (banner === 'unavailable') {
    return `Sidecar/lookup missing — heuristic-only rows still log. ${status}`
  }
  if (banner === 'lookup') {
    return `Sidecar offline — static Keras replay for the closest real hour. ${status}`
  }
  return status
}

export function DecisionLog({
  log,
  modelBanner,
  modelStatus,
  pendingOverride,
  onMarkCorrect,
  onOverrideAxis,
}: Props) {
  return (
    <section className="decision-ledger" aria-label="Decision ledger">
      <header className="ledger-header">
        <div>
          <p className="ledger-kicker">Decision ledger · not a neural animation</p>
          <h2>What changed, and why</h2>
        </div>
        <p className="ledger-claim">
          Keras forecast grades the <em>model</em>. Green axis/time is a <em>queue heuristic</em>{' '}
          driven by that forecast — the LSTM did not choose the light.
        </p>
      </header>

      <p className="ledger-model" data-banner={modelBanner} role="status">
        {modelBannerText(modelBanner, modelStatus)}
      </p>

      {pendingOverride ? (
        <p className="ledger-pending">
          Next heuristic recommendation will serve {axisShort(pendingOverride)} (user correction —
          model weights unchanged).
        </p>
      ) : null}

      {log.rows.length === 0 ? (
        <p className="ledger-empty">
          No decisions yet — open live and wait for a phase change or jam.
        </p>
      ) : (
        <ol className="ledger-list">
          {log.rows.map((row) => {
            const other = oppositeAxis(row.heuristic.axis)
            const fv = row.forecastVerdict
            return (
              <li key={row.id} className={`ledger-row status-${row.status}`}>
                <div className="ledger-row-top">
                  <span className="ledger-time">
                    {clock(row.atMs)} · tick {row.tick}
                  </span>
                  <span className="ledger-phase">{triggerLabel(row)}</span>
                  <span className={`ledger-status status-${row.status}`}>
                    {statusLabel(row.status)}
                  </span>
                </div>

                <p className="ledger-narrative">{row.narrative}</p>

                <dl className="ledger-facts">
                  <div>
                    <dt>Sensed</dt>
                    <dd>
                      Heaviest {row.sensed.heaviest} {row.sensed.pcu} PCU · live{' '}
                      {formatCounts(row.sensed.liveCounts)} · {formatQueues(row.sensed.queues)}
                    </dd>
                  </div>
                  <div>
                    <dt>Model</dt>
                    <dd>
                      {row.matchedHour
                        ? `${row.matchedHour.timestamp} · Keras ${
                            row.forecastVehicles != null ? Math.round(row.forecastVehicles) : '—'
                          } vs actual ${Math.round(row.matchedHour.actual)} · residual ${
                            fv.residual == null
                              ? 'n/a'
                              : `${fv.residual > 0 ? '+' : ''}${Math.round(fv.residual)}`
                          } (${fv.quality}${
                            fv.vsNaive === 'beats_naive'
                              ? ', beats naive last-hour MAE'
                              : fv.vsNaive === 'worse_than_naive'
                                ? ', worse than naive last-hour MAE'
                                : ''
                          })`
                        : `Forecast ${row.source === 'unavailable' ? 'unavailable' : 'pending'} — heuristic only`}
                    </dd>
                  </div>
                  <div>
                    <dt>Heuristic</dt>
                    <dd>
                      {axisShort(row.heuristic.axis)} {row.heuristic.greenSeconds.toFixed(1)}s —{' '}
                      {row.heuristic.why}
                      {row.heuristicVerdict.queueShrunk == null
                        ? ' · queue after green: pending'
                        : row.heuristicVerdict.queueShrunk
                          ? ' · served queue shrank'
                          : ' · served queue did not shrink'}
                    </dd>
                  </div>
                </dl>

                <div className="ledger-correct" aria-label="Correct this decision">
                  <button
                    type="button"
                    disabled={row.statusLocked && row.status === 'looks_right'}
                    onClick={() => onMarkCorrect(row.id)}
                  >
                    Looks right
                  </button>
                  <button type="button" onClick={() => onOverrideAxis(row.id, other)}>
                    I would have served {axisShort(other)}
                  </button>
                  <span className="ledger-next-label">Next heuristic:</span>
                  <button type="button" onClick={() => onOverrideAxis(row.id, 'ns')}>
                    Serve NS
                  </button>
                  <button type="button" onClick={() => onOverrideAxis(row.id, 'ew')}>
                    Serve EW
                  </button>
                </div>
              </li>
            )
          })}
        </ol>
      )}
    </section>
  )
}
