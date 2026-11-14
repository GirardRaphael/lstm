import { describe, expect, it } from 'vitest'
import { thinkDecision, type HourLookupFile, type ModelHourRow } from './decisionBrain'
import { createInitialState, type SimState } from './simulation'
import {
  applyUserOverride,
  emptyDecisionLog,
  forecastResidual,
  gradeForecast,
  ingestSnapshot,
  markRowCorrect,
  NAIVE_LAST_HOUR_MAE,
  servedLongestQueue,
  shouldAppend,
  type DecisionSnapshot,
  type QueueSnapshot,
} from './decisionLog'

function hour(
  partial: Partial<ModelHourRow> & Pick<ModelHourRow, 'actual_vehicles' | 'predicted_vehicles' | 'index'>,
): ModelHourRow {
  return {
    timestamp: '2018-04-12 16:00:00',
    gates: { i: 0.4, f: 0.5, g: 0.1, o: 0.4 },
    neurons: Array.from({ length: 32 }, (_, i) => i * 0.01),
    dense: Array.from({ length: 16 }, (_, i) => i * 0.02),
    top_neurons: [{ unit: 1, mean_abs: 0.2, final: 0.1, role: 'Mixed', mean_forget: 0.5 }],
    ...partial,
  }
}

const LOOKUP: HourLookupFile = {
  model: 'baseline_univariate',
  claim: 'Replayed Keras output for the closest real hour',
  hours: [hour({ index: 4009, actual_vehicles: 7213, predicted_vehicles: 6163.64 })],
}

function queues(partial: Partial<QueueSnapshot>): QueueSnapshot {
  return { north: 0, south: 0, east: 0, west: 0, ...partial }
}

function snap(partial: Partial<DecisionSnapshot> & Pick<DecisionSnapshot, 'trigger' | 'phase'>): DecisionSnapshot {
  return {
    phaseLabel: partial.phase,
    tick: 10,
    jamActive: false,
    queues: queues({ east: 16, west: 2, north: 4, south: 1 }),
    liveCounts: { cars: 12, trucks: 3, pedestrians: 8 },
    heaviest: 'east',
    heaviestPcu: 16,
    recommendedAxis: 'ew',
    recommendedGreen: 14,
    forecastVehicles: 6163.64,
    matchedHour: { index: 4009, timestamp: '2018-04-12 16:00:00', actual: 7213 },
    source: 'lookup',
    userOverride: null,
    adaptive: true,
    atMs: 1_000,
    ...partial,
  }
}

function stateWith(q: Partial<SimState['queues']>): SimState {
  const base = createInitialState(true)
  return { ...base, queues: { ...base.queues, ...q } }
}

describe('decision ledger builder', () => {
  it('seeds the first phase without a row, then appends on phase change', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ns-green' }))
    expect(log.rows).toHaveLength(0)
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ns-yellow' }))
    expect(log.rows).toHaveLength(1)
    expect(log.rows[0].phase).toBe('ns-yellow')
    expect(log.rows[0].trigger).toBe('phase')
  })

  it('does not append the same phase on every frame', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ns-green' }))
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ew-green', tick: 11 }))
    expect(log.rows).toHaveLength(1)
    for (let i = 0; i < 40; i += 1) {
      log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ew-green', tick: 20 + i, atMs: 2000 + i }))
    }
    expect(log.rows).toHaveLength(1)
    expect(shouldAppend(log, snap({ trigger: 'phase', phase: 'ew-green' }))).toBe(false)
  })

  it('appends jam created once, not while still jammed', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'jam', phase: 'ns-green', jamActive: false }))
    expect(log.rows).toHaveLength(0)
    log = ingestSnapshot(log, snap({ trigger: 'jam', phase: 'ns-green', jamActive: true }))
    expect(log.rows).toHaveLength(1)
    expect(log.rows[0].trigger).toBe('jam')
    log = ingestSnapshot(log, snap({ trigger: 'jam', phase: 'ns-green', jamActive: true, tick: 99 }))
    expect(log.rows).toHaveLength(1)
    log = ingestSnapshot(log, snap({ trigger: 'jam', phase: 'ns-green', jamActive: false }))
    expect(log.rows).toHaveLength(1)
    log = ingestSnapshot(log, snap({ trigger: 'jam', phase: 'ns-green', jamActive: true, tick: 100 }))
    expect(log.rows).toHaveLength(2)
  })

  it('appends when the closest hour / forecast changes, not on the first seed', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'forecast',
        phase: 'ns-green',
        forecastVehicles: 6163.64,
        matchedHour: { index: 4009, timestamp: '2018-04-12 16:00:00', actual: 7213 },
      }),
    )
    expect(log.rows).toHaveLength(0)
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'forecast',
        phase: 'ns-green',
        forecastVehicles: 3400,
        matchedHour: { index: 2000, timestamp: '2018-03-01 08:00:00', actual: 3600 },
      }),
    )
    expect(log.rows).toHaveLength(1)
    expect(log.rows[0].matchedHour?.index).toBe(2000)
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'forecast',
        phase: 'ns-green',
        forecastVehicles: 3400,
        matchedHour: { index: 2000, timestamp: '2018-03-01 08:00:00', actual: 3600 },
        tick: 50,
      }),
    )
    expect(log.rows).toHaveLength(1)
  })

  it('records an override and applies it to the next heuristic recommendation', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ns-green' }))
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ew-green' }))
    expect(log.rows[0].heuristic.axis).toBe('ew')
    log = applyUserOverride(log, log.rows[0].id, 'ns', snap({ trigger: 'override', phase: 'ew-green', atMs: 5000 }))
    expect(log.pendingAxisOverride).toBe('ns')
    expect(log.rows[0].trigger).toBe('override')
    expect(log.rows[0].heuristic.userOverride).toBe('ns')
    expect(log.rows[0].status).toBe('you_corrected')
    expect(log.rows[1].status).toBe('you_corrected')
    expect(log.rows[0].narrative).toMatch(/did not learn/i)

    const thought = thinkDecision(stateWith({ east: 16, north: 2 }), null, LOOKUP, log.pendingAxisOverride)
    expect(thought.recommendedAxis).toBe('ns')
    expect(thought.userOverride).toBe('ns')
    expect(thought.congested).toBe('east')
    expect(thought.why).toMatch(/did not learn/i)

    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ns-green',
        recommendedAxis: thought.recommendedAxis,
        userOverride: thought.userOverride,
        atMs: 6000,
      }),
    )
    expect(log.pendingAxisOverride).toBeNull()
    expect(log.rows[0].heuristic.axis).toBe('ns')
    expect(log.rows[0].heuristic.userOverride).toBe('ns')
  })

  it('computes forecast residual and grades against naive last-hour MAE', () => {
    expect(forecastResidual(6164, 7213)).toBe(-1049)
    const poor = gradeForecast(6163.64, 7213)
    expect(poor.quality).toBe('poor')
    expect(poor.vsNaive).toBe('worse_than_naive')
    expect(poor.absResidual).toBeGreaterThan(NAIVE_LAST_HOUR_MAE)
    const good = gradeForecast(7200, 7213)
    expect(good.quality).toBe('good')
    expect(good.vsNaive).toBe('beats_naive')
    expect(gradeForecast(Number.NaN, 7213).quality).toBe('unavailable')
  })

  it('flags whether the heuristic served the longest queue', () => {
    expect(servedLongestQueue('ew', 'east')).toBe(true)
    expect(servedLongestQueue('ew', 'west')).toBe(true)
    expect(servedLongestQueue('ns', 'east')).toBe(false)

    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ns-green' }))
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ew-green',
        recommendedAxis: 'ew',
        heaviest: 'east',
        heaviestPcu: 16,
      }),
    )
    expect(log.rows[0].heuristicVerdict.servedLongest).toBe(true)

    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ns-green',
        recommendedAxis: 'ns',
        heaviest: 'east',
        heaviestPcu: 16,
        userOverride: 'ns',
        atMs: 8000,
      }),
    )
    expect(log.rows[0].heuristicVerdict.servedLongest).toBe(false)
    expect(log.rows[0].status).toBe('you_corrected')
  })

  it('samples queues at decision time and when the green phase ends', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'all-red' }))
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ew-green',
        queues: queues({ east: 16, west: 4, north: 2, south: 1 }),
        recommendedAxis: 'ew',
      }),
    )
    const green = log.rows[0]
    expect(green.heuristicVerdict.servedQueueBefore).toBe(20)
    expect(green.heuristicVerdict.queueShrunk).toBeNull()

    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ew-yellow',
        queues: queues({ east: 7, west: 2, north: 3, south: 1 }),
      }),
    )
    const closed = log.rows.find((r) => r.id === green.id)
    expect(closed?.heuristicVerdict.servedQueueAfter).toBe(9)
    expect(closed?.heuristicVerdict.queueShrunk).toBe(true)
    expect(closed?.narrative).toMatch(/shrank/)
  })

  it('still builds a heuristic-only row when forecast is missing', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ns-green' }))
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ew-green',
        forecastVehicles: Number.NaN,
        matchedHour: null,
        source: 'unavailable',
      }),
    )
    expect(log.rows[0].forecastVerdict.quality).toBe('unavailable')
    expect(log.rows[0].narrative).toMatch(/heuristic-only/i)
    expect(log.rows[0].heuristic.axis).toBe('ew')
  })

  it('locks Looks right so later shrink data does not unmark it', () => {
    let log = emptyDecisionLog()
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'all-red' }))
    log = ingestSnapshot(log, snap({ trigger: 'phase', phase: 'ew-green' }))
    log = markRowCorrect(log, log.rows[0].id)
    expect(log.rows[0].status).toBe('looks_right')
    log = ingestSnapshot(
      log,
      snap({
        trigger: 'phase',
        phase: 'ew-yellow',
        queues: queues({ east: 20, west: 8 }),
      }),
    )
    const closed = log.rows.find((r) => r.statusLocked)
    expect(closed?.status).toBe('looks_right')
  })
})
