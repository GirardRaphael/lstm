import { describe, expect, it } from 'vitest'
import { createInitialState, type SimState } from './simulation'
import {
  demandFromCongestion,
  isRealModelRow,
  pickNearestHour,
  thinkDecision,
  type HourLookupFile,
  type LiveModelOutput,
  type ModelHourRow,
} from './decisionBrain'
import { SLIDES } from './slides'

function hour(partial: Partial<ModelHourRow> & Pick<ModelHourRow, 'actual_vehicles' | 'predicted_vehicles' | 'index'>): ModelHourRow {
  return {
    timestamp: '2018-01-01 00:00:00',
    gates: { i: 0.4, f: 0.5, g: 0.1, o: 0.4 },
    neurons: Array.from({ length: 32 }, (_, i) => i * 0.01),
    dense: Array.from({ length: 16 }, (_, i) => i * 0.02),
    top_neurons: [{ unit: 1, mean_abs: 0.2, final: 0.1, role: 'Mixed', mean_forget: 0.5 }],
    ...partial,
  }
}

function stateWith(queues: Partial<SimState['queues']>, jam = false): SimState {
  const base = createInitialState(true)
  return {
    ...base,
    jamActive: jam,
    queues: { ...base.queues, ...queues },
  }
}

const LOOKUP: HourLookupFile = {
  model: 'baseline_univariate',
  claim: 'Replayed Keras output for the closest real hour',
  hours: [
    hour({ index: 4068, actual_vehicles: 151, predicted_vehicles: 520.48 }),
    hour({ index: 2000, actual_vehicles: 3600, predicted_vehicles: 3400 }),
    hour({ index: 4009, actual_vehicles: 7213, predicted_vehicles: 6163.64 }),
  ],
}

describe('real model lookup — not a quiet/rush blend', () => {
  it('maps mid congestion onto the nearest real hour forecast exactly', () => {
    const state = stateWith({ east: 11, west: 0, north: 0, south: 0 })
    const thought = thinkDecision(state, null, LOOKUP)
    const demand = demandFromCongestion(thought.intensity)
    const picked = pickNearestHour(demand, LOOKUP.hours)
    expect(picked?.predicted_vehicles).toBe(3400)
    expect(thought.source).toBe('lookup')
    expect(thought.forecastVehicles).toBe(3400)
    const quietPred = 520.48
    const rushPred = 6163.64
    const blended = quietPred + thought.intensity * (rushPred - quietPred)
    expect(thought.forecastVehicles).not.toBeCloseTo(blended, 0)
    expect([quietPred, rushPred]).not.toContain(thought.forecastVehicles)
  })

  it('uses sidecar output when present instead of the static lookup', () => {
    const live: LiveModelOutput = {
      ...hour({ index: 77, actual_vehicles: 4100, predicted_vehicles: 3333 }),
      source: 'sidecar',
      label: 'Replayed Keras output for the closest real hour',
    }
    const thought = thinkDecision(stateWith({ east: 20 }, true), live, LOOKUP)
    expect(thought.source).toBe('sidecar')
    expect(thought.forecastVehicles).toBe(3333)
    expect(thought.forecastVehicles).not.toBe(6163.64)
  })

  it('does not silently invent a theatrical forecast when lookup is missing', () => {
    const thought = thinkDecision(stateWith({ east: 18 }, true), null, null)
    expect(thought.source).toBe('unavailable')
    expect(Number.isFinite(thought.forecastVehicles)).toBe(false)
    expect(thought.forecastVehicles).not.toBe(800 + thought.intensity * 5000)
    expect(thought.sourceNote.toLowerCase()).toMatch(/unavailable|heuristic/)
    expect(thought.userOverride).toBeNull()
  })

  it('applies a user axis override to the heuristic without changing the Keras hour', () => {
    const liveState = stateWith({ east: 20, north: 3 }, true)
    const baseline = thinkDecision(liveState, null, LOOKUP)
    expect(baseline.recommendedAxis).toBe('ew')
    const corrected = thinkDecision(liveState, null, LOOKUP, 'ns')
    expect(corrected.recommendedAxis).toBe('ns')
    expect(corrected.userOverride).toBe('ns')
    expect(corrected.forecastVehicles).toBe(baseline.forecastVehicles)
    expect(corrected.matchedHour).toEqual(baseline.matchedHour)
    expect(corrected.why).toMatch(/did not learn/i)
  })

  it('treats only finite Keras rows as real model output', () => {
    expect(isRealModelRow(LOOKUP.hours[0])).toBe(true)
    expect(isRealModelRow(hour({ index: 0, actual_vehicles: 1, predicted_vehicles: Number.NaN }))).toBe(
      false,
    )
  })
})

describe('honest comparison slides', () => {
  const text = JSON.stringify(SLIDES)

  it('states XGBoost beat LSTM on the motorway MAE table', () => {
    expect(text).toContain('154.3')
    expect(text).toContain('201.4')
    expect(text).toContain('228.8')
    expect(text).toContain('585.6')
    expect(text).toMatch(/XGBoost won on (the )?motorway/i)
  })

  it('states LSTM won narrowly on bike-sharing and that the verdict flips', () => {
    expect(text).toContain('41.3')
    expect(text).toContain('43.1')
    expect(text).toMatch(/LSTM won/i)
    expect(text).toMatch(/verdict flips/i)
    expect(text).toMatch(/bike/i)
  })

  it('says the intersection is a simulation, not city signals', () => {
    expect(text).toMatch(/simulation/i)
    expect(text).toMatch(/not (connected to )?city signals/i)
    expect(text).toMatch(/closed-loop controller/i)
  })
})
