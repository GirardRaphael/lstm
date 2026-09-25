import type { Approach, SimState } from './simulation'

export type GateKey = 'i' | 'f' | 'g' | 'o'

export type ModelNeuron = {
  unit: number
  mean_abs: number
  final: number
  role: string
  mean_forget: number
}

export type ModelHourRow = {
  index: number
  timestamp: string
  actual_vehicles: number
  predicted_vehicles: number
  gates: Record<GateKey, number>
  neurons: number[]
  dense: number[]
  top_neurons: ModelNeuron[]
}

export type HourLookupFile = {
  model: string
  claim: string
  hours: ModelHourRow[]
  quiet_actual?: number
  rush_actual?: number
}

export type ModelSource = 'sidecar' | 'lookup' | 'unavailable'

export type LiveModelOutput = ModelHourRow & {
  source: Exclude<ModelSource, 'unavailable'>
  demand_query?: number
  label?: string
  model?: string
}

export type DecisionThought = {
  intensity: number
  regime: 'calm' | 'building' | 'jam'
  forecastVehicles: number
  recommendedAxis: 'ns' | 'ew'
  recommendedGreen: number
  congested: Approach
  congestedQueue: number
  gates: Record<GateKey, number>
  neurons: number[]
  topNeurons: ModelNeuron[]
  dense: number[]
  steps: string[]
  why: string
  source: ModelSource
  sourceNote: string
  matchedHour: { index: number; timestamp: string; actual: number } | null
  /** Present when the operator forced the next heuristic axis. Not learned. */
  userOverride: 'ns' | 'ew' | null
}

/** Real Metro Interstate test-hour actuals used to map cartoon congestion. */
export const QUIET_ACTUAL = 151
export const RUSH_ACTUAL = 7213

export function congestionIntensity(state: SimState): number {
  const queues = Object.values(state.queues)
  const peak = Math.max(...queues, 0)
  const total = queues.reduce((s, v) => s + v, 0)
  const jamBoost = state.jamActive ? 0.25 : 0
  return Math.max(0, Math.min(1, peak / 28 + total / 120 + jamBoost))
}

export function heaviestApproach(state: SimState): { approach: Approach; queue: number } {
  const entries = Object.entries(state.queues) as [Approach, number][]
  entries.sort((a, b) => b[1] - a[1])
  return { approach: entries[0][0], queue: entries[0][1] }
}

export function demandFromCongestion(intensity: number): number {
  const t = Math.max(0, Math.min(1, intensity))
  return QUIET_ACTUAL + t * (RUSH_ACTUAL - QUIET_ACTUAL)
}

export function pickNearestHour(demand: number, hours: ModelHourRow[]): ModelHourRow | null {
  if (!hours.length) return null
  let best = hours[0]
  let bestDist = Math.abs(best.actual_vehicles - demand)
  for (let i = 1; i < hours.length; i += 1) {
    const dist = Math.abs(hours[i].actual_vehicles - demand)
    if (dist < bestDist) {
      best = hours[i]
      bestDist = dist
    }
  }
  return best
}

export function isRealModelRow(row: ModelHourRow | LiveModelOutput | null | undefined): boolean {
  return Boolean(
    row &&
      Number.isFinite(row.predicted_vehicles) &&
      row.gates &&
      Array.isArray(row.neurons) &&
      row.neurons.length > 0,
  )
}

function applyRow(row: ModelHourRow): Pick<
  DecisionThought,
  'forecastVehicles' | 'gates' | 'neurons' | 'topNeurons' | 'dense' | 'matchedHour'
> {
  return {
    forecastVehicles: row.predicted_vehicles,
    gates: { ...row.gates },
    neurons: row.neurons.slice(),
    topNeurons: row.top_neurons.slice(0, 6),
    dense: row.dense.slice(),
    matchedHour: {
      index: row.index,
      timestamp: row.timestamp,
      actual: row.actual_vehicles,
    },
  }
}

export function thinkDecision(
  state: SimState,
  live: LiveModelOutput | null,
  lookup: HourLookupFile | null,
  axisOverride: 'ns' | 'ew' | null = null,
): DecisionThought {
  const intensity = congestionIntensity(state)
  const { approach, queue } = heaviestApproach(state)
  const userOverride: 'ns' | 'ew' | null =
    axisOverride === 'ns' || axisOverride === 'ew' ? axisOverride : null
  const recommendedAxis: 'ns' | 'ew' =
    userOverride ?? (approach === 'north' || approach === 'south' ? 'ns' : 'ew')
  const queueForGreen = userOverride
    ? recommendedAxis === 'ns'
      ? Math.max(state.queues.north, state.queues.south)
      : Math.max(state.queues.east, state.queues.west)
    : queue
  const recommendedGreen = state.adaptive
    ? Math.min(14, Math.max(3.5, 3.5 + queueForGreen * 0.55))
    : recommendedAxis === 'ns'
      ? state.nsGreen
      : state.ewGreen

  const demand = demandFromCongestion(intensity)
  let source: ModelSource = 'unavailable'
  let sourceNote =
    'Model output unavailable — forecast hidden. Green time is still a local heuristic, not an LSTM control signal.'
  let forecastVehicles = Number.NaN
  let gates: Record<GateKey, number> = { i: 0, f: 0, g: 0, o: 0 }
  let neurons: number[] = []
  let topNeurons: ModelNeuron[] = []
  let dense: number[] = []
  let matchedHour: DecisionThought['matchedHour'] = null

  if (isRealModelRow(live) && live) {
    const applied = applyRow(live)
    forecastVehicles = applied.forecastVehicles
    gates = applied.gates
    neurons = applied.neurons
    topNeurons = applied.topNeurons
    dense = applied.dense
    matchedHour = applied.matchedHour
    source = 'sidecar'
    sourceNote =
      live.label ??
      'Live NumPy replay of baseline_univariate.keras on the closest real Metro Interstate hour. Replayed Keras output — not a closed-loop controller.'
  } else {
    const picked = pickNearestHour(demand, lookup?.hours ?? [])
    if (isRealModelRow(picked) && picked) {
      const applied = applyRow(picked)
      forecastVehicles = applied.forecastVehicles
      gates = applied.gates
      neurons = applied.neurons
      topNeurons = applied.topNeurons
      dense = applied.dense
      matchedHour = applied.matchedHour
      source = 'lookup'
      sourceNote =
        lookup?.claim ??
        'Replayed Keras output for the closest real hour. Not a blend of quiet/rush traces; not city signals.'
    }
  }

  const regime: DecisionThought['regime'] =
    intensity < 0.28 ? 'calm' : intensity < 0.62 ? 'building' : 'jam'

  const axisLabel = recommendedAxis === 'ns' ? 'North–South' : 'East–West'
  const forecastText = Number.isFinite(forecastVehicles)
    ? `${Math.round(forecastVehicles)}`
    : 'unavailable'
  const hourText = matchedHour
    ? `closest real hour ${matchedHour.timestamp} (test index ${matchedHour.index}, actual ${Math.round(matchedHour.actual)} veh/h)`
    : 'no real hour matched'

  const heuristicStep = userOverride
    ? `User correction (not learned): force ${axisLabel} green toward ${recommendedGreen.toFixed(1)}s. Model weights unchanged.`
    : state.adaptive
      ? `Heuristic (not the LSTM): stretch ${axisLabel} green toward ${recommendedGreen.toFixed(1)}s from the forecast/queue.`
      : `Heuristic held off — fixed ${axisLabel} green stays ${recommendedGreen.toFixed(1)}s.`

  const steps = [
    `Sense ${queue} PCU waiting on ${approach} (simulated intersection).`,
    source === 'unavailable'
      ? 'No real model row loaded — gates stay dark rather than a fake blend.'
      : `Map live congestion to demand ≈ ${Math.round(demand)} veh/h and select ${hourText}.`,
    source === 'unavailable'
      ? 'Forecast withheld.'
      : `Forget ${(gates.f * 100).toFixed(0)}% / input ${(gates.i * 100).toFixed(0)}% / output ${(gates.o * 100).toFixed(0)}% from that Keras-weight replay.`,
    `Forecast demand ≈ ${forecastText} veh/h (${source === 'sidecar' ? 'live sidecar forward pass' : source === 'lookup' ? 'replayed Keras output for the closest real hour' : 'unavailable'}).`,
    heuristicStep,
  ]

  const why = userOverride
    ? `An operator overrode the next heuristic axis to ${axisLabel}. That is a correction in this log, not training — the LSTM did not learn from it.`
    : source === 'unavailable'
      ? 'The live panel has no real model artifact, so it will not invent a quiet/rush blend. Cartoon lights are not a closed-loop controller.'
      : regime === 'jam'
        ? `Replay is a high-demand Metro Interstate hour. The forecast is real; the green stretch is a simulator heuristic driven by that number — the LSTM did not choose the light.`
        : regime === 'building'
          ? `Demand maps onto a mid-range real hour. Sequential state is inspectable here; this is still a simulation, not city signals.`
          : `Replay is a quiet-night Metro Interstate hour. Short balanced greens are a heuristic, not an LSTM control output.`

  return {
    intensity,
    regime,
    forecastVehicles,
    recommendedAxis,
    recommendedGreen,
    congested: approach,
    congestedQueue: queue,
    gates,
    neurons,
    topNeurons,
    dense,
    steps,
    why,
    source,
    sourceNote,
    matchedHour,
    userOverride,
  }
}
