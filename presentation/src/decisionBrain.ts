import type { Approach, SimState } from './simulation'

export type GateKey = 'i' | 'f' | 'g' | 'o'

export type NeuronTraceWindow = {
  label: string
  actual_vehicles: number
  predicted_vehicles: number
  max_abs_error_vs_keras?: number
  layers: Array<{
    name: string
    units: number
    gates_final: Record<GateKey, number>
    gate_series: Record<GateKey, number[]>
    hidden_final: number[]
    top_neurons: Array<{
      unit: number
      mean_abs: number
      final: number
      role: string
      mean_forget: number
    }>
  }>
  dense_hidden: number[] | null
  gate_names: Record<string, string>
}

export type NeuronTraceFile = {
  model: string
  claim: string
  architecture: Array<{ id: string; label: string; units: number }>
  windows: {
    rush: NeuronTraceWindow
    quiet: NeuronTraceWindow
  }
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
  topNeurons: NeuronTraceWindow['layers'][0]['top_neurons']
  dense: number[]
  steps: string[]
  why: string
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function lerpArr(a: number[], b: number[], t: number): number[] {
  const n = Math.min(a.length, b.length)
  return Array.from({ length: n }, (_, i) => lerp(a[i], b[i], t))
}

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

export function thinkDecision(state: SimState, traces: NeuronTraceFile | null): DecisionThought {
  const intensity = congestionIntensity(state)
  const { approach, queue } = heaviestApproach(state)
  const recommendedAxis: 'ns' | 'ew' =
    approach === 'north' || approach === 'south' ? 'ns' : 'ew'
  const recommendedGreen = state.adaptive
    ? Math.min(14, Math.max(3.5, 3.5 + queue * 0.55))
    : recommendedAxis === 'ns'
      ? state.nsGreen
      : state.ewGreen

  const quiet = traces?.windows.quiet
  const rush = traces?.windows.rush
  const t = intensity

  let gates: Record<GateKey, number> = { i: 0.4, f: 0.5, g: 0.1, o: 0.4 }
  let neurons: number[] = Array.from({ length: 32 }, (_, i) => Math.sin(i + intensity * 4) * 0.2)
  let topNeurons: DecisionThought['topNeurons'] = []
  let dense: number[] = Array.from({ length: 16 }, (_, i) => Math.max(0, intensity - i * 0.03))
  let forecastVehicles = 800 + intensity * 5000

  if (quiet && rush) {
    const qL = quiet.layers[0]
    const rL = rush.layers[0]
    gates = {
      i: lerp(qL.gates_final.i, rL.gates_final.i, t),
      f: lerp(qL.gates_final.f, rL.gates_final.f, t),
      g: lerp(qL.gates_final.g, rL.gates_final.g, t),
      o: lerp(qL.gates_final.o, rL.gates_final.o, t),
    }
    // Prefer second LSTM hidden (32) for compact neuron grid
    const qH = quiet.layers[1]?.hidden_final ?? quiet.layers[0].hidden_final
    const rH = rush.layers[1]?.hidden_final ?? rush.layers[0].hidden_final
    neurons = lerpArr(qH, rH, t)
    const qTop = quiet.layers[1]?.top_neurons ?? quiet.layers[0].top_neurons
    const rTop = rush.layers[1]?.top_neurons ?? rush.layers[0].top_neurons
    topNeurons = (t > 0.45 ? rTop : qTop).slice(0, 6)
    if (quiet.dense_hidden && rush.dense_hidden) {
      dense = lerpArr(quiet.dense_hidden, rush.dense_hidden, t)
    }
    forecastVehicles = lerp(quiet.predicted_vehicles, rush.predicted_vehicles, t)
  }

  const regime: DecisionThought['regime'] =
    intensity < 0.28 ? 'calm' : intensity < 0.62 ? 'building' : 'jam'

  const axisLabel = recommendedAxis === 'ns' ? 'North–South' : 'East–West'
  const steps = [
    `Sense ${queue} PCU waiting on ${approach} (live intersection).`,
    `Forget gate ${(gates.f * 100).toFixed(0)}% — ${
      gates.f > 0.55 ? 'keep recent demand memory' : 'discard stale calm pattern'
    }.`,
    `Input gate ${(gates.i * 100).toFixed(0)}% — ${
      gates.i > 0.45 ? 'write the jam evidence into cell state' : 'little new information'
    }.`,
    `Output gate ${(gates.o * 100).toFixed(0)}% — expose hidden state to the decision head.`,
    `Forecast demand ≈ ${Math.round(forecastVehicles)} vehicles/h from the trained LSTM blend.`,
    state.adaptive
      ? `Decision: stretch ${axisLabel} green toward ${recommendedGreen.toFixed(1)}s.`
      : `Decision: fixed timing held (adaptive OFF) — green stays ${recommendedGreen.toFixed(1)}s.`,
  ]

  const why =
    regime === 'jam'
      ? `Rush-like neuron pattern is active. Top units are firing hard, so the controller borrows green time for ${approach}.`
      : regime === 'building'
        ? `Demand is climbing — gates are opening and more LSTM units leave the quiet regime.`
        : `Quiet-night neuron pattern dominates — short balanced greens are enough.`

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
  }
}
