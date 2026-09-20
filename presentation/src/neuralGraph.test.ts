import { describe, expect, it } from 'vitest'
import type { DecisionThought } from './decisionBrain'
import { buildGraph } from './NeuralGraph'

function thought(overrides: Partial<DecisionThought>): DecisionThought {
  return {
    intensity: 0.2,
    regime: 'calm',
    forecastVehicles: 800,
    recommendedAxis: 'ew',
    recommendedGreen: 6,
    congested: 'east',
    congestedQueue: 2,
    gates: { i: 0.4, f: 0.5, g: 0.1, o: 0.4 },
    neurons: Array.from({ length: 32 }, () => 0.01),
    topNeurons: [],
    dense: Array.from({ length: 16 }, () => 0),
    steps: [],
    why: '',
    ...overrides,
  }
}

describe('neural graph lighting', () => {
  it('lights every layer and later-hop edges under jam even with sparse traces', () => {
    const { nodes, edges } = buildGraph(
      thought({
        intensity: 0.92,
        regime: 'jam',
        forecastVehicles: 6100,
        congestedQueue: 18,
        gates: { i: 0.56, f: 0.63, g: 0.02, o: 0.58 },
        neurons: Array.from({ length: 32 }, () => 0.02),
        dense: Array.from({ length: 16 }, () => 0),
      }),
    )

    for (let layer = 0; layer <= 4; layer += 1) {
      const layerNodes = nodes.filter((n) => n.layer === layer)
      const lit = layerNodes.filter((n) => n.activation > 0.3)
      expect(lit.length, `layer ${layer} should have visible activations`).toBeGreaterThan(0)
      if (layer === 2 || layer === 3) {
        expect(lit.length / layerNodes.length).toBeGreaterThan(0.5)
      }
    }

    const forecast = nodes.find((n) => n.layer === 4)
    expect(forecast).toBeDefined()
    expect(forecast!.activation).toBeGreaterThan(0.6)
    expect(forecast!.label).toBe('6100')

    for (let hop = 0; hop < 4; hop += 1) {
      const hopEdges = edges.filter((e) => e.from.layer === hop)
      const pulsing = hopEdges.filter((e) => e.weight > 0.25)
      expect(pulsing.length, `hop ${hop}→${hop + 1} needs traveling pulses`).toBeGreaterThan(0)
    }
  })

  it('keeps a residual glow on a few nodes per layer under calm', () => {
    const { nodes } = buildGraph(thought({ intensity: 0.08, regime: 'calm', congestedQueue: 1 }))
    for (let layer = 0; layer <= 4; layer += 1) {
      const layerNodes = nodes.filter((n) => n.layer === layer)
      const residual = layerNodes.filter((n) => n.activation > 0.08)
      expect(residual.length, `calm layer ${layer} should not be dead`).toBeGreaterThan(0)
      expect(Math.max(...layerNodes.map((n) => n.activation))).toBeLessThan(0.7)
    }
  })
})
