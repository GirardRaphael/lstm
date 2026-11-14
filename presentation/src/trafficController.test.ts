import { describe, expect, it } from 'vitest'
import { decideTraffic, observeTraffic } from './trafficController'
import { createInitialState, stepSimulation, type Approach, type SimState, type Vehicle } from './simulation'

function car(approach: Approach, overrides: Partial<Vehicle> = {}): Vehicle {
  return { id: 1, approach, kind: 'car', lane: 'straight', intent: 'straight', progress: 0.8,
    speed: 0.2, waiting: true, crossed: false, color: '#fff', ...overrides }
}
function scene(ns: number, ew: number): SimState {
  return { ...createInitialState(), phase: 'all-red', phaseTimer: 0.01,
    vehicles: [ ...Array.from({ length: ns }, (_, i) => car('north', { id: i + 1, progress: 0.9 - i * 0.11 })),
      ...Array.from({ length: ew }, (_, i) => car('east', { id: i + 100, progress: 0.9 - i * 0.11 })) ] }
}
const noArrivals = () => 0.99

describe('scene-based signal controller', () => {
  it('actually changes the next green when traffic moves to the other direction', () => {
    expect(stepSimulation(scene(7, 1), 0.05, noArrivals).phase).toBe('ns-green')
    expect(stepSimulation(scene(1, 7), 0.05, noArrivals).phase).toBe('ew-green')
  })
  it('applies the selected duration and logs the exact pre-movement observations', () => {
    const before = scene(1, 6)
    const after = stepSimulation(before, 0.05, noArrivals)
    const applied = after.decisions[0]
    expect(applied.action).toBe('ew')
    expect(after.phaseTimer).toBe(applied.seconds)
    expect(after.ewGreen).toBe(applied.seconds)
    expect(applied.observation).toEqual(observeTraffic(before))
    expect(applied.observation.queues.east).toBe(6)
  })
  it('gives more green to more visible traffic, within the limits', () => {
    expect(decideTraffic(scene(1, 7)).seconds).toBeGreaterThan(decideTraffic(scene(1, 2)).seconds)
    const crowded = scene(0, 1)
    crowded.vehicles = Array.from({ length: 100 }, (_, id) => car('east', { id }))
    expect(decideTraffic(crowded).seconds).toBe(16)
    expect(decideTraffic(scene(0, 0)).seconds).toBe(4)
  })
  it('counts trucks twice and ignores off-screen or already-crossed vehicles', () => {
    const state = scene(0, 0)
    state.vehicles = [car('west', { kind: 'truck' }), car('north', { progress: -0.2 }),
      car('south', { progress: 1.2, crossed: true })]
    expect(observeTraffic(state).queues).toEqual({ north: 0, south: 0, east: 0, west: 2 })
    expect(decideTraffic(state).action).toBe('ew')
  })
  it('does not read hidden demand, stale queue counters, or the jam flag', () => {
    const original = scene(1, 5)
    const altered = { ...original, jamActive: true, queues: { north: 999, south: 999, east: 0, west: 0 },
      demand: { north: 999, south: 999, east: 0, west: 0 } }
    expect(decideTraffic(altered)).toEqual(decideTraffic(original))
  })
  it('prioritizes a waiting small queue over the larger queue and a manual override', () => {
    const state = scene(1, 7)
    state.redWait = { ns: 30, ew: 0 }
    state.controllerOverride = 'ew'
    const after = stepSimulation(state, 0.05, noArrivals)
    expect(after.phase).toBe('ns-green')
    expect(after.decisions[0].reason).toContain('starvation')
    expect(after.controllerOverride).toBe('ew')
  })
  it('does not switch immediately when traffic changes during green', () => {
    const state = { ...scene(0, 7), phase: 'ns-green' as const, phaseTimer: 3 }
    const after = stepSimulation(state, 0.1, noArrivals)
    expect(after.phase).toBe('ns-green')
    expect(after.decisions).toHaveLength(0)
  })
  it('holds all-red until a vehicle in the box clears even without a pedestrian request', () => {
    const state = scene(0, 6)
    state.vehicles.push(car('north', { progress: 1.4, crossed: true }))
    const after = stepSimulation(state, 0.05, noArrivals)
    expect(after.phase).toBe('all-red')
    expect(after.decisions).toHaveLength(0)
    expect(stepSimulation({ ...after, phaseTimer: 0.01, vehicles: after.vehicles.filter(v => !v.crossed) }, 0.05, noArrivals).phase).toBe('ew-green')
  })
  it('holds all-red while pedestrians are still crossing', () => {
    const state = scene(6, 0)
    state.pedestrians = [{ id: 1, approach: 'east', progress: 0.4, crossing: true, waiting: false, done: false, waitTime: 0 }]
    expect(stepSimulation(state, 0.05, noArrivals).phase).toBe('all-red')
  })
  it('serves even one pedestrian after a long wait and logs the walk action', () => {
    const state = scene(7, 0)
    state.pedestrians = [{ id: 1, approach: 'east', progress: 0, crossing: false, waiting: true, done: false, waitTime: 21 }]
    const after = stepSimulation(state, 0.05, noArrivals)
    expect(after.phase).toBe('pedestrian-crossing')
    expect(after.decisions[0].action).toBe('walk')
    expect(after.pedestrians[0].crossing).toBe(true)
    expect(after.vehicles.every(v => !v.crossed)).toBe(true)
  })
  it('holds a manual request until a safe transition, then consumes it once', () => {
    const state = { ...scene(7, 1), phase: 'ns-green' as const, phaseTimer: 2, controllerOverride: 'ew' as const }
    const waiting = stepSimulation(state, 0.1, noArrivals)
    expect(waiting.controllerOverride).toBe('ew')
    expect(waiting.phase).toBe('ns-green')
    const after = stepSimulation({ ...waiting, phase: 'all-red', phaseTimer: 0.01 }, 0.05, noArrivals)
    expect(after.phase).toBe('ew-green')
    expect(after.controllerOverride).toBeNull()
  })
  it('waits for straight traffic to clear before releasing a single left-turning vehicle', () => {
    const state = { ...scene(0, 0), phase: 'ns-yellow' as const, phaseTimer: 0.01,
      vehicles: [car('north', { lane: 'left', intent: 'left' }), car('south', { id: 2, progress: 1.5, crossed: true })] }
    const held = stepSimulation(state, 0.05, noArrivals)
    expect(held.phase).toBe('ns-yellow')
    const after = stepSimulation({ ...held, phaseTimer: 0.01, vehicles: held.vehicles.filter(v => !v.crossed) }, 0.05, noArrivals)
    expect(after.phase).toBe('ns-left-green')
    expect(after.decisions[0].movement).toBe('left')
    expect(after.decisions[0].seconds).toBe(after.phaseTimer)
  })
  it('fixed mode alternates regardless of congestion and uses six-second greens', () => {
    const after = stepSimulation({ ...scene(7, 0), adaptive: false, nextAxis: 'ew', nsGreen: 16, ewGreen: 16 }, 0.05, noArrivals)
    expect(after.phase).toBe('ew-green')
    expect(after.phaseTimer).toBe(6)
  })
  it('serves both directions under persistent asymmetric traffic', () => {
    let state = scene(1, 7)
    // Freeze approach motion to keep the imbalance present; there is no box obstruction.
    state.vehicles = state.vehicles.map(v => ({ ...v, speed: 0 }))
    for (let i = 0; i < 1500; i++) state = stepSimulation(state, 0.1, noArrivals)
    expect(state.decisions.some(d => d.action === 'ns')).toBe(true)
    expect(state.decisions.filter(d => d.action === 'ew').length).toBeGreaterThan(1)
    expect(state.decisions.length).toBeLessThanOrEqual(32)
  })
})
