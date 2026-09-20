import { describe, expect, it } from 'vitest'
import {
  createInitialState,
  lightColor,
  stepSimulation,
  type SimState,
  type Vehicle,
} from './simulation'

function forcePhase(state: SimState, phase: SimState['phase'], timer = 10): SimState {
  return { ...state, phase, phaseTimer: timer }
}

function makeVehicle(overrides: Partial<Vehicle>): Vehicle {
  return {
    id: Math.floor(Math.random() * 1e9),
    kind: 'car',
    approach: 'east',
    lane: 'straight',
    intent: 'straight',
    progress: 0.5,
    speed: 0.05,
    waiting: false,
    crossed: false,
    color: '#3498db',
    ...overrides,
  }
}

describe('traffic light rules', () => {
  it('only opposite approaches are green together', () => {
    // NS green: north and south straight green, east/west red
    expect(lightColor('ns-green', 'ns')).toBe('green')
    expect(lightColor('ns-green', 'ew')).toBe('red')
    // EW green: east and west straight green, north/south red
    expect(lightColor('ew-green', 'ew')).toBe('green')
    expect(lightColor('ew-green', 'ns')).toBe('red')
    // Left phases: only that axis left lane green, everything else red
    expect(lightColor('ns-left-green', 'ns', 'left')).toBe('green')
    expect(lightColor('ns-left-green', 'ns', 'straight')).toBe('red')
    expect(lightColor('ns-left-green', 'ew')).toBe('red')
    expect(lightColor('ns-left-green', 'ew', 'left')).toBe('red')
    // Pedestrian crossing: all vehicles red
    expect(lightColor('pedestrian-crossing', 'ns')).toBe('red')
    expect(lightColor('pedestrian-crossing', 'ew')).toBe('red')
  })

  it('vehicles stop at a red light and never cross the stop line', () => {
    let state = createInitialState(false)
    state = forcePhase(state, 'ns-green', 30) // east/west red
    const car = makeVehicle({ approach: 'east', progress: 0.8, speed: 0.05 })
    state = { ...state, vehicles: [car] }

    for (let i = 0; i < 200; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99) // no random spawns
      const v = state.vehicles.find((x) => x.id === car.id)
      if (!v) break
      // East vehicle on red must never pass the stop line
      expect(v.progress).toBeLessThan(1.0)
      expect(v.crossed).toBe(false)
    }
  })

  it('vehicles proceed on green and clear the intersection', () => {
    let state = createInitialState(false)
    state = forcePhase(state, 'ew-green', 30)
    const car = makeVehicle({ approach: 'west', progress: 0.9, speed: 0.06 })
    state = { ...state, vehicles: [car] }

    let crossed = false
    for (let i = 0; i < 200; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const v = state.vehicles.find((x) => x.id === car.id)
      if (v?.crossed) {
        crossed = true
        break
      }
    }
    expect(crossed).toBe(true)
  })

  it('cars queue behind each other instead of overlapping', () => {
    let state = createInitialState(false)
    state = forcePhase(state, 'ns-green', 30) // east red
    const front = makeVehicle({ id: 1, approach: 'east', progress: 0.97, speed: 0.05 })
    const behind = makeVehicle({ id: 2, approach: 'east', progress: 0.8, speed: 0.05 })
    state = { ...state, vehicles: [front, behind] }

    for (let i = 0; i < 120; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const a = state.vehicles.find((x) => x.id === 1)
      const b = state.vehicles.find((x) => x.id === 2)
      if (a && b) {
        // The following car must stay behind with a gap
        expect(b.progress).toBeLessThanOrEqual(a.progress - 0.05)
      }
    }
  })

  it('left-turning vehicles wait for the left-turn phase', () => {
    let state = createInitialState(false)
    state = forcePhase(state, 'ns-green', 30)
    const leftCar = makeVehicle({ approach: 'east', intent: 'left', lane: 'left', progress: 0.9 })
    state = { ...state, vehicles: [leftCar] }

    for (let i = 0; i < 100; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const v = state.vehicles.find((x) => x.id === leftCar.id)
      if (!v) break
      expect(v.progress).toBeLessThan(1.0)
    }
  })
})
