import { describe, expect, it } from 'vitest'
import { applyVisionFrame, createInitialState, stepSimulation, type Phase, type SimState, type Vehicle } from './simulation'
import { cameraReady, decideTraffic, emptyObservation, observeTraffic, type TrafficObservation } from './trafficController'
import type { VisionFrame } from './screenshotVision'

const noArrivals = () => 1
function observation(ns = 0, ew = 0): TrafficObservation {
  return { ...emptyObservation(), queues: { north: ns, south: 0, east: ew, west: 0 }, cars: ns + ew }
}
function frame(capturedAt: number, measured = observation(), extra: Partial<VisionFrame> = {}): VisionFrame {
  return { valid: true, reason: 'Measured screenshot', width: 800, height: 800, capturedAt,
    observation: measured, detections: [], boxOccupied: false, crossingPedestrians: 0, nextTrackId: 1, ...extra }
}
function readyScene(measured = observation(1, 6), adaptive = true): SimState {
  let state = { ...createInitialState(adaptive, 'camera'), elapsed: 1, phaseTimer: 0.01 }
  state = applyVisionFrame(state, frame(0.7, measured))
  return applyVisionFrame(state, frame(1, measured))
}
function hiddenVehicle(overrides: Partial<Vehicle> = {}): Vehicle {
  return { id: 1, approach: 'north', kind: 'truck', lane: 'left', intent: 'left',
    progress: 0.8, speed: 0, waiting: true, crossed: false, color: '#f00', ...overrides }
}
function result(state: SimState) {
  return { phase: state.phase, phaseTimer: state.phaseTimer, redWait: state.redWait,
    decisions: state.decisions, crosswalkRequest: state.crosswalkRequest, controllerOverride: state.controllerOverride }
}

describe('screenshot-driven signal integration', () => {
  it('changes the applied green from measured pixels while scene objects stay identical', () => {
    const ns = stepSimulation(readyScene(observation(7, 1)), 0.05, noArrivals)
    const ew = stepSimulation(readyScene(observation(1, 7)), 0.05, noArrivals)
    expect(ns.phase).toBe('ns-green')
    expect(ew.phase).toBe('ew-green')
    expect(ew.decisions[0]).toMatchObject({ source: 'camera', frameCapturedAt: 1, action: 'ew' })
    expect(ew.decisions[0].observation).toEqual(observation(1, 7))
    expect(ew.phaseTimer).toBe(ew.decisions[0].seconds)
  })
  it('ignores hidden vehicles, pedestrian timers, stored queues and arrival demand for control', () => {
    const base = readyScene()
    const altered: SimState = { ...base,
      vehicles: [hiddenVehicle(), hiddenVehicle({ id: 2, progress: 1.5, crossed: true })],
      pedestrians: [{ id: 1, approach: 'east', progress: 0.4, crossing: true, waiting: false, done: false, waitTime: 999 }],
      queues: { north: 999, south: 999, east: 0, west: 0 },
      leftQueues: { north: 999, south: 999, east: 0, west: 0 },
      demand: { north: 999, south: 999, east: 0, west: 0 }, jamActive: true }
    expect(observeTraffic(altered)).toEqual(observeTraffic(base))
    expect(decideTraffic(altered)).toEqual(decideTraffic(base))
    expect(result(stepSimulation(altered, 0.05, noArrivals)))
      .toEqual(result(stepSimulation(base, 0.05, noArrivals)))
  })
  it('accumulates waiting time from screenshot queues instead of hidden scene queues', () => {
    const base = { ...readyScene(observation(4, 0)), phase: 'ew-green' as const,
      phaseTimer: 3, cameraRecoveryRequired: false, redWait: { ns: 12, ew: 8 } }
    const next = stepSimulation({ ...base, vehicles: [hiddenVehicle({ approach: 'east' })] }, 0.1, noArrivals)
    expect(next.redWait).toEqual({ ns: 12.1, ew: 0 })
  })
  it('uses screenshot left lanes in fixed cycle without stored-left-queue fallback', () => {
    const measured = observation(1, 0)
    measured.leftQueues.north = 1
    const base = { ...readyScene(measured, false), phase: 'ns-yellow' as const,
      phaseTimer: 0.01, cameraRecoveryRequired: false }
    expect(stepSimulation(base, 0.05, noArrivals).phase).toBe('ns-left-green')
    const noLeft = { ...readyScene(observation(1, 0), false), phase: 'ns-yellow' as const,
      phaseTimer: 0.01, cameraRecoveryRequired: false,
      vehicles: [hiddenVehicle()], leftQueues: { north: 200, south: 200, east: 0, west: 0 } }
    expect(stepSimulation(noLeft, 0.05, noArrivals).phase).toBe('all-red')
  })
  it('does not auto-press the pedestrian button from invisible pedestrian timers', () => {
    const state: SimState = { ...readyScene(), phase: 'ew-green', phaseTimer: 3, tick: 30,
      cameraRecoveryRequired: false, pedestrians: Array.from({ length: 20 }, (_, id) => ({
        id, approach: 'north', progress: 0, crossing: false, waiting: true, done: false, waitTime: 999,
      })) }
    const next = stepSimulation(state, 0.1, noArrivals)
    expect(next.crosswalkRequest).toEqual({ ns: false, ew: false })
    expect(decideTraffic({ ...next, phase: 'all-red' }).action).toBe('ew')
  })
  it('serves pedestrians from measured waiting observations without scene pedestrians', () => {
    const measured = { ...observation(5, 0), pedestrians: 1, oldestPedestrian: 21 }
    const next = stepSimulation(readyScene(measured), 0.05, noArrivals)
    expect(next.phase).toBe('pedestrian-crossing')
    expect(next.decisions[0]).toMatchObject({ action: 'walk', source: 'camera' })
  })
  it.each(['ns-green', 'ew-green', 'ns-left-green', 'ew-left-green'] as const)(
    'camera loss from %s completes yellow before holding red', phase => {
      let state: SimState = { ...readyScene(), phase, phaseTimer: 9, vision: null, cameraRecoveryRequired: false }
      state = stepSimulation(state, 0.05, noArrivals)
      const yellow = phase.replace('-green', '-yellow')
      expect(state.phase).toBe(yellow)
      expect(state.phaseTimer).toBe(1.4)
      for (let i = 0; i < 13; i++) state = stepSimulation(state, 0.1, noArrivals)
      expect(state.phase).toBe(yellow)
      state = stepSimulation(state, 0.11, noArrivals)
      expect(state.phase).toBe('all-red')
      for (let i = 0; i < 30; i++) state = stepSimulation(state, 0.1, noArrivals)
      expect(state.phase).toBe('all-red')
      expect(state.decisions).toHaveLength(0)
    },
  )
  it.each(['all-red', 'ns-yellow', 'ew-yellow', 'ns-left-yellow', 'ew-left-yellow', 'pedestrian-crossing'] as Phase[])(
    'invalid screenshots cannot release traffic from %s or be bypassed by requests', phase => {
      const state: SimState = { ...readyScene(), phase, phaseTimer: 0.01, controllerOverride: 'ns',
        crosswalkRequest: { ns: true, ew: true }, cameraRecoveryRequired: false,
        vision: frame(1, observation(100, 0), { valid: false, reason: 'Camera unreadable' }) }
      const next = stepSimulation(state, 0.05, noArrivals)
      expect(next.phase).toBe('all-red')
      expect(next.decisions).toHaveLength(0)
      expect(next.controllerOverride).toBe('ns')
    },
  )
  it('requires two fresh clear captures and the all-red interval before releasing green', () => {
    let state = { ...createInitialState(true, 'camera'), elapsed: 1, phaseTimer: 0.01 }
    state = applyVisionFrame(state, frame(1, observation(1, 5)))
    state = stepSimulation(state, 0.05, noArrivals)
    expect(state.visionClearFrames).toBe(1)
    expect(state.phase).toBe('all-red')
    state = applyVisionFrame(state, frame(state.elapsed, observation(1, 5)))
    expect(state.visionClearFrames).toBe(2)
    for (let i = 0; i < 10; i++) state = stepSimulation(state, 0.05, noArrivals)
    expect(state.phase).toBe('all-red')
    for (let i = 0; i < 7; i++) state = stepSimulation(state, 0.05, noArrivals)
    expect(state.phase).toBe('ew-green')
  })
  it.each([{ boxOccupied: true }, { crossingPedestrians: 1 }])(
    'detected occupancy holds the intersection with no scene traffic: %j', occupancy => {
      let state = readyScene()
      state = applyVisionFrame({ ...state, elapsed: 1.1 }, frame(1.1, observation(4, 0), occupancy))
      expect(stepSimulation(state, 0.05, noArrivals).phase).toBe('all-red')
      expect(state.visionClearFrames).toBe(0)
    },
  )
  it('rejects duplicate, out-of-order, future and nonfinite capture timestamps', () => {
    const state = readyScene()
    for (const timestamp of [1, 0.9, 1.1, -1, NaN, Infinity]) {
      expect(applyVisionFrame(state, frame(timestamp, observation(100, 0)))).toBe(state)
    }
    const empty = { ...createInitialState(true, 'camera'), elapsed: 3 }
    expect(applyVisionFrame(empty, frame(1.9))).toBe(empty)
  })
  it('holds on stale input and never reads scene data as fallback', () => {
    const state: SimState = { ...readyScene(), elapsed: 2.1, controllerOverride: 'ew',
      vehicles: [hiddenVehicle()], crosswalkRequest: { ns: true, ew: false } }
    expect(cameraReady(state)).toBe(false)
    expect(observeTraffic(state)).toEqual(emptyObservation())
    const next = stepSimulation(state, 0.05, noArrivals)
    expect(next.phase).toBe('all-red')
    expect(next.controllerOverride).toBe('ew')
    expect(next.decisions).toHaveLength(0)
  })
  it('recovery during yellow cannot shortcut yellow or the following all-red interval', () => {
    let state: SimState = { ...readyScene(), phase: 'ns-green', phaseTimer: 8, vision: null, cameraRecoveryRequired: false }
    state = stepSimulation(state, 0.05, noArrivals)
    for (let i = 0; i < 5; i++) {
      state = applyVisionFrame(state, frame(state.elapsed, observation(0, 5)))
      state = stepSimulation(state, 0.1, noArrivals)
    }
    expect(state.phase).toBe('ns-yellow')
    for (let i = 0; i < 10; i++) {
      state = applyVisionFrame(state, frame(state.elapsed, observation(0, 5)))
      state = stepSimulation(state, 0.1, noArrivals)
    }
    expect(state.phase).toBe('all-red')
    expect(state.decisions).toHaveLength(0)
    for (let i = 0; i < 10; i++) {
      state = applyVisionFrame(state, frame(state.elapsed, observation(0, 5)))
      state = stepSimulation(state, 0.1, noArrivals)
    }
    expect(state.phase).toBe('ew-green')
  })
})
