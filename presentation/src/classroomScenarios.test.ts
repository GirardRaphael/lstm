import { createCanvas } from '@napi-rs/canvas'
import { describe, expect, it } from 'vitest'
import { paintIntersection } from './IntersectionCanvas'
import { createClassroomScenario, DEMO_SCENARIOS, type DemoScenarioId } from './classroomScenarios'
import { analyzeScreenshot } from './screenshotVision'
import { applyVisionFrame, stepSimulation, type SimState } from './simulation'
import { cameraReady, emptyObservation, observeTraffic } from './trafficController'

const ids = Object.keys(DEMO_SCENARIOS) as DemoScenarioId[]
const noArrivals = () => 1
function capture(state: SimState) {
  const canvas = createCanvas(800, 800)
  const context = canvas.getContext('2d')
  paintIntersection(context as unknown as CanvasRenderingContext2D, state)
  return analyzeScreenshot(context.getImageData(0, 0, 800, 800), state.elapsed, state.vision ?? undefined)
}
function runWithCamera(start: SimState, seconds: number): SimState {
  let state = start
  for (let i = 0; i < Math.ceil(seconds / 0.1); i++) {
    state = applyVisionFrame(state, capture(state))
    state = stepSimulation(state, 0.1, noArrivals)
  }
  return state
}

describe('classroom scenario starting scenes', () => {
  it.each(ids)('%s resets controls and requires new camera observations', id => {
    const { state, cameraEnabled, counts } = createClassroomScenario(id)
    expect(cameraEnabled).toBe(true)
    expect(state.controlSource).toBe('camera')
    expect(state.phase).toBe('all-red')
    expect(state.operatorHold).toBe(false)
    expect(state.controllerOverride).toBeNull()
    expect(state.vision).toBeNull()
    expect(state.visionClearFrames).toBe(0)
    expect(state.decisions).toEqual([])
    expect(cameraReady(state)).toBe(false)
    expect(observeTraffic(state)).toEqual(emptyObservation())
    expect(counts.cars + counts.trucks).toBe(state.vehicles.length)
    expect(counts.pedestrians).toBe(state.pedestrians.length)
    let unobserved = state
    for (let i = 0; i < 30; i++) unobserved = stepSimulation(unobserved, 0.1, noArrivals)
    expect(unobserved.phase).toBe('all-red')
    expect(unobserved.decisions).toEqual([])
  })
  it('reproduces scenes across reloads without carrying previous requests or traffic', () => {
    const first = createClassroomScenario('balanced')
    createClassroomScenario('pedestrians')
    createClassroomScenario('east-jam')
    expect(createClassroomScenario('balanced')).toEqual(first)
    expect(first.state.crosswalkRequest).toEqual({ ns: false, ew: false })
    expect(first.state.jamActive).toBe(false)
  })
  it('keeps fixed and adaptive initial traffic identical', () => {
    const balanced = createClassroomScenario('balanced')
    const fixed = createClassroomScenario('fixed')
    expect(fixed.state).toEqual({ ...balanced.state, adaptive: false })
    expect(fixed.counts).toEqual({ cars: 12, trucks: 4, pedestrians: 0 })
    expect(capture(fixed.state).observation).toEqual(capture(balanced.state).observation)
  })
  it('makes the east queue visibly dominant without injecting observations', () => {
    const { state } = createClassroomScenario('east-jam')
    const frame = capture(state)
    expect(frame.valid).toBe(true)
    const queue = frame.observation.queues
    expect(queue.east).toBeGreaterThan(queue.north)
    expect(queue.east).toBeGreaterThan(queue.south)
    expect(queue.east).toBeGreaterThan(queue.west)
    expect(state.demand.east).toBeGreaterThan(state.demand.north)
    expect(state.vision).toBeNull()
    expect(runWithCamera(state, 1.4).decisions[0]).toMatchObject({ action: 'ew', source: 'camera' })
  })
  it('serves a real button request and visible pedestrians after measured clearance', () => {
    const { state } = createClassroomScenario('pedestrians')
    expect(state.crosswalkRequest).toEqual({ ns: true, ew: false })
    expect(state.pedestrians).toHaveLength(12)
    expect(state.pedestrians.every(person => person.waiting && !person.crossing)).toBe(true)
    expect(capture(state).observation.pedestrians).toBeGreaterThan(0)
    const running = runWithCamera(state, 1.4)
    expect(running.phase).toBe('pedestrian-crossing')
    expect(running.decisions[0]).toMatchObject({ action: 'walk', source: 'camera' })
    expect(running.pedestrians.some(person => person.crossing)).toBe(true)
  })
  it('starts camera interruption with capture available, then holds red when capture stops', () => {
    const scenario = createClassroomScenario('camera-loss')
    expect(scenario.cameraEnabled).toBe(true)
    let state = runWithCamera(scenario.state, 1.4)
    expect(state.phase.endsWith('-green')).toBe(true)
    // Stopping captures naturally makes the last frame stale, as in the UI.
    let sawYellow = false
    for (let i = 0; i < 45; i++) {
      state = stepSimulation(state, 0.1, noArrivals)
      sawYellow ||= state.phase.endsWith('-yellow')
    }
    expect(sawYellow).toBe(true)
    expect(state.phase).toBe('all-red')
    expect(cameraReady(state)).toBe(false)
  })
})
