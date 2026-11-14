import { describe, expect, it } from 'vitest'
import { applyReviewedActionToState } from './reviewedAction'
import { createInitialState, stepSimulation, type SimState } from './simulation'

const noArrivals = () => 1

describe('human-reviewed simulation requests', () => {
  it.each(['ns', 'ew'] as const)('applies a reviewed %s request after a fixed-cycle demonstration', action => {
    const fixed: SimState = {
      ...createInitialState(false, 'scene'),
      phase: 'all-red',
      phaseTimer: 0.01,
      operatorHold: true,
    }
    const requested = applyReviewedActionToState(fixed, action)
    expect(requested.adaptive).toBe(true)
    expect(requested.controllerOverride).toBe(action)
    expect(requested.operatorHold).toBe(false)
    expect(requested.phase).toBe('all-red')
    expect(fixed.adaptive).toBe(false)
    expect(fixed.controllerOverride).toBeNull()

    const applied = stepSimulation(requested, 0.05, noArrivals)
    expect(applied.phase).toBe(`${action}-green`)
    expect(applied.decisions[0]).toMatchObject({ action, automatic: true })
    expect(applied.decisions[0].reason).toContain('Operator requested')
    expect(applied.controllerOverride).toBeNull()
  })

  it('clears pending requests when holding, while preserving the yellow transition', () => {
    const active: SimState = {
      ...createInitialState(true, 'scene'),
      phase: 'ns-green',
      phaseTimer: 4,
      controllerOverride: 'ew',
      crosswalkRequest: { ns: true, ew: true },
    }
    const held = applyReviewedActionToState(active, 'hold')
    expect(held.operatorHold).toBe(true)
    expect(held.controllerOverride).toBeNull()
    expect(held.crosswalkRequest).toEqual({ ns: false, ew: false })
    expect(held.phase).toBe('ns-green')
    const ending = stepSimulation(held, 0.05, noArrivals)
    expect(ending.phase).toBe('ns-yellow')
    expect(ending.phaseTimer).toBe(1.4)
    expect(active.crosswalkRequest).toEqual({ ns: true, ew: true })
  })

  it('releases a reviewed hold and queues walk through the normal crossing request', () => {
    const held: SimState = {
      ...createInitialState(false, 'scene'),
      phase: 'all-red',
      phaseTimer: 0.01,
      operatorHold: true,
    }
    const requested = applyReviewedActionToState(held, 'walk')
    expect(requested.operatorHold).toBe(false)
    expect(requested.crosswalkRequest.ns).toBe(true)
    expect(requested.adaptive).toBe(false)
    expect(requested.phase).toBe('all-red')
    const applied = stepSimulation(requested, 0.05, noArrivals)
    expect(applied.phase).toBe('pedestrian-crossing')
    expect(applied.decisions[0].action).toBe('walk')
  })

  it('does not let a reviewed direction bypass unavailable camera input', () => {
    const state = createInitialState(false, 'camera')
    let requested = applyReviewedActionToState(state, 'ew')
    for (let index = 0; index < 30; index++) {
      requested = stepSimulation(requested, 0.1, noArrivals)
    }
    expect(requested.phase).toBe('all-red')
    expect(requested.controllerOverride).toBe('ew')
    expect(requested.decisions).toEqual([])
  })
})
