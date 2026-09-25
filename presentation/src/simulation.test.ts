import { describe, expect, it } from 'vitest'
import {
  spawnAgents,
  createInitialState,
  lightColor,
  pedestrianInCrosswalk,
  pedestrianSignal,
  pressCrosswalkButton,
  stepSimulation,
  STOP_LINE,
  vehicleOccupiesBox,
  type Pedestrian,
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

function expirePhase(state: SimState): SimState {
  return stepSimulation({ ...state, phaseTimer: 0.01 }, 0.05, () => 0.99)
}

describe('signal axis alternation', () => {
  it('alternates NS → EW → NS instead of locking onto one axis', () => {
    let state = createInitialState(false)
    expect(state.phase).toBe('ns-green')
    expect(state.nextAxis).toBe('ew')

    state = expirePhase(state)
    expect(state.phase).toBe('ns-yellow')
    expect(state.nextAxis).toBe('ew')

    state = expirePhase(state)
    expect(state.phase).toBe('all-red')
    expect(state.nextAxis).toBe('ew')

    state = expirePhase(state)
    expect(state.phase).toBe('ew-green')
    expect(state.nextAxis).toBe('ns')

    state = expirePhase(state)
    expect(state.phase).toBe('ew-yellow')
    expect(state.nextAxis).toBe('ns')

    state = expirePhase(state)
    expect(state.phase).toBe('all-red')
    expect(state.nextAxis).toBe('ns')

    state = expirePhase(state)
    expect(state.phase).toBe('ns-green')
    expect(state.nextAxis).toBe('ew')
  })

  it('resumes on the due axis after a pedestrian phase that interrupted NS', () => {
    let state = createInitialState(false)
    state = pressCrosswalkButton(state, 'ew')
    state = { ...state, phase: 'ns-yellow', phaseTimer: 0.01, nextAxis: 'ew' }
    state = stepSimulation(state, 0.05, () => 0.99)
    expect(state.phase).toBe('all-red')
    expect(state.nextAxis).toBe('ew')

    state = expirePhase(state)
    expect(state.phase).toBe('pedestrian-crossing')
    expect(state.nextAxis).toBe('ew')

    state = expirePhase(state)
    expect(state.phase).toBe('ew-green')
    expect(state.nextAxis).toBe('ns')
  })

  it('resumes on the due axis after a pedestrian phase that interrupted EW', () => {
    let state = createInitialState(false)
    state = pressCrosswalkButton(state, 'ns')
    state = { ...state, phase: 'ew-yellow', phaseTimer: 0.01, nextAxis: 'ns' }
    state = stepSimulation(state, 0.05, () => 0.99)
    expect(state.phase).toBe('all-red')
    expect(state.nextAxis).toBe('ns')

    state = expirePhase(state)
    expect(state.phase).toBe('pedestrian-crossing')
    expect(state.nextAxis).toBe('ns')

    state = expirePhase(state)
    expect(state.phase).toBe('ns-green')
    expect(state.nextAxis).toBe('ew')
  })
})

function makePedestrian(overrides: Partial<Pedestrian>): Pedestrian {
  return {
    id: 1,
    approach: 'north',
    progress: 0,
    crossing: false,
    waiting: true,
    done: false,
    waitTime: 0,
    ...overrides,
  }
}

describe('exclusive pedestrian crossing', () => {
  it('does not set crossing=true during ns-green or ew-green', () => {
    const peds = [
      makePedestrian({ id: 1, approach: 'north' }),
      makePedestrian({ id: 2, approach: 'west' }),
      makePedestrian({ id: 3, approach: 'south' }),
      makePedestrian({ id: 4, approach: 'east' }),
    ]

    for (const phase of ['ns-green', 'ew-green'] as const) {
      let state = createInitialState(false)
      state = forcePhase(state, phase, 30)
      state = { ...state, pedestrians: peds.map((p) => ({ ...p })), pedestrianCrossing: false }
      state = stepSimulation(state, 0.05, () => 0.99)
      expect(
        state.pedestrians.every((p) => p.crossing === false && p.waiting === true),
        `${phase} must keep pedestrians waiting`,
      ).toBe(true)
    }
  })

  it('starts waiting pedestrians crossing during pedestrian-crossing and holds vehicles at the stop line', () => {
    let state = createInitialState(false)
    const car = makeVehicle({ approach: 'north', progress: 0.9, speed: 0.2 })
    state = forcePhase(state, 'pedestrian-crossing', 30)
    state = {
      ...state,
      pedestrianCrossing: true,
      vehicles: [car],
      pedestrians: [
        makePedestrian({ id: 1, approach: 'north' }),
        makePedestrian({ id: 2, approach: 'west' }),
      ],
    }

    for (let i = 0; i < 40; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const v = state.vehicles.find((x) => x.id === car.id)
      expect(v).toBeDefined()
      expect(v!.progress).toBeLessThanOrEqual(STOP_LINE)
      expect(v!.crossed).toBe(false)
    }
    expect(state.pedestrians.length).toBeGreaterThan(0)
    expect(state.pedestrians.every((p) => p.crossing === true)).toBe(true)
    expect(state.pedestrians.every((p) => p.progress > 0)).toBe(true)
  })

  it('does not auto-press the crosswalk on tick 0', () => {
    let state = createInitialState(false)
    state = {
      ...state,
      tick: 0,
      pedestrians: [makePedestrian({ id: 1, approach: 'north', waitTime: 10 })],
    }
    state = stepSimulation(state, 0.05, () => 0.99)
    expect(state.crosswalkRequest.ns).toBe(false)
    expect(state.crosswalkRequest.ew).toBe(false)
  })

  it('holds all-red until a vehicle in the box clears, then starts walk', () => {
    let state = createInitialState(false)
    const car = makeVehicle({
      id: 42,
      approach: 'west',
      progress: 1.35,
      speed: 0.25,
      crossed: true,
    })
    state = pressCrosswalkButton(state, 'ew')
    state = {
      ...state,
      phase: 'all-red',
      phaseTimer: 0.01,
      vehicles: [car],
      pedestrians: [makePedestrian({ id: 1, approach: 'north' })],
    }

    let sawWalk = false
    for (let i = 0; i < 80; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const inBox = state.vehicles.some(vehicleOccupiesBox)
      const walking = state.pedestrians.some(pedestrianInCrosswalk)
      expect(inBox && walking, `tick ${i}: walk overlapped a vehicle in the box`).toBe(false)
      if (state.phase === 'pedestrian-crossing') {
        sawWalk = true
        expect(inBox).toBe(false)
        break
      }
    }
    expect(sawWalk).toBe(true)
    expect(state.pedestrians.some((p) => p.crossing)).toBe(true)
  })

  it('never overlaps a walking pedestrian with a vehicle in the box across a full cycle', () => {
    let state = createInitialState(false)
    state = pressCrosswalkButton(state, 'ns')
    state = {
      ...state,
      phase: 'ew-green',
      phaseTimer: 0.2,
      nextAxis: 'ns',
      vehicles: [
        makeVehicle({ id: 1, approach: 'west', progress: 0.92, speed: 0.2 }),
        makeVehicle({ id: 2, approach: 'east', progress: 1.2, speed: 0.2, crossed: true }),
      ],
      pedestrians: [
        makePedestrian({ id: 10, approach: 'north' }),
        makePedestrian({ id: 11, approach: 'west' }),
      ],
    }

    let sawWalk = false
    let sawWaiting = false
    for (let i = 0; i < 250; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const inBox = state.vehicles.some(vehicleOccupiesBox)
      const walking = state.pedestrians.some(pedestrianInCrosswalk)
      if (state.pedestrians.some((p) => p.waiting)) sawWaiting = true
      if (walking) sawWalk = true
      expect(inBox && walking, `tick ${i} phase ${state.phase}`).toBe(false)
      if (state.phase === 'pedestrian-crossing') {
        expect(inBox).toBe(false)
        expect(pedestrianSignal(state.phase, 'ns')).toBe('walk')
      } else {
        expect(pedestrianSignal(state.phase, 'ns')).toBe('dont-walk')
        expect(pedestrianSignal(state.phase, 'ew')).toBe('dont-walk')
      }
    }
    expect(sawWaiting).toBe(true)
    expect(sawWalk).toBe(true)
  })

  it('does not start walking during yellow or protected lefts', () => {
    for (const phase of ['ns-yellow', 'ew-yellow', 'ns-left-green', 'ew-left-green'] as const) {
      let state = createInitialState(false)
      state = forcePhase(state, phase, 8)
      state = {
        ...state,
        pedestrians: [makePedestrian({ id: 1, approach: 'north' }), makePedestrian({ id: 2, approach: 'west' })],
        pedestrianCrossing: false,
      }
      state = stepSimulation(state, 0.05, () => 0.99)
      expect(state.pedestrians.every((p) => p.waiting && !p.crossing), phase).toBe(true)
    }
  })

  it('holds a spawned crowd at the curb for several seconds so the cluster is visible', () => {
    let state = spawnAgents(createInitialState(false), { cars: 0, trucks: 0, pedestrians: 16 }, () => 0.5)
    expect(state.pedestrians.filter((p) => p.waiting).length).toBe(16)
    for (let i = 0; i < 80; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      expect(state.phase).not.toBe('pedestrian-crossing')
      expect(state.pedestrians.filter((p) => p.waiting).length).toBeGreaterThanOrEqual(16)
    }
  })

  it('releases every waiting pedestrian onto the zebra on the same walk tick', () => {
    const approaches = ['north', 'south', 'east', 'west'] as const
    const crowd = Array.from({ length: 12 }, (_, i) =>
      makePedestrian({ id: i + 1, approach: approaches[i % 4], waitTime: 9 }),
    )
    let state = createInitialState(false)
    state = {
      ...state,
      phase: 'pedestrian-crossing',
      phaseTimer: 6,
      pedestrianCrossing: true,
      pedestrians: crowd,
    }

    state = stepSimulation(state, 0.05, () => 0.99)
    const cohort = state.pedestrians.filter((p) => p.id >= 1 && p.id <= 12)
    expect(cohort).toHaveLength(12)
    expect(cohort.every((p) => p.crossing && !p.waiting)).toBe(true)
    const progresses = cohort.map((p) => p.progress)
    expect(Math.min(...progresses)).toBeGreaterThan(0)
    expect(Math.max(...progresses) - Math.min(...progresses)).toBeLessThan(1e-9)
  })

  it('keeps a pedestrian who arrives during walk waiting until the next walk', () => {
    let state = createInitialState(false)
    state = {
      ...state,
      phase: 'pedestrian-crossing',
      phaseTimer: 6,
      pedestrianCrossing: true,
      pedestrians: [makePedestrian({ id: 1, approach: 'north', waitTime: 8 })],
    }
    state = stepSimulation(state, 0.05, () => 0.99)
    expect(state.pedestrians.find((p) => p.id === 1)?.crossing).toBe(true)

    const latecomer = makePedestrian({ id: 99, approach: 'west', waitTime: 0 })
    state = { ...state, pedestrians: [...state.pedestrians, latecomer] }
    for (let i = 0; i < 20; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      expect(state.phase).toBe('pedestrian-crossing')
      const late = state.pedestrians.find((p) => p.id === 99)
      expect(late).toBeDefined()
      expect(late!.waiting).toBe(true)
      expect(late!.crossing).toBe(false)
      expect(late!.progress).toBe(0)
      expect(pedestrianInCrosswalk(late!)).toBe(false)
    }

    while (state.phase === 'pedestrian-crossing') {
      state = stepSimulation(state, 0.05, () => 0.99)
    }
    expect(state.pedestrians.find((p) => p.id === 99)?.waiting).toBe(true)

    state = pressCrosswalkButton(state, 'ns')
    let admittedOnNextWalk = false
    for (let i = 0; i < 400; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      if (state.phase === 'pedestrian-crossing') {
        const late = state.pedestrians.find((p) => p.id === 99)
        expect(late?.crossing).toBe(true)
        expect(late?.waiting).toBe(false)
        admittedOnNextWalk = true
        break
      }
    }
    expect(admittedOnNextWalk).toBe(true)
  })

  it('returns to a vehicle green after walk instead of locking into another walk', () => {
    const approaches = ['north', 'south', 'east', 'west'] as const
    const crowd = Array.from({ length: 16 }, (_, i) =>
      makePedestrian({ id: 100 + i, approach: approaches[i % 4], waitTime: 30 }),
    )
    const car = makeVehicle({ id: 1, approach: 'east', progress: STOP_LINE, speed: 0.25 })
    let state = createInitialState(false)
    state = pressCrosswalkButton(state, 'ns')
    state = {
      ...state,
      phase: 'all-red',
      phaseTimer: 0.01,
      nextAxis: 'ew',
      vehicles: [car],
      pedestrians: crowd,
    }

    state = stepSimulation(state, 0.05, () => 0.99)
    expect(state.phase).toBe('pedestrian-crossing')
    expect(state.pedestrians.filter((p) => p.id >= 100 && p.id < 116).every((p) => p.crossing)).toBe(
      true,
    )

    for (let i = 0; i < 8; i += 1) {
      state = {
        ...state,
        pedestrians: [
          ...state.pedestrians,
          makePedestrian({ id: 700 + i, approach: 'south', waitTime: 40 }),
        ],
      }
      state = stepSimulation(state, 0.05, () => 0.99)
      const late = state.pedestrians.find((p) => p.id === 700 + i)
      expect(late?.waiting && !late.crossing, `latecomer ${700 + i} joined the zebra`).toBe(true)
      const inBox = state.vehicles.some(vehicleOccupiesBox)
      const walking = state.pedestrians.some(pedestrianInCrosswalk)
      expect(inBox && walking).toBe(false)
    }

    const extras = Array.from({ length: 10 }, (_, i) =>
      makePedestrian({ id: 800 + i, approach: 'west', waitTime: 40 }),
    )
    state = { ...state, pedestrians: [...state.pedestrians, ...extras] }

    let greenPhase: SimState['phase'] | null = null
    for (let i = 0; i < 250; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const inBox = state.vehicles.some(vehicleOccupiesBox)
      const walking = state.pedestrians.some(pedestrianInCrosswalk)
      expect(inBox && walking, `tick ${i} phase ${state.phase}`).toBe(false)
      if (state.phase === 'ns-green' || state.phase === 'ew-green') {
        greenPhase = state.phase
        break
      }
    }
    expect(greenPhase).toBe('ew-green')

    const atGreen = state.vehicles.find((v) => v.id === 1)
    expect(atGreen).toBeDefined()
    const progressAtGreen = atGreen!.progress
    for (let i = 0; i < 12; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      expect(state.phase).toBe('ew-green')
    }
    const moving = state.vehicles.find((v) => v.id === 1)
    expect(moving).toBeDefined()
    expect(moving!.progress).toBeGreaterThan(progressAtGreen)
    expect(moving!.progress).toBeGreaterThan(STOP_LINE)

    let sawNsGreen = false
    for (let i = 0; i < 400; i += 1) {
      state = stepSimulation(state, 0.05, () => 0.99)
      const inBox = state.vehicles.some(vehicleOccupiesBox)
      const walking = state.pedestrians.some(pedestrianInCrosswalk)
      expect(inBox && walking, `post-walk tick ${i} phase ${state.phase}`).toBe(false)
      expect(state.phase).not.toBe('pedestrian-crossing')
      if (state.phase === 'ns-green') {
        sawNsGreen = true
        break
      }
    }
    expect(sawNsGreen).toBe(true)
  })

  it('still honours a manual crosswalk press during auto-press cooldown', () => {
    let state = createInitialState(false)
    state = {
      ...state,
      pedWalkCooldown: 12,
      phase: 'all-red',
      phaseTimer: 0.01,
      pedestrians: [makePedestrian({ id: 1, approach: 'north' })],
    }
    state = pressCrosswalkButton(state, 'ns')
    state = stepSimulation(state, 0.05, () => 0.99)
    expect(state.phase).toBe('pedestrian-crossing')
    expect(state.pedestrians[0]?.crossing).toBe(true)
  })
})

