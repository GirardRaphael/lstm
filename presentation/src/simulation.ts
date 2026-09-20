export type VehicleKind = 'car' | 'truck'
export type Approach = 'north' | 'south' | 'east' | 'west'
export type TurnIntent = 'left' | 'straight' | 'right'
export type Phase =
  | 'ns-green'
  | 'ns-yellow'
  | 'ns-left-green'
  | 'ns-left-yellow'
  | 'ew-green'
  | 'ew-yellow'
  | 'ew-left-green'
  | 'ew-left-yellow'
  | 'all-red'
  | 'pedestrian-crossing'

export type Lane = 'left' | 'straight' | 'right'

export type Vehicle = {
  id: number
  kind: VehicleKind
  approach: Approach
  lane: Lane
  intent: TurnIntent
  /** 0 at spawn, 1 at stop line, >1 crossing */
  progress: number
  speed: number
  waiting: boolean
  crossed: boolean
  color: string
}

export type Pedestrian = {
  id: number
  approach: Approach
  progress: number
  crossing: boolean
  waiting: boolean
  done: boolean
}

export type CrosswalkRequest = {
  ns: boolean
  ew: boolean
}

export type SimState = {
  vehicles: Vehicle[]
  pedestrians: Pedestrian[]
  phase: Phase
  phaseTimer: number
  nsGreen: number
  ewGreen: number
  nsLeftGreen: number
  ewLeftGreen: number
  queues: Record<Approach, number>
  leftQueues: Record<Approach, number>
  throughput: number
  jamActive: boolean
  demand: Record<Approach, number>
  tick: number
  adaptive: boolean
  crosswalkRequest: CrosswalkRequest
  pedestrianCrossing: boolean
}

export type SpawnConfig = {
  cars: number
  trucks: number
  pedestrians: number
}

const YELLOW = 1.4
const ALL_RED = 0.8
const MIN_GREEN = 4
const MAX_GREEN = 16
const MIN_LEFT_GREEN = 3
const MAX_LEFT_GREEN = 8
const PEDESTRIAN_TIME = 5

const CAR_COLORS = ['#e74c3c', '#3498db', '#f39c12', '#ecf0f1', '#9b59b6', '#1abc9c']

let nextId = 1

export function createInitialState(adaptive = true): SimState {
  return {
    vehicles: [],
    pedestrians: [],
    phase: 'ns-green',
    phaseTimer: 6,
    nsGreen: 6,
    ewGreen: 6,
    nsLeftGreen: 4,
    ewLeftGreen: 4,
    queues: { north: 0, south: 0, east: 0, west: 0 },
    leftQueues: { north: 0, south: 0, east: 0, west: 0 },
    throughput: 0,
    jamActive: false,
    demand: { north: 0.35, south: 0.35, east: 0.35, west: 0.35 },
    tick: 0,
    adaptive,
    crosswalkRequest: { ns: false, ew: false },
    pedestrianCrossing: false,
  }
}

function randomApproach(rng: () => number): Approach {
  const r = rng()
  if (r < 0.25) return 'north'
  if (r < 0.5) return 'south'
  if (r < 0.75) return 'east'
  return 'west'
}

function randomIntent(rng: () => number): TurnIntent {
  const r = rng()
  if (r < 0.15) return 'left'
  if (r < 0.75) return 'straight'
  return 'right'
}

function laneForIntent(intent: TurnIntent): Lane {
  return intent
}

function speedFor(kind: VehicleKind, rng: () => number): number {
  if (kind === 'truck') return 0.016 + rng() * 0.008
  return 0.024 + rng() * 0.012
}

function randomColor(rng: () => number): string {
  return CAR_COLORS[Math.floor(rng() * CAR_COLORS.length)]
}

export function randomSpawnCounts(rng: () => number = Math.random): SpawnConfig {
  return {
    cars: 4 + Math.floor(rng() * 16),
    trucks: Math.floor(rng() * 6),
    pedestrians: 1 + Math.floor(rng() * 8),
  }
}

export function randomApproachChoice(rng: () => number = Math.random): Approach {
  return randomApproach(rng)
}

export function spawnAgents(
  state: SimState,
  config: SpawnConfig,
  rng: () => number = Math.random,
): SimState {
  const vehicles = [...state.vehicles]
  const pedestrians = [...state.pedestrians]

  for (let i = 0; i < config.cars + config.trucks; i += 1) {
    const kind: VehicleKind = i < config.cars ? 'car' : 'truck'
    const approach = randomApproach(rng)
    const intent = randomIntent(rng)
    const lane = laneForIntent(intent)
    // Queue behind whatever is already in this lane — never overlap.
    let spawn = -0.03 - rng() * 0.05
    for (const v of vehicles) {
      if (v.approach === approach && v.lane === lane && v.progress < 0.35) {
        spawn = Math.min(spawn, v.progress - 0.09)
      }
    }
    vehicles.push({
      id: nextId++,
      kind,
      approach,
      lane,
      intent,
      progress: spawn,
      speed: speedFor(kind, rng),
      waiting: false,
      crossed: false,
      color: randomColor(rng),
    })
  }

  for (let i = 0; i < config.pedestrians; i += 1) {
    pedestrians.push({
      id: nextId++,
      approach: randomApproach(rng),
      progress: 0,
      crossing: false,
      waiting: true,
      done: false,
    })
  }

  return { ...state, vehicles, pedestrians }
}

export function createTrafficJam(
  state: SimState,
  approach: Approach = 'east',
  rng: () => number = Math.random,
): SimState {
  const vehicles = [...state.vehicles]
  const jamCount = 12 + Math.floor(rng() * 6)
  // Pack a stopped queue backward from the stop line, respecting follow gaps.
  // Track each lane's tail position so mixed car/truck gaps never overlap.
  const laneTail: Record<Lane, number> = { left: 0.95, straight: 0.95, right: 0.95 }
  for (let i = 0; i < jamCount; i += 1) {
    const kind: VehicleKind = rng() < 0.2 ? 'truck' : 'car'
    const intent = randomIntent(rng)
    const lane = laneForIntent(intent)
    const gap = kind === 'truck' ? 0.09 : 0.075
    const progress = laneTail[lane]
    laneTail[lane] = progress - gap
    vehicles.push({
      id: nextId++,
      kind,
      approach,
      lane,
      intent,
      progress,
      speed: speedFor(kind, rng) * 0.6,
      waiting: true,
      crossed: false,
      color: randomColor(rng),
    })
  }
  const demand = { ...state.demand, [approach]: 1.5 }
  return { ...state, vehicles, jamActive: true, demand }
}

export function pressCrosswalkButton(state: SimState, axis: 'ns' | 'ew'): SimState {
  const crosswalkRequest = { ...state.crosswalkRequest, [axis]: true }
  return { ...state, crosswalkRequest }
}

function canVehicleProceed(phase: Phase, vehicle: Vehicle): boolean {
  const isNS = vehicle.approach === 'north' || vehicle.approach === 'south'

  if (phase === 'pedestrian-crossing' || phase === 'all-red') return false

  if (vehicle.intent === 'left') {
    if (isNS) return phase === 'ns-left-green'
    return phase === 'ew-left-green'
  }

  if (vehicle.intent === 'right') {
    // Right turns allowed on green or after stop on red (simplified: on green)
    if (isNS) return phase === 'ns-green' || phase === 'ns-left-green'
    return phase === 'ew-green' || phase === 'ew-left-green'
  }

  // Straight
  if (isNS) return phase === 'ns-green'
  return phase === 'ew-green'
}

function recountQueues(vehicles: Vehicle[]): {
  queues: Record<Approach, number>
  leftQueues: Record<Approach, number>
} {
  const queues: Record<Approach, number> = { north: 0, south: 0, east: 0, west: 0 }
  const leftQueues: Record<Approach, number> = { north: 0, south: 0, east: 0, west: 0 }
  for (const v of vehicles) {
    if (!v.crossed && v.progress < 1) {
      const weight = v.kind === 'truck' ? 2 : 1
      queues[v.approach] += weight
      if (v.intent === 'left') {
        leftQueues[v.approach] += weight
      }
    }
  }
  return { queues, leftQueues }
}

function nextPhase(state: SimState): Pick<
  SimState,
  'phase' | 'phaseTimer' | 'nsGreen' | 'ewGreen' | 'nsLeftGreen' | 'ewLeftGreen' | 'pedestrianCrossing' | 'crosswalkRequest'
> {
  const { phase, queues, leftQueues, adaptive, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, crosswalkRequest } = state

  // Check for pedestrian crossing request
  const wantsPedestrian = crosswalkRequest.ns || crosswalkRequest.ew
  if (wantsPedestrian && phase !== 'pedestrian-crossing' && (phase === 'ns-yellow' || phase === 'ew-yellow')) {
    return {
      phase: 'pedestrian-crossing',
      phaseTimer: PEDESTRIAN_TIME,
      nsGreen,
      ewGreen,
      nsLeftGreen,
      ewLeftGreen,
      pedestrianCrossing: true,
      crosswalkRequest: { ns: false, ew: false },
    }
  }

  switch (phase) {
    case 'ns-green':
      return { phase: 'ns-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }

    case 'ns-yellow': {
      // Check if left turn needed
      const nsLeftQueue = leftQueues.north + leftQueues.south
      if (nsLeftQueue >= 2) {
        const green = adaptive
          ? Math.min(MAX_LEFT_GREEN, Math.max(MIN_LEFT_GREEN, 2 + nsLeftQueue * 0.8))
          : nsLeftGreen
        return { phase: 'ns-left-green', phaseTimer: green, nsGreen, ewGreen, nsLeftGreen: green, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }
      }
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }
    }

    case 'ns-left-green':
      return { phase: 'ns-left-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }

    case 'ns-left-yellow':
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }

    case 'all-red': {
      // Transition to EW or pedestrian
      if (wantsPedestrian) {
        return {
          phase: 'pedestrian-crossing',
          phaseTimer: PEDESTRIAN_TIME,
          nsGreen,
          ewGreen,
          nsLeftGreen,
          ewLeftGreen,
          pedestrianCrossing: true,
          crosswalkRequest: { ns: false, ew: false },
        }
      }
      let green = ewGreen
      if (adaptive) {
        const pressure = queues.east + queues.west
        green = Math.min(MAX_GREEN, Math.max(MIN_GREEN, 4 + pressure * 0.5))
      }
      return { phase: 'ew-green', phaseTimer: green, nsGreen, ewGreen: green, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }
    }

    case 'ew-green':
      return { phase: 'ew-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }

    case 'ew-yellow': {
      const ewLeftQueue = leftQueues.east + leftQueues.west
      if (ewLeftQueue >= 2) {
        const green = adaptive
          ? Math.min(MAX_LEFT_GREEN, Math.max(MIN_LEFT_GREEN, 2 + ewLeftQueue * 0.8))
          : ewLeftGreen
        return { phase: 'ew-left-green', phaseTimer: green, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen: green, pedestrianCrossing: false, crosswalkRequest }
      }
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }
    }

    case 'ew-left-green':
      return { phase: 'ew-left-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }

    case 'ew-left-yellow':
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }

    case 'pedestrian-crossing': {
      // After pedestrians, go to the axis that didn't just have green
      let green = nsGreen
      if (adaptive) {
        const pressure = queues.north + queues.south
        green = Math.min(MAX_GREEN, Math.max(MIN_GREEN, 4 + pressure * 0.5))
      }
      return { phase: 'ns-green', phaseTimer: green, nsGreen: green, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest }
    }
  }
}

export function stepSimulation(state: SimState, dt: number, rng: () => number = Math.random): SimState {
  let phase = state.phase
  let phaseTimer = state.phaseTimer - dt
  let nsGreen = state.nsGreen
  let ewGreen = state.ewGreen
  let nsLeftGreen = state.nsLeftGreen
  let ewLeftGreen = state.ewLeftGreen
  let throughput = state.throughput
  let pedestrianCrossing = state.pedestrianCrossing
  let crosswalkRequest = state.crosswalkRequest

  if (phaseTimer <= 0) {
    const nxt = nextPhase(state)
    phase = nxt.phase
    phaseTimer = nxt.phaseTimer
    nsGreen = nxt.nsGreen
    ewGreen = nxt.ewGreen
    nsLeftGreen = nxt.nsLeftGreen
    ewLeftGreen = nxt.ewLeftGreen
    pedestrianCrossing = nxt.pedestrianCrossing
    crosswalkRequest = nxt.crosswalkRequest
  }

  // Move vehicles — hard stop line on red, queue behind the car ahead
  const STOP_LINE = 0.97
  const vehicles: Vehicle[] = []
  for (const raw of state.vehicles) {
    const v = { ...raw }
    const canProceed = canVehicleProceed(phase, v)

    // Vehicle directly ahead in the same approach + lane
    let aheadProgress = Infinity
    for (const other of state.vehicles) {
      if (
        other.id !== v.id &&
        other.approach === v.approach &&
        other.lane === v.lane &&
        other.progress > v.progress &&
        other.progress < aheadProgress
      ) {
        aheadProgress = other.progress
      }
    }

    const turnSlowdown = v.intent === 'left' && v.progress > 1 ? 0.7 : 1
    let newProgress = v.progress + v.speed * dt * turnSlowdown

    // Red light: the stop line is a hard barrier until the phase allows this movement
    if (!v.crossed && !canProceed && v.progress < 1.0 && newProgress > STOP_LINE) {
      newProgress = STOP_LINE
    }

    // Car following: never close past the bumper of the vehicle ahead
    if (aheadProgress < Infinity) {
      const minGap = v.kind === 'truck' ? 0.08 : 0.062
      const maxAllowed = aheadProgress - minGap
      if (newProgress > maxAllowed) {
        newProgress = Math.max(v.progress, maxAllowed)
      }
    }

    v.waiting = !v.crossed && newProgress - v.progress < v.speed * dt * 0.25
    v.progress = newProgress

    if (v.progress >= 1.05 && !v.crossed) {
      v.crossed = true
      throughput += v.kind === 'truck' ? 2 : 1
    }

    if (v.progress < 3.4) {
      vehicles.push(v)
    }
  }

  // Move pedestrians
  const pedestrians: Pedestrian[] = []
  for (const raw of state.pedestrians) {
    const p = { ...raw }
    if (pedestrianCrossing && p.waiting) {
      p.crossing = true
      p.waiting = false
    }
    if (p.crossing && !p.done) {
      p.progress += dt * 0.25
      if (p.progress >= 1) {
        p.done = true
      }
    }
    if (!p.done || p.progress < 1.2) {
      pedestrians.push(p)
    }
  }

  // Ambient arrivals
  const demand = { ...state.demand }
  if (state.jamActive) {
    for (const key of Object.keys(demand) as Approach[]) {
      demand[key] = Math.max(0.25, demand[key] * (1 - 0.012 * dt))
    }
  }

  for (const approach of Object.keys(demand) as Approach[]) {
    const rate = demand[approach] * dt * 0.4
    if (rng() < rate) {
      const kind: VehicleKind = rng() < 0.15 ? 'truck' : 'car'
      const intent = randomIntent(rng)
      const lane = laneForIntent(intent)
      // Queue behind traffic already near the spawn point — never overlap.
      let spawn = 0
      for (const v of vehicles) {
        if (v.approach === approach && v.lane === lane && v.progress < 0.15) {
          spawn = Math.min(spawn, v.progress - 0.09)
        }
      }
      vehicles.push({
        id: nextId++,
        kind,
        approach,
        lane,
        intent,
        progress: spawn,
        speed: speedFor(kind, rng),
        waiting: false,
        crossed: false,
        color: randomColor(rng),
      })
    }
  }

  // Occasional pedestrian arrival
  if (rng() < 0.008 * dt * 60) {
    pedestrians.push({
      id: nextId++,
      approach: randomApproach(rng),
      progress: 0,
      crossing: false,
      waiting: true,
      done: false,
    })
  }

  const { queues, leftQueues } = recountQueues(vehicles)
  const jamActive = state.jamActive && Math.max(...Object.values(queues)) > 5

  return {
    ...state,
    vehicles,
    pedestrians,
    phase,
    phaseTimer,
    nsGreen,
    ewGreen,
    nsLeftGreen,
    ewLeftGreen,
    queues,
    leftQueues,
    throughput,
    jamActive,
    demand,
    tick: state.tick + 1,
    crosswalkRequest,
    pedestrianCrossing,
  }
}

export function phaseLabel(phase: Phase): string {
  switch (phase) {
    case 'ns-green': return 'North–South green'
    case 'ns-yellow': return 'North–South yellow'
    case 'ns-left-green': return 'NS left turn'
    case 'ns-left-yellow': return 'NS left yellow'
    case 'ew-green': return 'East–West green'
    case 'ew-yellow': return 'East–West yellow'
    case 'ew-left-green': return 'EW left turn'
    case 'ew-left-yellow': return 'EW left yellow'
    case 'all-red': return 'All red'
    case 'pedestrian-crossing': return 'Pedestrian crossing'
  }
}

export function lightColor(phase: Phase, axis: 'ns' | 'ew', lane: 'straight' | 'left' = 'straight'): 'red' | 'yellow' | 'green' {
  if (phase === 'pedestrian-crossing' || phase === 'all-red') return 'red'

  if (lane === 'left') {
    if (axis === 'ns') {
      if (phase === 'ns-left-green') return 'green'
      if (phase === 'ns-left-yellow') return 'yellow'
      return 'red'
    }
    if (phase === 'ew-left-green') return 'green'
    if (phase === 'ew-left-yellow') return 'yellow'
    return 'red'
  }

  if (axis === 'ns') {
    if (phase === 'ns-green') return 'green'
    if (phase === 'ns-yellow') return 'yellow'
    if (phase === 'ns-left-green' || phase === 'ns-left-yellow') return 'red' // straight stops during left turn
    return 'red'
  }
  if (phase === 'ew-green') return 'green'
  if (phase === 'ew-yellow') return 'yellow'
  if (phase === 'ew-left-green' || phase === 'ew-left-yellow') return 'red'
  return 'red'
}

export function pedestrianSignal(phase: Phase, axis: 'ns' | 'ew'): 'walk' | 'dont-walk' | 'flashing' {
  if (phase === 'pedestrian-crossing') return 'walk'
  // Pedestrians can walk parallel to green traffic
  if (axis === 'ns' && phase === 'ns-green') return 'walk'
  if (axis === 'ew' && phase === 'ew-green') return 'walk'
  return 'dont-walk'
}
