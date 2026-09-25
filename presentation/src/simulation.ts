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
  /** Seconds spent waiting at the curb. Used to auto-press the button. */
  waitTime: number
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
  /** Axis that gets the next green — the signal alternates NS ↔ EW. */
  nextAxis: 'ns' | 'ew'
  /** True after this walk released the curb snapshot; newcomers wait for the next walk. */
  walkCrowdReleased: boolean
  /** Seconds remaining before auto-press is allowed after a walk. Manual buttons ignore this. */
  pedWalkCooldown: number
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
/** Progress units per second on the zebra. 1 / 0.22 ≈ 4.55s to clear; PEDESTRIAN_TIME covers that. */
const PED_WALK_SPEED = 0.22
/** Fixed exclusive-walk length. Long enough for the snapshot cohort at PED_WALK_SPEED. */
const PEDESTRIAN_TIME = 6
/** Seconds a waiting pedestrian stands at the curb before auto-pressing.
 *  Long enough that a spawned crowd is visible before the exclusive walk. */
const PED_AUTO_PRESS_WAIT = 12
/** Don't auto-request walk until this many people have been waiting that long. */
const PED_AUTO_PRESS_MIN = 6
/** After a walk, wait this long before auto-press can fire again so NS/EW greens still run. */
const PED_AUTO_PRESS_COOLDOWN = 18
/** Hard barrier at the stop line for vehicles that may not enter the box. */
export const STOP_LINE = 0.97
/** Progress range that occupies the intersection box (after the stop line, before the exit). */
export const BOX_PROGRESS = { enter: 1, exit: 2 } as const

export function vehicleOccupiesBox(v: Vehicle): boolean {
  return v.progress > BOX_PROGRESS.enter && v.progress < BOX_PROGRESS.exit
}

export function pedestrianInCrosswalk(p: Pedestrian): boolean {
  return p.crossing && p.progress < 1
}

/** Follow / spawn gaps in progress units — sized to car (~0.09) and truck (~0.13) drawing length. */
const CAR_GAP = 0.10
const TRUCK_GAP = 0.135
const SPAWN_GAP = 0.11
const APPROACHES: Approach[] = ['north', 'east', 'south', 'west']
const LANES: Lane[] = ['straight', 'left', 'right']

const CAR_COLORS = ['#e74c3c', '#3498db', '#f39c12', '#ecf0f1', '#9b59b6', '#1abc9c']

let nextId = 1
let nextPedApproach = 0

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
    // Ambient demand tuned so a normal intersection never sits empty:
    // roughly one vehicle every 2–3 s per approach.
    demand: { north: 0.95, south: 0.95, east: 0.95, west: 0.95 },
    tick: 0,
    adaptive,
    crosswalkRequest: { ns: false, ew: false },
    pedestrianCrossing: false,
    nextAxis: 'ew',
    walkCrowdReleased: false,
    pedWalkCooldown: 0,
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
  // Progress units per second (before the global 1.35× time scale).
  // ~0.20 ≈ four seconds from the canvas edge to the stop line — fast enough
  // to read as motion, slow enough that approaches stay populated.
  if (kind === 'truck') return 0.14 + rng() * 0.05
  return 0.18 + rng() * 0.07
}

function randomColor(rng: () => number): string {
  return CAR_COLORS[Math.floor(rng() * CAR_COLORS.length)]
}

export function randomSpawnCounts(rng: () => number = Math.random): SpawnConfig {
  return {
    cars: 10 + Math.floor(rng() * 8),
    trucks: 2 + Math.floor(rng() * 4),
    pedestrians: 16 + Math.floor(rng() * 12),
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
    // Round-robin approaches and lanes so every arm of the intersection is
    // populated instead of RNG dumping a platoon down one off-screen lane.
    const approach = APPROACHES[i % APPROACHES.length]
    const lane = LANES[Math.floor(i / APPROACHES.length) % LANES.length]
    const intent: TurnIntent = lane
    // Keep the whole seed on the visible approach (progress 0 is the canvas
    // edge). The placement loop strictly decreases spawn each pass, so it
    // always terminates.
    let spawn = 0.18 + rng() * 0.62
    const gap = kind === 'truck' ? TRUCK_GAP : SPAWN_GAP
    for (;;) {
      let next = spawn
      for (const v of vehicles) {
        if (v.approach === approach && v.lane === lane && Math.abs(v.progress - spawn) < gap) {
          next = Math.min(next, v.progress - gap)
        }
      }
      if (next === spawn) break
      spawn = next
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
      approach: APPROACHES[i % APPROACHES.length],
      progress: 0,
      crossing: false,
      waiting: true,
      done: false,
      waitTime: 0,
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
  // Pack a stopped queue backward from the stop line, round-robin across
  // lanes so the jam is a visible wall at the stop line, not one lane of
  // cars stacked off-canvas. Track each lane's tail so mixed gaps never overlap.
  const laneTail: Record<Lane, number> = { left: 0.95, straight: 0.95, right: 0.95 }
  for (let i = 0; i < jamCount; i += 1) {
    const kind: VehicleKind = rng() < 0.2 ? 'truck' : 'car'
    const lane = LANES[i % LANES.length]
    const intent: TurnIntent = lane
    const gap = kind === 'truck' ? TRUCK_GAP : CAR_GAP
    // Pack behind any pre-existing traffic the queue reaches this far back.
    // Strictly decreasing placement loop, so it always terminates.
    let progress = laneTail[lane]
    for (;;) {
      let next = progress
      for (const v of vehicles) {
        if (v.approach === approach && v.lane === lane && Math.abs(v.progress - progress) < gap) {
          next = Math.min(next, v.progress - gap)
        }
      }
      if (next === progress) break
      progress = next
    }
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
  | 'phase'
  | 'phaseTimer'
  | 'nsGreen'
  | 'ewGreen'
  | 'nsLeftGreen'
  | 'ewLeftGreen'
  | 'pedestrianCrossing'
  | 'crosswalkRequest'
  | 'nextAxis'
> {
  const {
    phase,
    queues,
    leftQueues,
    adaptive,
    nsGreen,
    ewGreen,
    nsLeftGreen,
    ewLeftGreen,
    crosswalkRequest,
    nextAxis,
  } = state

  // A pedestrian request is honoured on all-red so the box can clear before
  // walkers enter the zebra. nextAxis already points at the axis that did
  // NOT just have green, so service resumes on the correct side afterwards.
  const wantsPedestrian = crosswalkRequest.ns || crosswalkRequest.ew

  const greenFor = (axis: 'ns' | 'ew'): number => {
    if (!adaptive) return axis === 'ns' ? nsGreen : ewGreen
    const pressure = axis === 'ns' ? queues.north + queues.south : queues.east + queues.west
    return Math.min(MAX_GREEN, Math.max(MIN_GREEN, 4 + pressure * 0.5))
  }

  switch (phase) {
    case 'ns-green':
      return { phase: 'ns-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }

    case 'ns-yellow': {
      // Check if left turn needed
      const nsLeftQueue = leftQueues.north + leftQueues.south
      if (nsLeftQueue >= 2) {
        const green = adaptive
          ? Math.min(MAX_LEFT_GREEN, Math.max(MIN_LEFT_GREEN, 2 + nsLeftQueue * 0.8))
          : nsLeftGreen
        return { phase: 'ns-left-green', phaseTimer: green, nsGreen, ewGreen, nsLeftGreen: green, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }
      }
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }
    }

    case 'ns-left-green':
      return { phase: 'ns-left-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }

    case 'ns-left-yellow':
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }

    case 'all-red': {
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
          nextAxis,
        }
      }
      // Serve whichever axis is due, then point nextAxis at the other side
      // so the following cycle (including a walk phase) resumes correctly.
      const green = greenFor(nextAxis)
      if (nextAxis === 'ns') {
        return { phase: 'ns-green', phaseTimer: green, nsGreen: green, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }
      }
      return { phase: 'ew-green', phaseTimer: green, nsGreen, ewGreen: green, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }
    }

    case 'ew-green':
      return { phase: 'ew-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }

    case 'ew-yellow': {
      const ewLeftQueue = leftQueues.east + leftQueues.west
      if (ewLeftQueue >= 2) {
        const green = adaptive
          ? Math.min(MAX_LEFT_GREEN, Math.max(MIN_LEFT_GREEN, 2 + ewLeftQueue * 0.8))
          : ewLeftGreen
        return { phase: 'ew-left-green', phaseTimer: green, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen: green, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }
      }
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }
    }

    case 'ew-left-green':
      return { phase: 'ew-left-yellow', phaseTimer: YELLOW, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }

    case 'ew-left-yellow':
      return { phase: 'all-red', phaseTimer: ALL_RED, nsGreen, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }

    case 'pedestrian-crossing': {
      // Resume on the axis that was waiting, not a hard-coded NS green.
      const green = greenFor(nextAxis)
      if (nextAxis === 'ns') {
        return { phase: 'ns-green', phaseTimer: green, nsGreen: green, ewGreen, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ew' }
      }
      return { phase: 'ew-green', phaseTimer: green, nsGreen, ewGreen: green, nsLeftGreen, ewLeftGreen, pedestrianCrossing: false, crosswalkRequest, nextAxis: 'ns' }
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
  let nextAxis = state.nextAxis
  let walkCrowdReleased = state.walkCrowdReleased
  let pedWalkCooldown = Math.max(0, state.pedWalkCooldown - dt)

  if (phaseTimer <= 0) {
    const wantsPed = state.crosswalkRequest.ns || state.crosswalkRequest.ew
    const boxBusy = state.vehicles.some(vehicleOccupiesBox)
    const pedsInRoad = state.pedestrians.some(pedestrianInCrosswalk)
    // Walk never starts while a vehicle is still in the box. Hold only while
    // the snapshot cohort is still on the zebra — newcomers wait at the curb
    // and must not extend this walk.
    if (phase === 'all-red' && wantsPed && boxBusy) {
      phaseTimer = ALL_RED
    } else if (phase === 'pedestrian-crossing' && pedsInRoad) {
      phaseTimer = 0.4
    } else {
      const nxt = nextPhase(state)
      phase = nxt.phase
      phaseTimer = nxt.phaseTimer
      nsGreen = nxt.nsGreen
      ewGreen = nxt.ewGreen
      nsLeftGreen = nxt.nsLeftGreen
      ewLeftGreen = nxt.ewLeftGreen
      pedestrianCrossing = nxt.pedestrianCrossing
      crosswalkRequest = nxt.crosswalkRequest
      nextAxis = nxt.nextAxis
    }
  }

  if (phase !== 'pedestrian-crossing') {
    if (state.phase === 'pedestrian-crossing') {
      pedWalkCooldown = PED_AUTO_PRESS_COOLDOWN
    }
    walkCrowdReleased = false
  }

  // Move vehicles — hard stop line on red, queue behind the car ahead
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

    // Red / walk / all-red: nobody who has not yet entered the box may roll
    // past the stop line. Vehicles already in the box keep clearing.
    if (!v.crossed && !canProceed && v.progress < BOX_PROGRESS.enter) {
      if (v.progress <= STOP_LINE) {
        if (newProgress > STOP_LINE) newProgress = STOP_LINE
      } else {
        newProgress = v.progress
      }
    }

    // Car following: never close past the bumper of the vehicle ahead
    if (aheadProgress < Infinity) {
      const minGap = v.kind === 'truck' ? TRUCK_GAP : CAR_GAP
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

  // Move pedestrians — they ONLY enter the zebra during the dedicated
  // all-vehicle-red walk, and only if they were already waiting when that
  // walk started. New arrivals during walk stay at the curb for the next one.
  const pedestrians: Pedestrian[] = []
  let autoRequestNS = false
  let autoRequestEW = false
  const admitWaitingCrowd = phase === 'pedestrian-crossing' && !walkCrowdReleased
  for (const raw of state.pedestrians) {
    const p = { ...raw }
    if (p.waiting && admitWaitingCrowd) {
      p.crossing = true
      p.waiting = false
    }
    // Keep walking a little past the far curb (progress 1.2) so finished
    // pedestrians step onto the sidewalk instead of vanishing mid-crosswalk —
    // and so they actually leave the state instead of accumulating forever.
    if (p.crossing && p.progress < 1.2) {
      p.progress += dt * PED_WALK_SPEED
      if (p.progress >= 1) {
        p.done = true
      }
    }
    // Auto-press after a few seconds of waiting. Never on tick 0 so the
    // opening frame shows a cluster rather than an instant walk.
    if (p.waiting) {
      p.waitTime = (p.waitTime ?? 0) + dt
    }
    if (!p.done || p.progress < 1.2) {
      pedestrians.push(p)
    }
  }
  if (admitWaitingCrowd) walkCrowdReleased = true

  const readyWaiters = pedestrians.filter(
    (p) => p.waiting && (p.waitTime ?? 0) >= PED_AUTO_PRESS_WAIT,
  ).length
  const autoPressAllowed =
    state.tick > 0 &&
    phase !== 'pedestrian-crossing' &&
    pedWalkCooldown <= 0 &&
    readyWaiters >= PED_AUTO_PRESS_MIN
  if (autoPressAllowed) {
    for (const p of pedestrians) {
      if (p.waiting && (p.waitTime ?? 0) >= PED_AUTO_PRESS_WAIT) {
        if (p.approach === 'north' || p.approach === 'south') autoRequestEW = true
        else autoRequestNS = true
      }
    }
  }
  if (autoRequestNS) crosswalkRequest = { ...crosswalkRequest, ns: true }
  if (autoRequestEW) crosswalkRequest = { ...crosswalkRequest, ew: true }

  // Ambient arrivals
  const demand = { ...state.demand }
  if (state.jamActive) {
    for (const key of Object.keys(demand) as Approach[]) {
      demand[key] = Math.max(0.25, demand[key] * (1 - 0.012 * dt))
    }
  }

  for (const approach of Object.keys(demand) as Approach[]) {
    const rate = demand[approach] * dt * 0.55
    if (rng() < rate) {
      const kind: VehicleKind = rng() < 0.15 ? 'truck' : 'car'
      const intent = randomIntent(rng)
      const lane = laneForIntent(intent)
      const gap = kind === 'truck' ? TRUCK_GAP : SPAWN_GAP
      // Appear at the canvas edge so new arrivals are visible immediately,
      // then queue behind anyone already near the entry.
      let spawn = 0.02 + rng() * 0.04
      for (const v of vehicles) {
        if (v.approach === approach && v.lane === lane && v.progress < 0.35) {
          spawn = Math.min(spawn, v.progress - gap)
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

  // Ambient arrivals high enough that a waiting cluster rebuilds between walks.
  // Round-robin corners so the pile doesn't collapse onto one sidewalk.
  if (rng() < 0.035 * dt * 60) {
    const approach = APPROACHES[nextPedApproach % APPROACHES.length]
    nextPedApproach += 1
    pedestrians.push({
      id: nextId++,
      approach,
      progress: 0,
      crossing: false,
      waiting: true,
      done: false,
      waitTime: 0,
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
    nextAxis,
    walkCrowdReleased,
    pedWalkCooldown,
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

export function pedestrianSignal(phase: Phase, _axis: 'ns' | 'ew'): 'walk' | 'dont-walk' | 'flashing' {
  if (phase === 'pedestrian-crossing') return 'walk'
  return 'dont-walk'
}
