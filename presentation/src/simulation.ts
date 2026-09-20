export type VehicleKind = 'car' | 'truck' | 'pedestrian'
export type Approach = 'north' | 'south' | 'east' | 'west'
export type Phase = 'ns-green' | 'ns-yellow' | 'ew-green' | 'ew-yellow'

export type Agent = {
  id: number
  kind: VehicleKind
  approach: Approach
  /** 0 at spawn far from stop line, 1 at stop line, >1 crossing / leaving */
  progress: number
  speed: number
  waiting: boolean
  crossed: boolean
}

export type SimState = {
  agents: Agent[]
  phase: Phase
  phaseTimer: number
  nsGreen: number
  ewGreen: number
  queues: Record<Approach, number>
  throughput: number
  jamActive: boolean
  demand: Record<Approach, number>
  tick: number
  adaptive: boolean
}

export type SpawnConfig = {
  cars: number
  trucks: number
  pedestrians: number
}

const YELLOW = 1.2
const MIN_GREEN = 3.5
const MAX_GREEN = 14

let nextId = 1

export function createInitialState(adaptive = true): SimState {
  return {
    agents: [],
    phase: 'ns-green',
    phaseTimer: 6,
    nsGreen: 6,
    ewGreen: 6,
    queues: { north: 0, south: 0, east: 0, west: 0 },
    throughput: 0,
    jamActive: false,
    demand: { north: 0.35, south: 0.35, east: 0.35, west: 0.35 },
    tick: 0,
    adaptive,
  }
}

function randomApproach(rng: () => number): Approach {
  const r = rng()
  if (r < 0.25) return 'north'
  if (r < 0.5) return 'south'
  if (r < 0.75) return 'east'
  return 'west'
}

function speedFor(kind: VehicleKind, rng: () => number): number {
  if (kind === 'pedestrian') return 0.012 + rng() * 0.008
  if (kind === 'truck') return 0.018 + rng() * 0.01
  return 0.028 + rng() * 0.015
}

export function spawnAgents(
  state: SimState,
  config: SpawnConfig,
  rng: () => number = Math.random,
): SimState {
  const agents = [...state.agents]
  const push = (kind: VehicleKind, count: number) => {
    for (let i = 0; i < count; i += 1) {
      const approach = randomApproach(rng)
      agents.push({
        id: nextId++,
        kind,
        approach,
        progress: rng() * 0.35,
        speed: speedFor(kind, rng),
        waiting: false,
        crossed: false,
      })
    }
  }
  push('car', config.cars)
  push('truck', config.trucks)
  push('pedestrian', config.pedestrians)
  return { ...state, agents }
}

export function createTrafficJam(
  state: SimState,
  approach: Approach = 'east',
  rng: () => number = Math.random,
): SimState {
  const agents = [...state.agents]
  const jamCount = 18 + Math.floor(rng() * 10)
  for (let i = 0; i < jamCount; i += 1) {
    const kind: VehicleKind = rng() < 0.25 ? 'truck' : 'car'
    agents.push({
      id: nextId++,
      kind,
      approach,
      progress: Math.min(0.92, 0.05 + i * 0.045 + rng() * 0.02),
      speed: speedFor(kind, rng) * 0.7,
      waiting: true,
      crossed: false,
    })
  }
  const demand = { ...state.demand, [approach]: 1.4 }
  return { ...state, agents, jamActive: true, demand }
}

function isGreen(phase: Phase, approach: Approach): boolean {
  if (approach === 'north' || approach === 'south') {
    return phase === 'ns-green'
  }
  return phase === 'ew-green'
}

function isYellow(phase: Phase, approach: Approach): boolean {
  if (approach === 'north' || approach === 'south') {
    return phase === 'ns-yellow'
  }
  return phase === 'ew-yellow'
}

function recountQueues(agents: Agent[]): Record<Approach, number> {
  const queues: Record<Approach, number> = { north: 0, south: 0, east: 0, west: 0 }
  for (const a of agents) {
    if (!a.crossed && a.progress < 1 && a.kind !== 'pedestrian') {
      queues[a.approach] += a.kind === 'truck' ? 2 : 1
    }
  }
  return queues
}

function nextPhase(state: SimState): Pick<SimState, 'phase' | 'phaseTimer' | 'nsGreen' | 'ewGreen'> {
  const { phase, queues, adaptive, nsGreen, ewGreen } = state
  if (phase === 'ns-green') {
    return { phase: 'ns-yellow', phaseTimer: YELLOW, nsGreen, ewGreen }
  }
  if (phase === 'ns-yellow') {
    let green = ewGreen
    if (adaptive) {
      const pressure = queues.east + queues.west
      green = Math.min(MAX_GREEN, Math.max(MIN_GREEN, 3.5 + pressure * 0.55))
    }
    return { phase: 'ew-green', phaseTimer: green, nsGreen, ewGreen: green }
  }
  if (phase === 'ew-green') {
    return { phase: 'ew-yellow', phaseTimer: YELLOW, nsGreen, ewGreen }
  }
  let green = nsGreen
  if (adaptive) {
    const pressure = queues.north + queues.south
    green = Math.min(MAX_GREEN, Math.max(MIN_GREEN, 3.5 + pressure * 0.55))
  }
  return { phase: 'ns-green', phaseTimer: green, nsGreen: green, ewGreen }
}

export function stepSimulation(state: SimState, dt: number, rng: () => number = Math.random): SimState {
  let phase = state.phase
  let phaseTimer = state.phaseTimer - dt
  let nsGreen = state.nsGreen
  let ewGreen = state.ewGreen
  let throughput = state.throughput

  if (phaseTimer <= 0) {
    const nxt = nextPhase(state)
    phase = nxt.phase
    phaseTimer = nxt.phaseTimer
    nsGreen = nxt.nsGreen
    ewGreen = nxt.ewGreen
  }

  const agents: Agent[] = []
  for (const raw of state.agents) {
    const agent = { ...raw }
    const green = isGreen(phase, agent.approach)
    const yellow = isYellow(phase, agent.approach)
    const atStop = agent.progress >= 0.95 && agent.progress < 1.02

    if (!agent.crossed && atStop && !green) {
      agent.waiting = true
      // creep a little on yellow if already committed
      if (yellow && agent.progress > 0.98) {
        agent.progress += agent.speed * dt * 0.4
      }
    } else {
      agent.waiting = false
      agent.progress += agent.speed * dt * (agent.kind === 'pedestrian' ? 0.85 : 1)
    }

    if (agent.progress >= 1.05 && !agent.crossed) {
      agent.crossed = true
      if (agent.kind !== 'pedestrian') throughput += agent.kind === 'truck' ? 2 : 1
    }

    if (agent.progress < 2.2) {
      agents.push(agent)
    }
  }

  // Ambient arrivals from demand
  const demand = { ...state.demand }
  if (state.jamActive) {
    // jam fades slowly
    for (const key of Object.keys(demand) as Approach[]) {
      demand[key] = Math.max(0.25, demand[key] * (1 - 0.015 * dt))
    }
  }

  for (const approach of Object.keys(demand) as Approach[]) {
    const rate = demand[approach] * dt * 0.55
    if (rng() < rate) {
      const roll = rng()
      const kind: VehicleKind = roll < 0.12 ? 'pedestrian' : roll < 0.28 ? 'truck' : 'car'
      agents.push({
        id: nextId++,
        kind,
        approach,
        progress: 0,
        speed: speedFor(kind, rng),
        waiting: false,
        crossed: false,
      })
    }
  }

  const queues = recountQueues(agents)
  const jamActive = state.jamActive && Math.max(queues.east, queues.west, queues.north, queues.south) > 6

  return {
    ...state,
    agents,
    phase,
    phaseTimer,
    nsGreen,
    ewGreen,
    queues,
    throughput,
    jamActive,
    demand,
    tick: state.tick + 1,
  }
}

export function phaseLabel(phase: Phase): string {
  switch (phase) {
    case 'ns-green':
      return 'North–South green'
    case 'ns-yellow':
      return 'North–South yellow'
    case 'ew-green':
      return 'East–West green'
    case 'ew-yellow':
      return 'East–West yellow'
  }
}

export function lightColor(phase: Phase, axis: 'ns' | 'ew'): 'red' | 'yellow' | 'green' {
  if (axis === 'ns') {
    if (phase === 'ns-green') return 'green'
    if (phase === 'ns-yellow') return 'yellow'
    return 'red'
  }
  if (phase === 'ew-green') return 'green'
  if (phase === 'ew-yellow') return 'yellow'
  return 'red'
}
