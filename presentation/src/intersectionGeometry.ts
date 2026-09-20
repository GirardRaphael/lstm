import type { Approach, Vehicle } from './simulation'

export const SIZE = 800
export const CENTER = SIZE / 2
export const ROAD = 180
export const LANE_W = 28
export const LANE_GAP = 4

export type Lane = 'left' | 'straight' | 'right'
export type Vec = { x: number; y: number }

export function laneOffset(lane: Lane): number {
  // Left-turn lane sits nearest the centre line (right-hand traffic).
  if (lane === 'left') return LANE_W + LANE_GAP
  if (lane === 'right') return -(LANE_W + LANE_GAP)
  return 0
}

// Stop-line distance from the canvas edge for each approach.
export const STOP = CENTER - ROAD / 2 - 10 // 300
// Spawn sits just off the canvas edge so progress≈0 is already readable
// as a vehicle rolling onto the approach instead of 40px off-screen.
const SPAWN_DIST = 8

// Heading angles (canvas coords, y down): 0 = east, π/2 = south.
export const HEADING: Record<Approach, number> = {
  west: 0,
  north: Math.PI / 2,
  east: Math.PI,
  south: -Math.PI / 2,
}

// Exact departure heading per approach + intent (right-hand traffic), so a
// vehicle tracks the centre of its exit lane after clearing the box instead
// of drifting along the Bezier's terminal tangent.
export const EXIT_HEADING: Record<Approach, Record<Lane, number>> = {
  north: { straight: Math.PI / 2, right: Math.PI, left: 0 },
  south: { straight: -Math.PI / 2, right: 0, left: Math.PI },
  west: { straight: 0, right: Math.PI / 2, left: -Math.PI / 2 },
  east: { straight: Math.PI, right: -Math.PI / 2, left: Math.PI / 2 },
}

/**
 * Right-hand traffic destination lane centres (canvas Y-down).
 *   southbound → west half of NS  (x < CENTER)
 *   northbound → east half of NS  (x > CENTER)
 *   eastbound  → south half of EW (y > CENTER)
 *   westbound  → north half of EW (y < CENTER)
 */
export function southboundX(lane: Lane): number {
  return CENTER - ROAD / 4 + laneOffset(lane)
}
export function northboundX(lane: Lane): number {
  return CENTER + ROAD / 4 - laneOffset(lane)
}
export function eastboundY(lane: Lane): number {
  return CENTER + ROAD / 4 - laneOffset(lane)
}
export function westboundY(lane: Lane): number {
  return CENTER - ROAD / 4 + laneOffset(lane)
}

export function laneBase(approach: Approach, lane: Lane): Vec {
  switch (approach) {
    case 'north':
      return { x: southboundX(lane), y: 0 }
    case 'south':
      return { x: northboundX(lane), y: 0 }
    case 'west':
      return { x: 0, y: eastboundY(lane) }
    case 'east':
      return { x: 0, y: westboundY(lane) }
  }
}

export function stopPoint(approach: Approach, lane: Lane): Vec {
  const base = laneBase(approach, lane)
  switch (approach) {
    case 'north':
      return { x: base.x, y: STOP }
    case 'south':
      return { x: base.x, y: SIZE - STOP }
    case 'west':
      return { x: STOP, y: base.y }
    case 'east':
      return { x: SIZE - STOP, y: base.y }
  }
}

export function spawnPoint(approach: Approach, lane: Lane): Vec {
  const base = laneBase(approach, lane)
  switch (approach) {
    case 'north':
      return { x: base.x, y: -SPAWN_DIST }
    case 'south':
      return { x: base.x, y: SIZE + SPAWN_DIST }
    case 'west':
      return { x: -SPAWN_DIST, y: base.y }
    case 'east':
      return { x: SIZE + SPAWN_DIST, y: base.y }
  }
}

export function exitPoint(approach: Approach, intent: Lane, lane: Lane): Vec {
  const past = ROAD / 2 + 70
  switch (approach) {
    case 'north': // heading south
      if (intent === 'straight') return { x: southboundX(lane), y: CENTER + past }
      if (intent === 'right') return { x: CENTER - past, y: westboundY(lane) } // westbound, north half (y < CENTER)
      return { x: CENTER + past, y: eastboundY(lane) } // left → eastbound, south half (y > CENTER)
    case 'south': // heading north
      if (intent === 'straight') return { x: northboundX(lane), y: CENTER - past }
      if (intent === 'right') return { x: CENTER + past, y: eastboundY(lane) } // eastbound, south half
      return { x: CENTER - past, y: westboundY(lane) } // left → westbound, north half
    case 'west': // heading east
      if (intent === 'straight') return { x: CENTER + past, y: eastboundY(lane) }
      if (intent === 'right') return { x: southboundX(lane), y: CENTER + past } // southbound, west half
      return { x: northboundX(lane), y: CENTER - past } // left → northbound, east half
    case 'east': // heading west
      if (intent === 'straight') return { x: CENTER - past, y: westboundY(lane) }
      if (intent === 'right') return { x: northboundX(lane), y: CENTER - past } // northbound, east half
      return { x: southboundX(lane), y: CENTER + past } // left → southbound, west half
  }
}

export function bezier(p0: Vec, p1: Vec, p2: Vec, t: number): Vec {
  const u = 1 - t
  return {
    x: u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
    y: u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y,
  }
}

function bezierDir(p0: Vec, p1: Vec, p2: Vec, t: number): Vec {
  return {
    x: 2 * (1 - t) * (p1.x - p0.x) + 2 * t * (p2.x - p1.x),
    y: 2 * (1 - t) * (p1.y - p0.y) + 2 * t * (p2.y - p1.y),
  }
}

export function controlPoint(approach: Approach, intent: Lane, p0: Vec, p2: Vec): Vec {
  const h = HEADING[approach]
  const fwd = { x: Math.cos(h), y: Math.sin(h) }
  if (intent === 'straight') {
    return { x: p0.x + fwd.x * (ROAD / 2), y: p0.y + fwd.y * (ROAD / 2) }
  }
  // Elbow of the two lanes: stay in the departure lane until the destination
  // lane so the quadratic does not cut across the centre line into oncoming.
  const isNS = approach === 'north' || approach === 'south'
  if (intent === 'right') {
    const inset = 10
    if (isNS) return { x: p0.x + fwd.x * inset, y: p2.y }
    return { x: p2.x, y: p0.y + fwd.y * inset }
  }
  if (isNS) return { x: p0.x, y: p2.y }
  return { x: p2.x, y: p0.y }
}

export function vehiclePose(v: Pick<Vehicle, 'approach' | 'lane' | 'intent' | 'progress'>): {
  x: number
  y: number
  angle: number
} {
  const p0 = stopPoint(v.approach, v.lane)
  const sp = spawnPoint(v.approach, v.lane)
  const heading = HEADING[v.approach]

  if (v.progress <= 1) {
    const t = v.progress
    return {
      x: sp.x + (p0.x - sp.x) * t,
      y: sp.y + (p0.y - sp.y) * t,
      angle: heading,
    }
  }

  const p2 = exitPoint(v.approach, v.intent, v.lane)
  const p1 = controlPoint(v.approach, v.intent, p0, p2)
  const t = Math.min(v.progress - 1, 1)
  const pos = bezier(p0, p1, p2, t)
  const dir = bezierDir(p0, p1, p2, t)
  let angle = Math.atan2(dir.y, dir.x)

  if (v.progress > 2) {
    const exitHeading = EXIT_HEADING[v.approach][v.intent]
    const extra = (v.progress - 2) * 230
    pos.x = p2.x + Math.cos(exitHeading) * extra
    pos.y = p2.y + Math.sin(exitHeading) * extra
    angle = exitHeading
  }

  return { x: pos.x, y: pos.y, angle }
}
