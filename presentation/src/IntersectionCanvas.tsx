import { useLayoutEffect, useRef } from 'react'
import {
  type Approach,
  type Pedestrian,
  type Vehicle,
  lightColor,
  pedestrianSignal,
  type SimState,
} from './simulation'

const SIZE = 800
const CENTER = SIZE / 2
const ROAD = 180
const LANE_W = 28
const LANE_GAP = 4

type Lane = 'left' | 'straight' | 'right'

type Vec = { x: number; y: number }

function laneOffset(lane: Lane): number {
  // Left-turn lane sits nearest the centre line (right-hand traffic).
  if (lane === 'left') return LANE_W + LANE_GAP
  if (lane === 'right') return -(LANE_W + LANE_GAP)
  return 0
}

// Stop-line distance from the canvas edge for each approach.
const STOP = CENTER - ROAD / 2 - 10 // 300
// Spawn sits just off the canvas edge so progress≈0 is already readable
// as a vehicle rolling onto the approach instead of 40px off-screen.
const SPAWN_DIST = 8

// Heading angles (canvas coords, y down): 0 = east, π/2 = south.
const HEADING: Record<Approach, number> = {
  west: 0,
  north: Math.PI / 2,
  east: Math.PI,
  south: -Math.PI / 2,
}

// Exact departure heading per approach + intent (right-hand traffic), so a
// vehicle tracks the centre of its exit lane after clearing the box instead
// of drifting along the Bezier's terminal tangent.
const EXIT_HEADING: Record<Approach, Record<'left' | 'straight' | 'right', number>> = {
  north: { straight: Math.PI / 2, right: Math.PI, left: 0 },
  south: { straight: -Math.PI / 2, right: 0, left: Math.PI },
  west: { straight: 0, right: Math.PI / 2, left: -Math.PI / 2 },
  east: { straight: Math.PI, right: -Math.PI / 2, left: Math.PI / 2 },
}

function laneBase(approach: Approach, lane: Lane): Vec {
  // Right-hand traffic: the approach lane sits on the driver's right side.
  const o = laneOffset(lane)
  switch (approach) {
    case 'north': return { x: CENTER - ROAD / 4 + o, y: 0 }
    case 'south': return { x: CENTER + ROAD / 4 - o, y: 0 }
    case 'west': return { x: 0, y: CENTER + ROAD / 4 - o }
    case 'east': return { x: 0, y: CENTER - ROAD / 4 + o }
  }
}

function stopPoint(approach: Approach, lane: Lane): Vec {
  const base = laneBase(approach, lane)
  switch (approach) {
    case 'north': return { x: base.x, y: STOP }
    case 'south': return { x: base.x, y: SIZE - STOP }
    case 'west': return { x: STOP, y: base.y }
    case 'east': return { x: SIZE - STOP, y: base.y }
  }
}

function spawnPoint(approach: Approach, lane: Lane): Vec {
  const base = laneBase(approach, lane)
  switch (approach) {
    case 'north': return { x: base.x, y: -SPAWN_DIST }
    case 'south': return { x: base.x, y: SIZE + SPAWN_DIST }
    case 'west': return { x: -SPAWN_DIST, y: base.y }
    case 'east': return { x: SIZE + SPAWN_DIST, y: base.y }
  }
}

// Where a vehicle exits after crossing, per intent (right-hand traffic).
function exitPoint(approach: Approach, intent: 'left' | 'straight' | 'right', lane: Lane): Vec {
  const o = laneOffset(lane)
  const past = ROAD / 2 + 70
  switch (approach) {
    case 'north': // heading south
      if (intent === 'straight') return { x: CENTER - ROAD / 4 + o, y: CENTER + past }
      if (intent === 'right') return { x: CENTER - past, y: CENTER - ROAD / 4 + o } // westbound
      return { x: CENTER + past, y: CENTER + ROAD / 4 - o } // left → eastbound
    case 'south': // heading north
      if (intent === 'straight') return { x: CENTER + ROAD / 4 - o, y: CENTER - past }
      if (intent === 'right') return { x: CENTER + past, y: CENTER + ROAD / 4 - o } // eastbound
      return { x: CENTER - past, y: CENTER - ROAD / 4 + o } // left → westbound
    case 'west': // heading east
      if (intent === 'straight') return { x: CENTER + past, y: CENTER + ROAD / 4 - o }
      if (intent === 'right') return { x: CENTER + ROAD / 4 - o, y: CENTER + past } // southbound
      return { x: CENTER - ROAD / 4 + o, y: CENTER - past } // left → northbound
    case 'east': // heading west
      if (intent === 'straight') return { x: CENTER - past, y: CENTER - ROAD / 4 + o }
      if (intent === 'right') return { x: CENTER - ROAD / 4 + o, y: CENTER - past } // northbound
      return { x: CENTER + ROAD / 4 - o, y: CENTER + past } // left → southbound
  }
}

function bezier(p0: Vec, p1: Vec, p2: Vec, t: number): Vec {
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

function controlPoint(approach: Approach, intent: 'left' | 'straight' | 'right', p0: Vec): Vec {
  const h = HEADING[approach]
  const fwd = { x: Math.cos(h), y: Math.sin(h) }
  if (intent === 'straight') {
    return { x: p0.x + fwd.x * (ROAD / 2), y: p0.y + fwd.y * (ROAD / 2) }
  }
  if (intent === 'right') {
    // tight corner
    return { x: p0.x + fwd.x * 34, y: p0.y + fwd.y * 34 }
  }
  // left: sweep toward the middle of the box
  return {
    x: p0.x + fwd.x * (ROAD * 0.62),
    y: p0.y + fwd.y * (ROAD * 0.62),
  }
}

function vehiclePose(v: Vehicle): { x: number; y: number; angle: number } {
  const p0 = stopPoint(v.approach, v.lane)
  const sp = spawnPoint(v.approach, v.lane)
  const heading = HEADING[v.approach]

  if (v.progress <= 1) {
    // Negative progress extends backward past the spawn point so queued
    // arrivals stay on the correct heading instead of stacking at the edge.
    const t = v.progress
    return {
      x: sp.x + (p0.x - sp.x) * t,
      y: sp.y + (p0.y - sp.y) * t,
      angle: heading,
    }
  }

  const p2 = exitPoint(v.approach, v.intent, v.lane)
  const p1 = controlPoint(v.approach, v.intent, p0)
  const t = Math.min(v.progress - 1, 1)
  const pos = bezier(p0, p1, p2, t)
  const dir = bezierDir(p0, p1, p2, t)
  let angle = Math.atan2(dir.y, dir.x)

  if (v.progress > 2) {
    // continue straight along the exit lane's exact heading
    const exitHeading = EXIT_HEADING[v.approach][v.intent]
    const extra = (v.progress - 2) * 230
    pos.x = p2.x + Math.cos(exitHeading) * extra
    pos.y = p2.y + Math.sin(exitHeading) * extra
    angle = exitHeading
  }

  return { x: pos.x, y: pos.y, angle }
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

function drawRoad(ctx: CanvasRenderingContext2D) {
  const bg = ctx.createLinearGradient(0, 0, 0, SIZE)
  bg.addColorStop(0, '#0a0f14')
  bg.addColorStop(1, '#0d141a')
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, SIZE, SIZE)

  const half = (SIZE - ROAD) / 2

  // Corner blocks — sidewalk ring + inner grass
  const corners: Array<[number, number]> = [
    [0, 0],
    [SIZE - half, 0],
    [0, SIZE - half],
    [SIZE - half, SIZE - half],
  ]
  for (const [cx, cy] of corners) {
    // sidewalk
    ctx.fillStyle = '#232b33'
    ctx.fillRect(cx, cy, half, half)
    // grass inset
    const g = ctx.createLinearGradient(cx, cy, cx + half, cy + half)
    g.addColorStop(0, '#163026')
    g.addColorStop(1, '#0e1f19')
    ctx.fillStyle = g
    ctx.fillRect(cx + 10, cy + 10, half - 20, half - 20)
    // sidewalk seam lines
    ctx.strokeStyle = 'rgba(244,239,230,0.05)'
    ctx.lineWidth = 1
    for (let i = 1; i < 4; i += 1) {
      const o = (half / 4) * i
      ctx.beginPath()
      ctx.moveTo(cx + o, cy + 4)
      ctx.lineTo(cx + o, cy + half - 4)
      ctx.moveTo(cx + 4, cy + o)
      ctx.lineTo(cx + half - 4, cy + o)
      ctx.stroke()
    }
  }

  // Road shadow
  ctx.fillStyle = 'rgba(0,0,0,0.4)'
  ctx.fillRect(half - 4, 0, ROAD + 8, SIZE)
  ctx.fillRect(0, half - 4, SIZE, ROAD + 8)

  // Asphalt
  const road = ctx.createLinearGradient(half, 0, half + ROAD, 0)
  road.addColorStop(0, '#1a232c')
  road.addColorStop(0.5, '#212d38')
  road.addColorStop(1, '#182028')
  ctx.fillStyle = road
  ctx.fillRect(half, 0, ROAD, SIZE)

  const roadH = ctx.createLinearGradient(0, half, 0, half + ROAD)
  roadH.addColorStop(0, '#1a232c')
  roadH.addColorStop(0.5, '#212d38')
  roadH.addColorStop(1, '#182028')
  ctx.fillStyle = roadH
  ctx.fillRect(0, half, SIZE, ROAD)

  // Asphalt texture — deterministic sparse speckle
  ctx.fillStyle = 'rgba(244,239,230,0.025)'
  for (let i = 0; i < 130; i += 1) {
    const sx = half + ((i * 137.5) % ROAD)
    const sy = (i * 89.3) % SIZE
    ctx.fillRect(sx, sy, 2, 2)
    const hx = (i * 97.7) % SIZE
    const hy = half + ((i * 61.3) % ROAD)
    ctx.fillRect(hx, hy, 2, 2)
  }

  // Center box
  ctx.fillStyle = '#1e2933'
  ctx.fillRect(half, half, ROAD, ROAD)

  // Lane markings — dashed separators BETWEEN the three lanes of each
  // carriageway. Vehicles drive centered in their lane, so markings must
  // run in the gaps, never under the vehicles.
  const SEP = LANE_W / 2 + LANE_GAP / 2 // midpoint between adjacent lane centers
  ctx.strokeStyle = 'rgba(232,224,200,0.45)'
  ctx.lineWidth = 2
  ctx.setLineDash([16, 14])

  // Vertical road lane separators
  for (const dir of [-1, 1]) {
    const baseX = CENTER + dir * (ROAD / 4)
    for (const off of [-SEP, SEP]) {
      const x = baseX + off
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, half)
      ctx.moveTo(x, half + ROAD)
      ctx.lineTo(x, SIZE)
      ctx.stroke()
    }
  }

  // Horizontal road lane separators
  for (const dir of [-1, 1]) {
    const baseY = CENTER + dir * (ROAD / 4)
    for (const off of [-SEP, SEP]) {
      const y = baseY + off
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(half, y)
      ctx.moveTo(half + ROAD, y)
      ctx.lineTo(SIZE, y)
      ctx.stroke()
    }
  }
  ctx.setLineDash([])

  // Painted lane arrows on approach lanes (left / straight / right)
  const arrow = (x: number, y: number, angle: number, kind: 'left' | 'straight' | 'right') => {
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.strokeStyle = 'rgba(244,239,230,0.55)'
    ctx.fillStyle = 'rgba(244,239,230,0.55)'
    ctx.lineWidth = 2.5
    ctx.lineCap = 'round'
    if (kind === 'straight') {
      ctx.beginPath()
      ctx.moveTo(-10, 0)
      ctx.lineTo(8, 0)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(4, -5)
      ctx.lineTo(10, 0)
      ctx.lineTo(4, 5)
      ctx.closePath()
      ctx.fill()
    } else {
      const dir = kind === 'left' ? -1 : 1
      ctx.beginPath()
      ctx.moveTo(-10, 0)
      ctx.lineTo(2, 0)
      ctx.quadraticCurveTo(8, 0, 8, dir * 6)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(8 - 4, dir * 2)
      ctx.lineTo(8, dir * 8)
      ctx.lineTo(8 + 4, dir * 2)
      ctx.closePath()
      ctx.fill()
    }
    ctx.restore()
  }

  const arrowDist = half - 46
  const lanes: Array<'left' | 'straight' | 'right'> = ['left', 'straight', 'right']
  for (const lane of lanes) {
    const o = laneOffset(lane)
    // north approach (heading south)
    arrow(CENTER - ROAD / 4 + o, arrowDist, Math.PI / 2, lane)
    // south approach (heading north)
    arrow(CENTER + ROAD / 4 - o, SIZE - arrowDist, -Math.PI / 2, lane)
    // west approach (heading east)
    arrow(arrowDist, CENTER + ROAD / 4 - o, 0, lane)
    // east approach (heading west)
    arrow(SIZE - arrowDist, CENTER - ROAD / 4 + o, Math.PI, lane)
  }

  // Center lines (double yellow)
  ctx.strokeStyle = 'rgba(240,162,2,0.5)'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(CENTER - 2, 0)
  ctx.lineTo(CENTER - 2, half)
  ctx.moveTo(CENTER + 2, 0)
  ctx.lineTo(CENTER + 2, half)
  ctx.moveTo(CENTER - 2, half + ROAD)
  ctx.lineTo(CENTER - 2, SIZE)
  ctx.moveTo(CENTER + 2, half + ROAD)
  ctx.lineTo(CENTER + 2, SIZE)
  ctx.moveTo(0, CENTER - 2)
  ctx.lineTo(half, CENTER - 2)
  ctx.moveTo(0, CENTER + 2)
  ctx.lineTo(half, CENTER + 2)
  ctx.moveTo(half + ROAD, CENTER - 2)
  ctx.lineTo(SIZE, CENTER - 2)
  ctx.moveTo(half + ROAD, CENTER + 2)
  ctx.lineTo(SIZE, CENTER + 2)
  ctx.stroke()

  // Crosswalks — zebra stripes on the intersection side of the stop line,
  // so stopped vehicles never stand on the zebra.
  ctx.fillStyle = 'rgba(244,239,230,0.75)'
  const cwStripe = 8
  const cwGap = 6

  // North crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + 10 + i * (cwStripe + cwGap), half + 2, cwStripe, 16)
  }
  // South crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + 10 + i * (cwStripe + cwGap), half + ROAD - 18, cwStripe, 16)
  }
  // West crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + 2, half + 10 + i * (cwStripe + cwGap), 16, cwStripe)
  }
  // East crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + ROAD - 18, half + 10 + i * (cwStripe + cwGap), 16, cwStripe)
  }

  // Stop lines
  ctx.fillStyle = 'rgba(244,239,230,0.7)'
  ctx.fillRect(half + 6, half - 6, ROAD / 2 - 12, 4)
  ctx.fillRect(half + ROAD / 2 + 6, half + ROAD + 2, ROAD / 2 - 12, 4)
  ctx.fillRect(half - 6, half + ROAD / 2 + 6, 4, ROAD / 2 - 12)
  ctx.fillRect(half + ROAD + 2, half + 6, 4, ROAD / 2 - 12)
}

function drawTrafficLight(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  straightColor: 'red' | 'yellow' | 'green',
  leftColor: 'red' | 'yellow' | 'green',
  vertical: boolean,
) {
  ctx.save()

  // Housing
  ctx.shadowColor = 'rgba(0,0,0,0.5)'
  ctx.shadowBlur = 12
  ctx.shadowOffsetY = 4
  ctx.fillStyle = '#0a0e13'

  if (vertical) {
    roundRect(ctx, x - 14, y - 44, 28, 88, 8)
  } else {
    roundRect(ctx, x - 44, y - 14, 88, 28, 8)
  }
  ctx.fill()
  ctx.restore()

  // Housing border
  ctx.strokeStyle = 'rgba(244,239,230,0.1)'
  ctx.lineWidth = 1
  if (vertical) {
    roundRect(ctx, x - 14, y - 44, 28, 88, 8)
  } else {
    roundRect(ctx, x - 44, y - 14, 88, 28, 8)
  }
  ctx.stroke()

  const colors = { red: '#ff5a5f', yellow: '#ffd166', green: '#2ec4b6' }

  // Left turn arrow (smaller, offset)
  const leftX = vertical ? x : x - 30
  const leftY = vertical ? y - 30 : y
  ctx.beginPath()
  ctx.fillStyle = leftColor === 'red' ? '#2a323c' : colors[leftColor]
  ctx.arc(leftX, leftY, 5, 0, Math.PI * 2)
  ctx.fill()
  if (leftColor !== 'red') {
    ctx.shadowColor = colors[leftColor]
    ctx.shadowBlur = 10
    ctx.fill()
    ctx.shadowBlur = 0
  }
  // Arrow indicator
  ctx.fillStyle = 'rgba(255,255,255,0.6)'
  ctx.font = 'bold 6px sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('←', leftX, leftY + 2)

  // Main lights
  const mainLights = [
    { color: 'red' as const, on: straightColor === 'red' },
    { color: 'yellow' as const, on: straightColor === 'yellow' },
    { color: 'green' as const, on: straightColor === 'green' },
  ]

  mainLights.forEach((light, i) => {
    const lx = vertical ? x : x - 10 + i * 20
    const ly = vertical ? y - 10 + i * 20 : y
    ctx.beginPath()
    ctx.fillStyle = light.on ? colors[light.color] : '#2a323c'
    ctx.arc(lx, ly, 7, 0, Math.PI * 2)
    ctx.fill()
    if (light.on) {
      ctx.shadowColor = colors[light.color]
      ctx.shadowBlur = 14
      ctx.fill()
      ctx.shadowBlur = 0
      // Highlight
      ctx.beginPath()
      ctx.fillStyle = 'rgba(255,255,255,0.35)'
      ctx.arc(lx - 2, ly - 2, 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
  })
}

function drawPedestrianSignal(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  signal: 'walk' | 'dont-walk' | 'flashing',
  pressed: boolean,
) {
  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.4)'
  ctx.shadowBlur = 8
  ctx.shadowOffsetY = 3
  ctx.fillStyle = '#0a0e13'
  roundRect(ctx, x - 10, y - 10, 20, 20, 4)
  ctx.fill()
  ctx.restore()

  ctx.strokeStyle = pressed ? 'rgba(240,162,2,0.6)' : 'rgba(244,239,230,0.1)'
  ctx.lineWidth = pressed ? 2 : 1
  roundRect(ctx, x - 10, y - 10, 20, 20, 4)
  ctx.stroke()

  if (signal === 'walk') {
    ctx.fillStyle = '#2ec4b6'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('🚶', x, y + 4)
  } else {
    ctx.fillStyle = '#ff5a5f'
    ctx.font = 'bold 10px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('✋', x, y + 4)
  }

  // Button indicator
  if (pressed) {
    ctx.beginPath()
    ctx.fillStyle = '#f0a202'
    ctx.arc(x + 12, y - 12, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowColor = '#f0a202'
    ctx.shadowBlur = 8
    ctx.fill()
    ctx.shadowBlur = 0
  }
}

function drawLights(ctx: CanvasRenderingContext2D, state: SimState) {
  const half = (SIZE - ROAD) / 2
  const { phase, crosswalkRequest } = state

  // NS lights (vertical orientation)
  const nsStraight = lightColor(phase, 'ns', 'straight')
  const nsLeft = lightColor(phase, 'ns', 'left')
  drawTrafficLight(ctx, half - 24, half - 16, nsStraight, nsLeft, true)
  drawTrafficLight(ctx, half + ROAD + 24, half + ROAD + 16, nsStraight, nsLeft, true)

  // EW lights (horizontal orientation)
  const ewStraight = lightColor(phase, 'ew', 'straight')
  const ewLeft = lightColor(phase, 'ew', 'left')
  drawTrafficLight(ctx, half + ROAD + 16, half - 24, ewStraight, ewLeft, false)
  drawTrafficLight(ctx, half - 16, half + ROAD + 24, ewStraight, ewLeft, false)

  // Pedestrian signals sit at the corner where those pedestrians wait.
  // North/south walkers cross the NS roadway with EW traffic; east/west
  // walkers cross the EW roadway with NS traffic.
  const nsPed = pedestrianSignal(phase, 'ns')
  const ewPed = pedestrianSignal(phase, 'ew')
  drawPedestrianSignal(ctx, half - 40, half - 40, ewPed, crosswalkRequest.ew)
  drawPedestrianSignal(ctx, half + ROAD + 40, half + ROAD + 40, ewPed, crosswalkRequest.ew)
  drawPedestrianSignal(ctx, half + ROAD + 40, half - 40, nsPed, crosswalkRequest.ns)
  drawPedestrianSignal(ctx, half - 40, half + ROAD + 40, nsPed, crosswalkRequest.ns)
}

function drawVehicle(ctx: CanvasRenderingContext2D, v: Vehicle) {
  const { x, y, angle } = vehiclePose(v)
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(angle)

  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.45)'
  ctx.shadowBlur = 6
  ctx.shadowOffsetY = 3

  if (v.kind === 'truck') {
    const grad = ctx.createLinearGradient(-20, -11, 20, 11)
    grad.addColorStop(0, v.waiting ? '#b45309' : '#f59e0b')
    grad.addColorStop(1, v.waiting ? '#7c2d12' : '#d97706')
    ctx.fillStyle = grad
    roundRect(ctx, -20, -11, 40, 22, 4)
    ctx.fill()
    ctx.strokeStyle = 'rgba(0,0,0,0.45)'
    ctx.lineWidth = 1.2
    ctx.stroke()
    ctx.restore()

    ctx.fillStyle = '#1f2937'
    roundRect(ctx, 10, -9, 10, 18, 2)
    ctx.fill()
    ctx.fillStyle = 'rgba(147,197,253,0.4)'
    roundRect(ctx, 12, -7, 6, 14, 1)
    ctx.fill()
  } else {
    const grad = ctx.createLinearGradient(-14, -8, 14, 8)
    grad.addColorStop(0, v.color)
    grad.addColorStop(1, shadeColor(v.color, -20))
    ctx.fillStyle = grad
    roundRect(ctx, -14, -8, 28, 16, 4)
    ctx.fill()
    ctx.strokeStyle = 'rgba(0,0,0,0.5)'
    ctx.lineWidth = 1.1
    ctx.stroke()
    ctx.restore()

    ctx.fillStyle = shadeColor(v.color, -30)
    roundRect(ctx, -6, -6, 12, 12, 3)
    ctx.fill()
    ctx.fillStyle = 'rgba(147,197,253,0.35)'
    roundRect(ctx, 4, -5, 5, 10, 1)
    ctx.fill()
  }

  if (v.waiting && v.intent !== 'straight') {
    ctx.fillStyle = v.intent === 'left' ? '#f0a202' : '#2ec4b6'
    ctx.beginPath()
    ctx.arc(v.intent === 'left' ? -10 : 10, 0, 3, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.restore()
}

function shadeColor(hex: string, percent: number): string {
  const num = parseInt(hex.replace('#', ''), 16)
  const amt = Math.round(2.55 * percent)
  const R = Math.min(255, Math.max(0, (num >> 16) + amt))
  const G = Math.min(255, Math.max(0, ((num >> 8) & 0xff) + amt))
  const B = Math.min(255, Math.max(0, (num & 0xff) + amt))
  return `#${((1 << 24) + (R << 16) + (G << 8) + B).toString(16).slice(1)}`
}

function drawPedestrian(ctx: CanvasRenderingContext2D, p: Pedestrian) {
  const half = (SIZE - ROAD) / 2
  let x = 0
  let y = 0
  const slot = p.id % 8
  const spread = (slot % 4) * 9
  const row = Math.floor(slot / 4) * 11
  const lane = (p.id % 3) - 1

  if (p.crossing) {
    // Crossing the road along the painted zebra band; offset so a group
    // doesn't occupy a single pixel.
    const t = p.progress
    switch (p.approach) {
      case 'north':
        x = half + 30 + t * (ROAD - 60)
        y = half + 10 + lane * 5
        break
      case 'south':
        x = half + 30 + t * (ROAD - 60)
        y = half + ROAD - 10 + lane * 5
        break
      case 'west':
        x = half + 10 + lane * 5
        y = half + 30 + t * (ROAD - 60)
        break
      case 'east':
        x = half + ROAD - 10 + lane * 5
        y = half + 30 + t * (ROAD - 60)
        break
    }
  } else {
    // Waiting on the sidewalk, fanned out from the inner corner.
    switch (p.approach) {
      case 'north':
        x = half - 22 - spread
        y = half - 22 - row
        break
      case 'south':
        x = half + ROAD + 22 + spread
        y = half + ROAD + 22 + row
        break
      case 'west':
        x = half - 22 - spread
        y = half + ROAD + 22 + row
        break
      case 'east':
        x = half + ROAD + 22 + spread
        y = half - 22 - row
        break
    }
  }

  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.3)'
  ctx.shadowBlur = 4
  ctx.shadowOffsetY = 2

  const grad = ctx.createRadialGradient(x - 1, y - 1, 0.5, x, y, 6)
  grad.addColorStop(0, p.crossing ? '#fff' : '#f4efe6')
  grad.addColorStop(1, p.crossing ? '#d9d0b4' : '#b8ae9a')
  ctx.fillStyle = grad
  ctx.beginPath()
  ctx.arc(x, y, 6, 0, Math.PI * 2)
  ctx.fill()
  ctx.strokeStyle = 'rgba(0,0,0,0.35)'
  ctx.lineWidth = 1
  ctx.stroke()
  ctx.restore()

  // Waiting indicator
  if (p.waiting) {
    ctx.beginPath()
    ctx.strokeStyle = 'rgba(240,162,2,0.6)'
    ctx.lineWidth = 2
    ctx.arc(x, y, 8, 0, Math.PI * 2)
    ctx.stroke()
  }
}

function paint(ctx: CanvasRenderingContext2D, state: SimState) {
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.globalAlpha = 1
  drawRoad(ctx)
  drawLights(ctx, state)

  // Draw vehicles sorted by approach for proper layering
  const sorted = [...state.vehicles].sort((a, b) => a.progress - b.progress)
  for (const v of sorted) {
    drawVehicle(ctx, v)
  }

  // Draw pedestrians
  for (const p of state.pedestrians) {
    drawPedestrian(ctx, p)
  }

  // Center status
  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.4)'
  ctx.shadowBlur = 12
  ctx.shadowOffsetY = 3
  ctx.fillStyle = state.jamActive
    ? 'rgba(124,45,18,0.85)'
    : state.pedestrianCrossing
      ? 'rgba(30,64,60,0.85)'
      : 'rgba(10,15,20,0.85)'
  roundRect(ctx, CENTER - 70, CENTER - 16, 140, 32, 8)
  ctx.fill()
  ctx.restore()

  ctx.strokeStyle = state.jamActive
    ? 'rgba(255,122,89,0.5)'
    : state.pedestrianCrossing
      ? 'rgba(46,196,182,0.5)'
      : 'rgba(244,239,230,0.15)'
  ctx.lineWidth = 1
  roundRect(ctx, CENTER - 70, CENTER - 16, 140, 32, 8)
  ctx.stroke()

  ctx.fillStyle = '#f4efe6'
  ctx.font = '600 11px "Instrument Sans", sans-serif'
  ctx.textAlign = 'center'
  const label = state.jamActive
    ? 'JAM RESPONSE'
    : state.pedestrianCrossing
      ? 'PEDESTRIANS'
      : 'LIVE'
  ctx.fillText(label, CENTER, CENTER + 4)
}

type Props = {
  state: SimState
}

export function IntersectionCanvas({ state }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useLayoutEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    paint(ctx, state)
  }, [state])

  return (
    <div className="intersection-frame">
      <canvas
        ref={ref}
        width={SIZE}
        height={SIZE}
        className="intersection-canvas"
        aria-label="Two-dimensional traffic intersection simulation"
      />
    </div>
  )
}
