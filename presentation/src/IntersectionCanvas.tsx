import { useEffect, useRef } from 'react'
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

function laneOffset(lane: Lane): number {
  if (lane === 'left') return -(LANE_W + LANE_GAP)
  if (lane === 'right') return LANE_W + LANE_GAP
  return 0
}

function approachOrigin(
  approach: Approach,
  progress: number,
  lane: Lane,
): { x: number; y: number; angle: number } {
  const stop = CENTER - ROAD / 2 - 10
  const travel = stop + Math.max(0, progress - 1) * (ROAD + 140)
  const approachDist = stop * (1 - Math.min(progress, 1))
  const offset = laneOffset(lane)

  switch (approach) {
    case 'north':
      return {
        x: CENTER - ROAD / 4 + offset,
        y: approachDist + (progress > 1 ? travel - stop : 0),
        angle: Math.PI / 2,
      }
    case 'south':
      return {
        x: CENTER + ROAD / 4 - offset,
        y: SIZE - approachDist - (progress > 1 ? travel - stop : 0),
        angle: -Math.PI / 2,
      }
    case 'west':
      return {
        x: approachDist + (progress > 1 ? travel - stop : 0),
        y: CENTER + ROAD / 4 - offset,
        angle: 0,
      }
    case 'east':
      return {
        x: SIZE - approachDist - (progress > 1 ? travel - stop : 0),
        y: CENTER - ROAD / 4 + offset,
        angle: Math.PI,
      }
  }
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

  // Corner blocks
  const corner = ctx.createLinearGradient(0, 0, half, half)
  corner.addColorStop(0, '#14251f')
  corner.addColorStop(1, '#0d1a16')
  ctx.fillStyle = corner
  ctx.fillRect(0, 0, half, half)
  ctx.fillRect(SIZE - half, 0, half, half)
  ctx.fillRect(0, SIZE - half, half, half)
  ctx.fillRect(SIZE - half, SIZE - half, half, half)

  // Sidewalk edges
  ctx.strokeStyle = 'rgba(244,239,230,0.08)'
  ctx.lineWidth = 2
  ctx.strokeRect(half - 4, half - 4, ROAD + 8, ROAD + 8)

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

  // Center box
  ctx.fillStyle = '#1e2933'
  ctx.fillRect(half, half, ROAD, ROAD)

  // Lane markings — 3 lanes each direction
  ctx.strokeStyle = 'rgba(232,224,200,0.45)'
  ctx.lineWidth = 2
  ctx.setLineDash([16, 14])

  // Vertical road lanes
  for (const dir of [-1, 1]) {
    const baseX = CENTER + dir * (ROAD / 4)
    for (const laneOff of [-LANE_W - LANE_GAP, 0, LANE_W + LANE_GAP]) {
      const x = baseX + laneOff
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, half)
      ctx.moveTo(x, half + ROAD)
      ctx.lineTo(x, SIZE)
      ctx.stroke()
    }
  }

  // Horizontal road lanes
  for (const dir of [-1, 1]) {
    const baseY = CENTER + dir * (ROAD / 4)
    for (const laneOff of [-LANE_W - LANE_GAP, 0, LANE_W + LANE_GAP]) {
      const y = baseY + laneOff
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(half, y)
      ctx.moveTo(half + ROAD, y)
      ctx.lineTo(SIZE, y)
      ctx.stroke()
    }
  }
  ctx.setLineDash([])

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

  // Crosswalks — zebra stripes
  ctx.fillStyle = 'rgba(244,239,230,0.75)'
  const cwStripe = 8
  const cwGap = 6

  // North crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + 10 + i * (cwStripe + cwGap), half - 20, cwStripe, 16)
  }
  // South crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + 10 + i * (cwStripe + cwGap), half + ROAD + 4, cwStripe, 16)
  }
  // West crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half - 20, half + 10 + i * (cwStripe + cwGap), 16, cwStripe)
  }
  // East crosswalk
  for (let i = 0; i < 8; i++) {
    ctx.fillRect(half + ROAD + 4, half + 10 + i * (cwStripe + cwGap), 16, cwStripe)
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

  // Pedestrian signals
  const nsPed = pedestrianSignal(phase, 'ns')
  const ewPed = pedestrianSignal(phase, 'ew')
  drawPedestrianSignal(ctx, half - 40, half - 40, nsPed, crosswalkRequest.ns)
  drawPedestrianSignal(ctx, half + ROAD + 40, half + ROAD + 40, nsPed, crosswalkRequest.ns)
  drawPedestrianSignal(ctx, half + ROAD + 40, half - 40, ewPed, crosswalkRequest.ew)
  drawPedestrianSignal(ctx, half - 40, half + ROAD + 40, ewPed, crosswalkRequest.ew)
}

function drawVehicle(ctx: CanvasRenderingContext2D, v: Vehicle) {
  const { x, y, angle } = approachOrigin(v.approach, v.progress, v.lane)
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(angle)

  // Shadow
  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.45)'
  ctx.shadowBlur = 6
  ctx.shadowOffsetY = 3

  if (v.kind === 'truck') {
    // Truck body
    const grad = ctx.createLinearGradient(-20, -11, 20, 11)
    grad.addColorStop(0, v.waiting ? '#b45309' : '#f59e0b')
    grad.addColorStop(1, v.waiting ? '#7c2d12' : '#d97706')
    ctx.fillStyle = grad
    roundRect(ctx, -20, -11, 40, 22, 4)
    ctx.fill()
    ctx.restore()
    ctx.save()
    // Cab
    ctx.fillStyle = '#1f2937'
    roundRect(ctx, 10, -9, 10, 18, 2)
    ctx.fill()
    // Windshield
    ctx.fillStyle = 'rgba(147,197,253,0.4)'
    roundRect(ctx, 12, -7, 6, 14, 1)
    ctx.fill()
  } else {
    // Car body
    const grad = ctx.createLinearGradient(-14, -8, 14, 8)
    grad.addColorStop(0, v.color)
    grad.addColorStop(1, shadeColor(v.color, -20))
    ctx.fillStyle = grad
    roundRect(ctx, -14, -8, 28, 16, 4)
    ctx.fill()
    ctx.restore()
    ctx.save()
    // Roof
    ctx.fillStyle = shadeColor(v.color, -30)
    roundRect(ctx, -6, -6, 12, 12, 3)
    ctx.fill()
    // Windshield
    ctx.fillStyle = 'rgba(147,197,253,0.35)'
    roundRect(ctx, 4, -5, 5, 10, 1)
    ctx.fill()
  }

  // Turn signal indicator
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
  let x: number, y: number

  if (p.crossing) {
    // Crossing the road
    const t = p.progress
    switch (p.approach) {
      case 'north':
        x = half + 30 + t * (ROAD - 60)
        y = half - 12
        break
      case 'south':
        x = half + 30 + t * (ROAD - 60)
        y = half + ROAD + 12
        break
      case 'west':
        x = half - 12
        y = half + 30 + t * (ROAD - 60)
        break
      case 'east':
        x = half + ROAD + 12
        y = half + 30 + t * (ROAD - 60)
        break
    }
  } else {
    // Waiting at corner
    switch (p.approach) {
      case 'north':
        x = half - 30
        y = half - 30
        break
      case 'south':
        x = half + ROAD + 30
        y = half + ROAD + 30
        break
      case 'west':
        x = half - 30
        y = half + ROAD + 30
        break
      case 'east':
        x = half + ROAD + 30
        y = half - 30
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
  ctx.arc(x, y, 5, 0, Math.PI * 2)
  ctx.fill()
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

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    paint(ctx, state)
  }, [state])

  return (
    <canvas
      ref={ref}
      width={SIZE}
      height={SIZE}
      className="intersection-canvas"
      aria-label="Two-dimensional traffic intersection simulation"
    />
  )
}
