import { useEffect, useRef } from 'react'
import {
  type Agent,
  type Approach,
  type Phase,
  lightColor,
  type SimState,
} from './simulation'

const SIZE = 720
const CENTER = SIZE / 2
const ROAD = 150
const LANE = 36

function approachOrigin(
  approach: Approach,
  progress: number,
  laneOffset = 0,
): { x: number; y: number; angle: number } {
  const stop = CENTER - ROAD / 2 - 8
  const travel = stop + Math.max(0, progress - 1) * (ROAD + 120)
  const approachDist = stop * (1 - Math.min(progress, 1))
  const offset = laneOffset * 14

  switch (approach) {
    case 'north':
      return {
        x: CENTER - LANE / 2 + offset,
        y: approachDist + (progress > 1 ? travel - stop : 0),
        angle: Math.PI / 2,
      }
    case 'south':
      return {
        x: CENTER + LANE / 2 + offset,
        y: SIZE - approachDist - (progress > 1 ? travel - stop : 0),
        angle: -Math.PI / 2,
      }
    case 'west':
      return {
        x: approachDist + (progress > 1 ? travel - stop : 0),
        y: CENTER + LANE / 2 + offset,
        angle: 0,
      }
    case 'east':
      return {
        x: SIZE - approachDist - (progress > 1 ? travel - stop : 0),
        y: CENTER - LANE / 2 + offset,
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
  // deep backdrop
  const bg = ctx.createLinearGradient(0, 0, 0, SIZE)
  bg.addColorStop(0, '#0b1117')
  bg.addColorStop(1, '#101a22')
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, SIZE, SIZE)

  const half = (SIZE - ROAD) / 2

  // corner blocks with soft depth
  const corner = ctx.createLinearGradient(0, 0, half, half)
  corner.addColorStop(0, '#16302a')
  corner.addColorStop(1, '#0f241f')
  ctx.fillStyle = corner
  ctx.fillRect(0, 0, half, half)
  ctx.fillRect(SIZE - half, 0, half, half)
  ctx.fillRect(0, SIZE - half, half, half)
  ctx.fillRect(SIZE - half, SIZE - half, half, half)

  // subtle texture on corners
  ctx.strokeStyle = 'rgba(46,196,182,0.05)'
  ctx.lineWidth = 1
  for (let i = 0; i < 6; i += 1) {
    ctx.beginPath()
    ctx.moveTo(0, 30 + i * 26)
    ctx.lineTo(half - 8, 30 + i * 26)
    ctx.stroke()
  }

  // road shadow (underlay)
  ctx.fillStyle = 'rgba(0,0,0,0.45)'
  ctx.fillRect(half - 6, 0, ROAD + 12, SIZE)
  ctx.fillRect(0, half - 6, SIZE, ROAD + 12)

  // asphalt with gradient
  const road = ctx.createLinearGradient(half, 0, half + ROAD, 0)
  road.addColorStop(0, '#1d2630')
  road.addColorStop(0.5, '#232f3b')
  road.addColorStop(1, '#1a232c')
  ctx.fillStyle = road
  ctx.fillRect(half, 0, ROAD, SIZE)

  const roadH = ctx.createLinearGradient(0, half, 0, half + ROAD)
  roadH.addColorStop(0, '#1d2630')
  roadH.addColorStop(0.5, '#232f3b')
  roadH.addColorStop(1, '#1a232c')
  ctx.fillStyle = roadH
  ctx.fillRect(0, half, SIZE, ROAD)

  // center box
  ctx.fillStyle = '#202b36'
  ctx.fillRect(half, half, ROAD, ROAD)

  // curbs
  ctx.strokeStyle = 'rgba(232,224,200,0.14)'
  ctx.lineWidth = 2
  ctx.strokeRect(half + 1, half + 1, ROAD - 2, ROAD - 2)

  // lane markings
  ctx.strokeStyle = 'rgba(232,224,200,0.5)'
  ctx.setLineDash([20, 18])
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(CENTER, 0)
  ctx.lineTo(CENTER, half)
  ctx.moveTo(CENTER, half + ROAD)
  ctx.lineTo(CENTER, SIZE)
  ctx.moveTo(0, CENTER)
  ctx.lineTo(half, CENTER)
  ctx.moveTo(half + ROAD, CENTER)
  ctx.lineTo(SIZE, CENTER)
  ctx.stroke()
  ctx.setLineDash([])

  // crosswalks — crisp bars
  ctx.fillStyle = 'rgba(244,239,230,0.72)'
  for (let i = 0; i < 6; i += 1) {
    const o = i * 14
    ctx.fillRect(half - 24, half + 18 + o, 20, 8)
    ctx.fillRect(half + ROAD + 4, half + 18 + o, 20, 8)
    ctx.fillRect(half + 18 + o, half - 24, 8, 20)
    ctx.fillRect(half + 18 + o, half + ROAD + 4, 8, 20)
  }

  // stop lines
  ctx.fillStyle = 'rgba(244,239,230,0.6)'
  ctx.fillRect(half + 8, half - 8, LANE, 4)
  ctx.fillRect(half + ROAD - LANE - 8, half + ROAD + 4, LANE, 4)
  ctx.fillRect(half - 8, half + ROAD - LANE - 8, 4, LANE)
  ctx.fillRect(half + ROAD + 4, half + 8, 4, LANE)
}

function drawLight(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: 'red' | 'yellow' | 'green',
) {
  // housing with rounded corners + drop shadow
  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.55)'
  ctx.shadowBlur = 14
  ctx.shadowOffsetY = 5
  ctx.fillStyle = '#0a0e13'
  roundRect(ctx, x - 11, y - 30, 22, 60, 6)
  ctx.fill()
  ctx.restore()

  // housing highlight
  ctx.strokeStyle = 'rgba(244,239,230,0.08)'
  ctx.lineWidth = 1
  roundRect(ctx, x - 11, y - 30, 22, 60, 6)
  ctx.stroke()

  const lamps = [
    { on: color === 'red', col: '#ff5a5f' },
    { on: color === 'yellow', col: '#ffd166' },
    { on: color === 'green', col: '#2ec4b6' },
  ] as const

  lamps.forEach((lamp, i) => {
    const cy = y - 17 + i * 17
    ctx.beginPath()
    ctx.fillStyle = lamp.on ? lamp.col : '#232b33'
    ctx.arc(x, cy, 6.5, 0, Math.PI * 2)
    ctx.fill()
    if (lamp.on) {
      ctx.shadowColor = lamp.col
      ctx.shadowBlur = 16
      ctx.fill()
      ctx.shadowBlur = 0
      // inner highlight
      ctx.beginPath()
      ctx.fillStyle = 'rgba(255,255,255,0.35)'
      ctx.arc(x - 2, cy - 2, 2.2, 0, Math.PI * 2)
      ctx.fill()
    }
  })
}

function drawLights(ctx: CanvasRenderingContext2D, phase: Phase) {
  const ns = lightColor(phase, 'ns')
  const ew = lightColor(phase, 'ew')
  const half = (SIZE - ROAD) / 2
  drawLight(ctx, half - 30, half - 12, ns)
  drawLight(ctx, half + ROAD + 30, half + ROAD + 12, ns)
  drawLight(ctx, half + ROAD + 12, half - 30, ew)
  drawLight(ctx, half - 12, half + ROAD + 30, ew)
}

function drawAgent(ctx: CanvasRenderingContext2D, agent: Agent, laneOffset: number) {
  const { x, y, angle } = approachOrigin(agent.approach, agent.progress, laneOffset)
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(angle)

  // soft shadow
  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.5)'
  ctx.shadowBlur = 8
  ctx.shadowOffsetY = 4

  if (agent.kind === 'pedestrian') {
    const grad = ctx.createRadialGradient(-1.5, -1.5, 0.5, 0, 0, 6)
    grad.addColorStop(0, agent.waiting ? '#ffe3a3' : '#ffffff')
    grad.addColorStop(1, agent.waiting ? '#f0a202' : '#d9d0b4')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(0, 0, 5.5, 0, Math.PI * 2)
    ctx.fill()
  } else if (agent.kind === 'truck') {
    const grad = ctx.createLinearGradient(-18, -10, 18, 10)
    grad.addColorStop(0, agent.waiting ? '#b45309' : '#f59e0b')
    grad.addColorStop(1, agent.waiting ? '#7c2d12' : '#d97706')
    ctx.fillStyle = grad
    roundRect(ctx, -18, -10, 36, 20, 4)
    ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.fillStyle = 'rgba(255,255,255,0.14)'
    roundRect(ctx, -16, -8, 32, 4, 2)
    ctx.fill()
    ctx.fillStyle = '#111827'
    roundRect(ctx, 8, -7, 9, 14, 2)
    ctx.fill()
  } else {
    const grad = ctx.createLinearGradient(-12, -7, 12, 7)
    grad.addColorStop(0, agent.waiting ? '#64748b' : '#7dd3fc')
    grad.addColorStop(1, agent.waiting ? '#334155' : '#38bdf8')
    ctx.fillStyle = grad
    roundRect(ctx, -12, -7, 24, 14, 4)
    ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.fillStyle = 'rgba(255,255,255,0.2)'
    roundRect(ctx, -10, -5, 20, 3, 1.5)
    ctx.fill()
    ctx.fillStyle = '#0b1220'
    roundRect(ctx, 4, -4, 7, 8, 2)
    ctx.fill()
  }
  ctx.restore()
}

function paint(ctx: CanvasRenderingContext2D, state: SimState) {
  drawRoad(ctx)
  drawLights(ctx, state.phase)
  const laneCursor: Record<Approach, number> = { north: 0, south: 0, east: 0, west: 0 }
  for (const agent of state.agents) {
    const offsetIndex = laneCursor[agent.approach]
    laneCursor[agent.approach] += 1
    const laneOffset = (offsetIndex % 3) - 1
    drawAgent(ctx, agent, agent.kind === 'pedestrian' ? laneOffset * 1.4 : laneOffset * 0.55)
  }

  // center status plaque — glassy
  ctx.save()
  ctx.shadowColor = 'rgba(0,0,0,0.4)'
  ctx.shadowBlur = 16
  ctx.shadowOffsetY = 4
  ctx.fillStyle = state.jamActive ? 'rgba(124,45,18,0.82)' : 'rgba(13,20,27,0.82)'
  roundRect(ctx, CENTER - 78, CENTER - 17, 156, 34, 9)
  ctx.fill()
  ctx.restore()
  ctx.strokeStyle = state.jamActive ? 'rgba(255,122,89,0.5)' : 'rgba(46,196,182,0.35)'
  ctx.lineWidth = 1
  roundRect(ctx, CENTER - 78, CENTER - 17, 156, 34, 9)
  ctx.stroke()
  ctx.fillStyle = '#f4efe6'
  ctx.font = '600 12px "Instrument Sans", sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText(state.jamActive ? 'JAM RESPONSE' : 'LIVE SIMULATION', CENTER, CENTER + 4)
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
