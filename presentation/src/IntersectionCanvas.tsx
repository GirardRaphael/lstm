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

function approachOrigin(approach: Approach, progress: number): { x: number; y: number; angle: number } {
  // progress 0 far away, 1 at center stop, >1 past center
  const stop = CENTER - ROAD / 2 - 8
  const travel = stop + Math.max(0, progress - 1) * (ROAD + 120)
  const approachDist = stop * (1 - Math.min(progress, 1))

  switch (approach) {
    case 'north':
      return { x: CENTER - LANE / 2, y: approachDist + (progress > 1 ? travel - stop : 0), angle: Math.PI / 2 }
    case 'south':
      return { x: CENTER + LANE / 2, y: SIZE - approachDist - (progress > 1 ? travel - stop : 0), angle: -Math.PI / 2 }
    case 'west':
      return { x: approachDist + (progress > 1 ? travel - stop : 0), y: CENTER + LANE / 2, angle: 0 }
    case 'east':
      return { x: SIZE - approachDist - (progress > 1 ? travel - stop : 0), y: CENTER - LANE / 2, angle: Math.PI }
  }
}

function drawRoad(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = '#1a2330'
  ctx.fillRect(0, 0, SIZE, SIZE)

  // grass corners
  ctx.fillStyle = '#1c3a32'
  ctx.fillRect(0, 0, (SIZE - ROAD) / 2, (SIZE - ROAD) / 2)
  ctx.fillRect((SIZE + ROAD) / 2, 0, (SIZE - ROAD) / 2, (SIZE - ROAD) / 2)
  ctx.fillRect(0, (SIZE + ROAD) / 2, (SIZE - ROAD) / 2, (SIZE - ROAD) / 2)
  ctx.fillRect((SIZE + ROAD) / 2, (SIZE + ROAD) / 2, (SIZE - ROAD) / 2, (SIZE - ROAD) / 2)

  // roads
  ctx.fillStyle = '#2a3545'
  ctx.fillRect((SIZE - ROAD) / 2, 0, ROAD, SIZE)
  ctx.fillRect(0, (SIZE - ROAD) / 2, SIZE, ROAD)

  // lane markings
  ctx.strokeStyle = '#d9d0b4'
  ctx.setLineDash([18, 16])
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(CENTER, 0)
  ctx.lineTo(CENTER, (SIZE - ROAD) / 2)
  ctx.moveTo(CENTER, (SIZE + ROAD) / 2)
  ctx.lineTo(CENTER, SIZE)
  ctx.moveTo(0, CENTER)
  ctx.lineTo((SIZE - ROAD) / 2, CENTER)
  ctx.moveTo((SIZE + ROAD) / 2, CENTER)
  ctx.lineTo(SIZE, CENTER)
  ctx.stroke()
  ctx.setLineDash([])

  // crosswalks
  ctx.fillStyle = 'rgba(244,239,230,0.55)'
  for (let i = 0; i < 6; i += 1) {
    const o = i * 14
    ctx.fillRect((SIZE - ROAD) / 2 - 22, (SIZE - ROAD) / 2 + 20 + o, 18, 8)
    ctx.fillRect((SIZE + ROAD) / 2 + 4, (SIZE - ROAD) / 2 + 20 + o, 18, 8)
    ctx.fillRect((SIZE - ROAD) / 2 + 20 + o, (SIZE - ROAD) / 2 - 22, 8, 18)
    ctx.fillRect((SIZE - ROAD) / 2 + 20 + o, (SIZE + ROAD) / 2 + 4, 8, 18)
  }
}

function drawLight(ctx: CanvasRenderingContext2D, x: number, y: number, color: 'red' | 'yellow' | 'green') {
  ctx.fillStyle = '#0b0f14'
  ctx.fillRect(x - 10, y - 28, 20, 56)
  const lamps = [
    { c: 'red', on: color === 'red', col: '#e63946' },
    { c: 'yellow', on: color === 'yellow', col: '#ffd166' },
    { c: 'green', on: color === 'green', col: '#2ec4b6' },
  ] as const
  lamps.forEach((lamp, i) => {
    ctx.beginPath()
    ctx.fillStyle = lamp.on ? lamp.col : '#2a323c'
    ctx.arc(x, y - 16 + i * 16, 6, 0, Math.PI * 2)
    ctx.fill()
    if (lamp.on) {
      ctx.shadowColor = lamp.col
      ctx.shadowBlur = 12
      ctx.fill()
      ctx.shadowBlur = 0
    }
  })
}

function drawLights(ctx: CanvasRenderingContext2D, phase: Phase) {
  const ns = lightColor(phase, 'ns')
  const ew = lightColor(phase, 'ew')
  drawLight(ctx, (SIZE - ROAD) / 2 - 28, (SIZE - ROAD) / 2 - 10, ns)
  drawLight(ctx, (SIZE + ROAD) / 2 + 28, (SIZE + ROAD) / 2 + 10, ns)
  drawLight(ctx, (SIZE + ROAD) / 2 + 10, (SIZE - ROAD) / 2 - 28, ew)
  drawLight(ctx, (SIZE - ROAD) / 2 - 10, (SIZE + ROAD) / 2 + 28, ew)
}

function drawAgent(ctx: CanvasRenderingContext2D, agent: Agent) {
  const { x, y, angle } = approachOrigin(agent.approach, agent.progress)
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(angle)

  if (agent.kind === 'pedestrian') {
    ctx.fillStyle = agent.waiting ? '#ffd166' : '#f4efe6'
    ctx.beginPath()
    ctx.arc(0, 0, 5, 0, Math.PI * 2)
    ctx.fill()
  } else if (agent.kind === 'truck') {
    ctx.fillStyle = agent.waiting ? '#c45c26' : '#d97706'
    ctx.fillRect(-18, -10, 36, 20)
    ctx.fillStyle = '#1f2937'
    ctx.fillRect(8, -8, 10, 16)
  } else {
    ctx.fillStyle = agent.waiting ? '#94a3b8' : '#38bdf8'
    ctx.fillRect(-12, -7, 24, 14)
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(4, -5, 7, 10)
  }
  ctx.restore()
}

function paint(ctx: CanvasRenderingContext2D, state: SimState) {
  drawRoad(ctx)
  drawLights(ctx, state.phase)
  for (const agent of state.agents) {
    drawAgent(ctx, agent)
  }

  // center plaque
  ctx.fillStyle = 'rgba(14,20,27,0.72)'
  ctx.fillRect(CENTER - 70, CENTER - 18, 140, 36)
  ctx.fillStyle = '#f4efe6'
  ctx.font = '600 13px "Instrument Sans", sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText(state.jamActive ? 'JAM RESPONSE' : 'SIMULATION', CENTER, CENTER + 5)
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
