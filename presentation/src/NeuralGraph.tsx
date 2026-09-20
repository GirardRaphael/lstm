import { useEffect, useRef } from 'react'
import type { DecisionThought } from './decisionBrain'

type Props = {
  thought: DecisionThought
}

type Node = {
  id: string
  layer: number
  x: number
  y: number
  activation: number
  label?: string
}

type Edge = {
  from: Node
  to: Node
  weight: number
}

const LAYERS = [
  { id: 'input', label: 'Input', count: 6, x: 0.06 },
  { id: 'lstm1', label: 'LSTM 64', count: 8, x: 0.24 },
  { id: 'lstm2', label: 'LSTM 32', count: 8, x: 0.46 },
  { id: 'dense', label: 'Dense 16', count: 6, x: 0.68 },
  { id: 'output', label: 'Forecast', count: 1, x: 0.88 },
]

function buildGraph(thought: DecisionThought): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const intensity = thought.intensity

  // Input layer — queue pressure per approach
  const inputActs = [
    thought.congestedQueue / 30,
    intensity * 0.8,
    intensity * 0.6,
    intensity * 0.4,
    intensity * 0.3,
    intensity * 0.2,
  ]
  LAYERS.forEach((layer, li) => {
    for (let i = 0; i < layer.count; i += 1) {
      let activation = 0
      if (li === 0) {
        activation = inputActs[i] ?? 0
      } else if (li === 1) {
        // LSTM-1: blend of input + gate opening
        const gateAvg = (thought.gates.i + thought.gates.f) / 2
        activation = Math.sin(i * 1.3 + intensity * 5) * 0.3 + gateAvg * 0.7
      } else if (li === 2) {
        // LSTM-2: use real neuron data
        const n = thought.neurons[i % thought.neurons.length] ?? 0
        activation = Math.abs(n)
      } else if (li === 3) {
        // Dense
        const d = thought.dense[i % thought.dense.length] ?? 0
        activation = Math.max(0, d)
      } else {
        // Output
        activation = intensity
      }
      const ySpread = 0.72
      const yStart = 0.14
      const yStep = layer.count > 1 ? ySpread / (layer.count - 1) : 0
      nodes.push({
        id: `${layer.id}-${i}`,
        layer: li,
        x: layer.x,
        y: layer.count > 1 ? yStart + i * yStep : 0.5,
        activation: Math.max(0, Math.min(1, activation)),
        label: li === 4 ? `${Math.round(thought.forecastVehicles)}` : undefined,
      })
    }
  })

  // Edges between adjacent layers
  const edges: Edge[] = []
  for (let li = 0; li < LAYERS.length - 1; li += 1) {
    const fromNodes = nodes.filter((n) => n.layer === li)
    const toNodes = nodes.filter((n) => n.layer === li + 1)
    for (const from of fromNodes) {
      for (const to of toNodes) {
        // Weight: product of activations + deterministic pseudo-random
        const seed = Math.sin(from.x * 100 + to.y * 200) * 0.5 + 0.5
        const weight = from.activation * to.activation * (0.3 + seed * 0.7)
        edges.push({ from, to, weight })
      }
    }
  }

  return { nodes, edges }
}

function activationGlow(v: number): string {
  if (v > 0.6) return '#2ec4b6'
  if (v > 0.3) return '#f0a202'
  return '#3a4a5a'
}

export function NeuralGraph({ thought }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef(0)
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = 560
    const H = 340
    canvas.width = W * 2
    canvas.height = H * 2
    ctx.scale(2, 2)

    const { nodes, edges } = buildGraph(thought)

    const draw = () => {
      timeRef.current += 0.016
      const t = timeRef.current

      ctx.clearRect(0, 0, W, H)

      // Background
      ctx.fillStyle = 'rgba(11, 17, 23, 0.6)'
      ctx.fillRect(0, 0, W, H)

      // Layer labels
      ctx.font = '500 9px "IBM Plex Mono", monospace'
      ctx.textAlign = 'center'
      LAYERS.forEach((layer) => {
        ctx.fillStyle = 'rgba(143, 163, 184, 0.6)'
        ctx.fillText(layer.label, layer.x * W, 14)
      })

      // Edges — draw dim first, then active on top
      for (const edge of edges) {
        const x1 = edge.from.x * W
        const y1 = edge.from.y * H
        const x2 = edge.to.x * W
        const y2 = edge.to.y * H

        // Dim base connection
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.strokeStyle = 'rgba(58, 74, 90, 0.15)'
        ctx.lineWidth = 0.5
        ctx.stroke()

        // Active connection with pulse
        if (edge.weight > 0.08) {
          const pulse = Math.sin(t * 3 + x1 * 0.01 + y1 * 0.02) * 0.5 + 0.5
          const alpha = edge.weight * (0.3 + pulse * 0.7)
          const color = edge.weight > 0.3 ? '46, 196, 182' : '240, 162, 2'

          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.strokeStyle = `rgba(${color}, ${alpha * 0.6})`
          ctx.lineWidth = 0.8 + edge.weight * 1.5
          ctx.stroke()

          // Traveling pulse dot on strong connections
          if (edge.weight > 0.25) {
            const progress = (t * 0.8 + x1 * 0.005) % 1
            const px = x1 + (x2 - x1) * progress
            const py = y1 + (y2 - y1) * progress
            ctx.beginPath()
            ctx.fillStyle = `rgba(46, 196, 182, ${0.6 * pulse})`
            ctx.arc(px, py, 1.5, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }

      // Nodes
      for (const node of nodes) {
        const x = node.x * W
        const y = node.y * H
        const r = 4 + node.activation * 5
        const pulse = Math.sin(t * 2.5 + node.y * 10) * 0.15 + 0.85
        const glow = node.activation * pulse

        // Outer glow
        if (glow > 0.15) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 3)
          const c = node.activation > 0.5 ? '46, 196, 182' : '240, 162, 2'
          gradient.addColorStop(0, `rgba(${c}, ${glow * 0.4})`)
          gradient.addColorStop(1, `rgba(${c}, 0)`)
          ctx.beginPath()
          ctx.fillStyle = gradient
          ctx.arc(x, y, r * 3, 0, Math.PI * 2)
          ctx.fill()
        }

        // Core
        ctx.beginPath()
        ctx.fillStyle = activationGlow(node.activation)
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fill()

        // Inner highlight
        ctx.beginPath()
        ctx.fillStyle = `rgba(255, 255, 255, ${0.2 + node.activation * 0.3})`
        ctx.arc(x - r * 0.25, y - r * 0.25, r * 0.35, 0, Math.PI * 2)
        ctx.fill()

        // Output label
        if (node.label) {
          ctx.font = '600 10px "Instrument Sans", sans-serif'
          ctx.fillStyle = '#f4efe6'
          ctx.fillText(node.label, x, y - r - 6)
        }
      }

      animRef.current = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animRef.current)
  }, [thought])

  return (
    <div className="neural-graph-container">
      <canvas ref={canvasRef} className="neural-graph" />
    </div>
  )
}
