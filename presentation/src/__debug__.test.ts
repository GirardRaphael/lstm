import { writeFileSync } from 'node:fs'
import { it } from 'vitest'
import {
  createInitialState,
  spawnAgents,
  createTrafficJam,
  stepSimulation,
} from './simulation'

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

it('debug east:left compression', () => {
  const rng = mulberry32(1234)
  let state = createInitialState()
  state = spawnAgents(state, { cars: 14, trucks: 4, pedestrians: 3 }, rng)
  state = createTrafficJam(state, 'east', rng)
  const lines: string[] = []
  for (let i = 0; i <= 596; i += 1) {
    const prev = state
    state = stepSimulation(state, 0.1, rng)
    if (i >= 585) {
      const fmt = (v: { id: number; kind: string; intent: string; progress: number; speed: number; crossed: boolean; waiting: boolean }) =>
        `#${v.id}${v.kind[0]}${v.intent[0]} p=${v.progress.toFixed(4)} sp=${v.speed.toFixed(4)}${v.crossed ? ' X' : ''}${v.waiting ? ' W' : ''}`
      const before = prev.vehicles
        .filter((v) => v.approach === 'east' && v.lane === 'left')
        .sort((a, b) => a.progress - b.progress)
      const after = state.vehicles
        .filter((v) => v.approach === 'east' && v.lane === 'left')
        .sort((a, b) => a.progress - b.progress)
      lines.push(`tick ${i} phase(prev)=${prev.phase}`)
      lines.push(`  before: ${before.map(fmt).join(' | ')}`)
      lines.push(`  after:  ${after.map(fmt).join(' | ')}`)
    }
  }
  writeFileSync('/tmp/debug-lane.txt', lines.join('\n'))
})
