import { describe, it, expect } from 'vitest'
import {
  createInitialState,
  spawnAgents,
  createTrafficJam,
  stepSimulation,
  type SimState,
  type Vehicle,
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

function laneGroups(vehicles: Vehicle[]): Map<string, number[]> {
  const groups = new Map<string, number[]>()
  for (const v of vehicles) {
    const key = `${v.approach}:${v.lane}`
    const arr = groups.get(key) ?? []
    arr.push(v.progress)
    groups.set(key, arr)
  }
  return groups
}

function expectNoOverlap(state: SimState, minGap: number, ctx: string) {
  for (const [key, progresses] of laneGroups(state.vehicles)) {
    const sorted = [...progresses].sort((a, b) => a - b)
    for (let i = 1; i < sorted.length; i += 1) {
      expect(
        sorted[i] - sorted[i - 1],
        `${ctx}: overlap in ${key} at ${sorted[i - 1].toFixed(3)} / ${sorted[i].toFixed(3)}`,
      ).toBeGreaterThanOrEqual(minGap)
    }
  }
}

describe('traffic-flow fix verification', () => {
  it('spawns vehicles along the approach without same-lane overlap', () => {
    for (let seed = 1; seed <= 25; seed += 1) {
      const rng = mulberry32(seed * 42 + 1)
      let state = createInitialState()
      state = spawnAgents(state, { cars: 18, trucks: 5, pedestrians: 2 }, rng)
      expect(state.vehicles.length).toBe(23)
      for (const v of state.vehicles) {
        // distributed along the approach, never past the stop line region
        expect(v.progress, `seed ${seed} spawn above the box`).toBeLessThan(0.85)
        expect(v.progress, `seed ${seed} spawn unreasonably far back`).toBeGreaterThan(-3)
      }
      // spawn gap is 0.09; allow small tolerance
      expectNoOverlap(state, 0.08, `seed ${seed} spawn`)
    }
  })

  it('traffic jam packs a stopped queue backward from the stop line', () => {
    const rng = mulberry32(7)
    let state = createInitialState()
    state = createTrafficJam(state, 'east', rng)
    expect(state.vehicles.length).toBeGreaterThanOrEqual(12)
    for (const v of state.vehicles) {
      expect(v.progress).toBeLessThanOrEqual(0.951)
      expect(v.progress).toBeGreaterThan(0)
      expect(v.waiting).toBe(true)
    }
    // lead vehicle of each lane sits just behind the stop line (0.97)
    for (const [, progresses] of laneGroups(state.vehicles)) {
      expect(Math.max(...progresses)).toBeCloseTo(0.95, 5)
    }
    expectNoOverlap(state, 0.07, 'jam')
  })

  it('vehicles enter, cross, and exit over time without overlap or NaN', { timeout: 60000 }, () => {
    for (let seed = 1; seed <= 25; seed += 1) {
      const rng = mulberry32(seed * 1234 + 7)
      let state = createInitialState()
      state = spawnAgents(state, { cars: 14, trucks: 4, pedestrians: 3 }, rng)
      state = createTrafficJam(state, 'east', rng)

      let sawCrossing = false
      for (let i = 0; i < 1200; i += 1) {
        state = stepSimulation(state, 0.1, rng)
        for (const v of state.vehicles) {
          expect(Number.isNaN(v.progress)).toBe(false)
          expect(v.progress).toBeLessThan(3.4)
        }
        // no same-lane overlap once everyone is past the spawn region
        expectNoOverlap(
          { ...state, vehicles: state.vehicles.filter((v) => v.progress > -0.5) },
          0.04,
          `seed ${seed} tick ${i}`,
        )
        if (state.throughput > 0) sawCrossing = true
      }
      expect(sawCrossing, `seed ${seed}: vehicles must cross the intersection`).toBe(true)
      expect(state.throughput).toBeGreaterThan(5)
    }
  })

  it('red light holds uncrossed vehicles at the stop line', () => {
    const rng = mulberry32(99)
    let state = createInitialState()
    state = spawnAgents(state, { cars: 16, trucks: 4, pedestrians: 0 }, rng)
    for (let i = 0; i < 800; i += 1) {
      state = stepSimulation(state, 0.1, rng)
      for (const v of state.vehicles) {
        if (!v.crossed) {
          expect(
            v.progress,
            `uncrossed ${v.approach}/${v.lane} passed stop line during ${state.phase}`,
          ).toBeLessThanOrEqual(1.06)
        }
      }
    }
  })
})
