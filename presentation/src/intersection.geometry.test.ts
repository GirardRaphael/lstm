import { describe, expect, it } from 'vitest'
import type { Approach } from './simulation'
import {
  CENTER,
  exitPoint,
  ROAD,
  SIZE,
  vehiclePose,
} from './intersectionGeometry'

const APPROACHES: Approach[] = ['north', 'south', 'east', 'west']
const INTENTS = ['left', 'straight', 'right'] as const
const HALF = (SIZE - ROAD) / 2

function dummyPose(
  approach: Approach,
  intent: (typeof INTENTS)[number],
  progress: number,
) {
  return vehiclePose({ approach, lane: intent, intent, progress })
}

/**
 * Right-hand traffic destination half-planes (canvas Y-down):
 *   west  straight → eastbound  (south half, y > CENTER)
 *   west  right    → southbound (west half,  x < CENTER)
 *   west  left     → northbound (east half,  x > CENTER)
 *   east  straight → westbound  (north half, y < CENTER)
 *   east  right    → northbound (east half,  x > CENTER)
 *   east  left     → southbound (west half,  x < CENTER)
 *   north straight → southbound (west half,  x < CENTER)
 *   north right    → westbound  (north half, y < CENTER)
 *   north left     → eastbound  (south half, y > CENTER)
 *   south straight → northbound (east half,  x > CENTER)
 *   south right    → eastbound  (south half, y > CENTER)
 *   south left     → westbound  (north half, y < CENTER)
 */
function expectExitHalfPlane(
  approach: Approach,
  intent: (typeof INTENTS)[number],
  point: { x: number; y: number },
) {
  const label = `${approach} ${intent} → (${point.x.toFixed(1)}, ${point.y.toFixed(1)})`
  switch (approach) {
    case 'west':
      if (intent === 'straight') expect(point.y, label).toBeGreaterThan(CENTER)
      else if (intent === 'right') expect(point.x, label).toBeLessThan(CENTER)
      else expect(point.x, label).toBeGreaterThan(CENTER)
      break
    case 'east':
      if (intent === 'straight') expect(point.y, label).toBeLessThan(CENTER)
      else if (intent === 'right') expect(point.x, label).toBeGreaterThan(CENTER)
      else expect(point.x, label).toBeLessThan(CENTER)
      break
    case 'north':
      if (intent === 'straight') expect(point.x, label).toBeLessThan(CENTER)
      else if (intent === 'right') expect(point.y, label).toBeLessThan(CENTER)
      else expect(point.y, label).toBeGreaterThan(CENTER)
      break
    case 'south':
      if (intent === 'straight') expect(point.x, label).toBeGreaterThan(CENTER)
      else if (intent === 'right') expect(point.y, label).toBeGreaterThan(CENTER)
      else expect(point.y, label).toBeLessThan(CENTER)
      break
  }
}

function outsideBox(p: { x: number; y: number }): boolean {
  return p.x < HALF || p.x > HALF + ROAD || p.y < HALF || p.y > HALF + ROAD
}

describe('right-hand-traffic exit lanes', () => {
  it('lands every (approach, intent) exitPoint in the correct half-plane', () => {
    for (const approach of APPROACHES) {
      for (const intent of INTENTS) {
        const point = exitPoint(approach, intent, intent)
        expectExitHalfPlane(approach, intent, point)
      }
    }
  })

  it('west right-turn and east left-turn both enter the southbound (west) half', () => {
    const westRight = exitPoint('west', 'right', 'right')
    const eastLeft = exitPoint('east', 'left', 'left')
    expect(westRight.x).toBeLessThan(CENTER)
    expect(eastLeft.x).toBeLessThan(CENTER)
    expect(westRight.y).toBeGreaterThan(CENTER)
    expect(eastLeft.y).toBeGreaterThan(CENTER)
    expect(westRight.x).toBeGreaterThan(HALF - 40)
    expect(eastLeft.x).toBeGreaterThan(HALF - 40)
  })

  it('west left-turn and east right-turn both enter the northbound (east) half', () => {
    const westLeft = exitPoint('west', 'left', 'left')
    const eastRight = exitPoint('east', 'right', 'right')
    expect(westLeft.x).toBeGreaterThan(CENTER)
    expect(eastRight.x).toBeGreaterThan(CENTER)
    expect(westLeft.y).toBeLessThan(CENTER)
    expect(eastRight.y).toBeLessThan(CENTER)
  })

  it('vehiclePose after the box stays on the destination half-plane and heading', () => {
    const westRightExit = dummyPose('west', 'right', 2)
    const westRightDown = dummyPose('west', 'right', 2.35)
    expect(westRightExit.x).toBeLessThan(CENTER)
    expect(westRightDown.x).toBeLessThan(CENTER)
    expect(westRightDown.y).toBeGreaterThan(westRightExit.y)

    const eastLeftExit = dummyPose('east', 'left', 2)
    const eastLeftDown = dummyPose('east', 'left', 2.35)
    expect(eastLeftExit.x).toBeLessThan(CENTER)
    expect(eastLeftDown.x).toBeLessThan(CENTER)
    expect(eastLeftDown.y).toBeGreaterThan(eastLeftExit.y)

    for (const approach of APPROACHES) {
      for (const intent of INTENTS) {
        const atExit = dummyPose(approach, intent, 2)
        const further = dummyPose(approach, intent, 2.3)
        expectExitHalfPlane(approach, intent, atExit)
        expectExitHalfPlane(approach, intent, further)
        const lateCurve = dummyPose(approach, intent, 1.85)
        expectExitHalfPlane(approach, intent, lateCurve)
      }
    }
  })

  it('turn curves do not enter the oncoming half once they leave the box', () => {
    for (const approach of APPROACHES) {
      for (const intent of INTENTS) {
        if (intent === 'straight') continue
        for (let s = 7; s <= 10; s += 1) {
          const pt = dummyPose(approach, intent, 1 + s / 10)
          if (!outsideBox(pt) && s < 10) continue
          expectExitHalfPlane(approach, intent, pt)
        }
      }
    }
  })
})
