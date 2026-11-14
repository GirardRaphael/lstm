/// <reference types="node" />
import { createCanvas } from '@napi-rs/canvas'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { paintIntersection } from './IntersectionCanvas'
import { analyzeScreenshot } from './screenshotVision'
import { createInitialState, type Approach, type Pedestrian, type SimState, type Vehicle } from './simulation'

const approaches: Approach[] = ['north', 'south', 'east', 'west']
const palette = ['#e74c3c', '#3498db', '#f39c12', '#ecf0f1', '#9b59b6', '#1abc9c']
function vehicle(approach: Approach, options: Partial<Vehicle> = {}): Vehicle {
  return { id: 1, approach, kind: 'car', color: '#3498db', lane: 'straight', intent: 'straight',
    waiting: false, crossed: false, speed: 0.2, progress: 0.6, ...options }
}
function person(approach: Approach, id: number, options: Partial<Pedestrian> = {}): Pedestrian {
  return { id, approach, crossing: false, waiting: true, done: false, progress: 0, waitTime: 0, ...options }
}
function render(vehicles: Vehicle[] = [], pedestrians: Pedestrian[] = [], overrides: Partial<SimState> = {}) {
  const state = { ...createInitialState(), vehicles, pedestrians, ...overrides }
  const canvas = createCanvas(800, 800)
  const context = canvas.getContext('2d')
  paintIntersection(context as unknown as CanvasRenderingContext2D, state)
  const pixels = context.getImageData(0, 0, 800, 800)
  return { canvas, result: analyzeScreenshot(pixels, 1) }
}

describe('recognition of actual intersection renderer screenshots', () => {
  it('accepts an empty rendered intersection and ignores all signal states and markings', () => {
    for (const phase of ['all-red', 'ns-green', 'ew-green', 'ns-left-green', 'pedestrian-crossing'] as const) {
      const { result } = render([], [], { phase })
      expect(result.valid, phase).toBe(true)
      expect(result.detections, phase).toHaveLength(0)
      expect(result.boxOccupied, phase).toBe(false)
    }
  })
  it('recognizes each car paint color on every approach without reading renderer objects', () => {
    for (const approach of approaches) for (const color of palette) {
      const { result } = render([vehicle(approach, { color })])
      expect(result.observation.queues[approach], `${approach} ${color}`).toBe(1)
      expect(result.observation.cars, `${approach} ${color}`).toBe(1)
      expect(result.observation.trucks, `${approach} ${color}`).toBe(0)
    }
  })
  it('recognizes moving and waiting trucks on every approach', () => {
    for (const approach of approaches) for (const waiting of [true, false]) {
      const { result } = render([vehicle(approach, { kind: 'truck', waiting })])
      expect(result.observation.queues[approach], `${approach} waiting=${waiting}`).toBe(2)
      expect(result.observation.trucks, `${approach} waiting=${waiting}`).toBe(1)
      expect(result.boxOccupied).toBe(false)
    }
  })
  it('counts the painted left lanes and keeps legal stopped truck noses out of clearance', () => {
    for (const approach of approaches) {
      const { result } = render([vehicle(approach, { kind: 'truck', waiting: true, lane: 'left', intent: 'left', progress: 0.97 })])
      expect(result.observation.leftQueues[approach], approach).toBe(2)
      expect(result.boxOccupied, approach).toBe(false)
    }
  })
  it('recognizes people on all four sidewalk corners, including those drawn over signal housings', () => {
    const people = approaches.flatMap((approach, i) => Array.from({ length: 3 }, (_, k) => person(approach, i * 3 + k + 1)))
    const { result } = render([], people)
    expect(result.observation.pedestrians).toBe(12)
    expect(result.crossingPedestrians).toBe(0)
    expect(result.observation.cars).toBe(0)
    expect(result.boxOccupied).toBe(false)
  })
  it('detects a visible crossing person on every zebra', () => {
    for (const approach of approaches) {
      const { result } = render([], [person(approach, 1, { crossing: true, waiting: false, progress: 0.5 })])
      expect(result.crossingPedestrians, approach).toBe(1)
      expect(result.observation.pedestrians).toBe(0)
    }
  })
  it('holds clearance for turning and straight vehicles in the junction', () => {
    for (const approach of approaches) for (const lane of ['left', 'straight', 'right'] as const) {
      const { result } = render([vehicle(approach, { lane, intent: lane, crossed: true, progress: 1.4 })])
      expect(result.boxOccupied, `${approach} ${lane}`).toBe(true)
    }
  })
  it('sees crossing traffic immediately after the line, including tight right turns', () => {
    for (const approach of approaches) for (const lane of ['left', 'straight', 'right'] as const) {
      const { result } = render([vehicle(approach, { lane, intent: lane, crossed: true, progress: 1.05 })])
      expect(result.boxOccupied, `${approach} ${lane}`).toBe(true)
    }
  })
  it('keeps a visible crossing crowd protected even when its sprites overlap', () => {
    for (const progress of [0, 0.15, 0.5, 0.85, 0.99]) {
      const people = approaches.flatMap((approach, i) => Array.from({ length: 5 }, (_, k) =>
        person(approach, i * 5 + k + 1, { crossing: true, waiting: false, progress })))
      const { result } = render([], people, { phase: 'pedestrian-crossing' })
      expect(result.crossingPedestrians > 0 || result.boxOccupied, `progress ${progress}`).toBe(true)
    }
  })
  it('can export a review screenshot and recognition boxes without saving simulator metadata', () => {
    const vehicles = approaches.flatMap((approach, i) => [
      vehicle(approach, { id: i * 3 + 1, color: palette[i], progress: 0.85, lane: 'left', intent: 'left', waiting: true }),
      vehicle(approach, { id: i * 3 + 2, kind: 'truck', progress: 0.6 }),
    ])
    const people = approaches.flatMap((approach, i) => Array.from({ length: 5 }, (_, k) => person(approach, i * 5 + k + 1)))
    const { canvas, result } = render(vehicles, people)
    expect(result.observation.cars).toBe(4)
    expect(result.observation.trucks).toBe(4)
    if (process.env.VISION_SNAPSHOT_DIR) {
      mkdirSync(process.env.VISION_SNAPSHOT_DIR, { recursive: true })
      writeFileSync(join(process.env.VISION_SNAPSHOT_DIR, 'vision-renderer.png'), canvas.toBuffer('image/png'))
      writeFileSync(join(process.env.VISION_SNAPSHOT_DIR, 'vision-recognition.json'), JSON.stringify(result, null, 2))
      const context = canvas.getContext('2d')
      const pixels = context.getImageData(0, 0, 800, 800)
      const start = performance.now()
      for (let i = 0; i < 10; i++) analyzeScreenshot(pixels, 1)
      writeFileSync(join(process.env.VISION_SNAPSHOT_DIR, 'vision-timing.json'), JSON.stringify({
        detectorMsPerFrame: (performance.now() - start) / 10,
        repetitions: 10, imageWidth: 800, imageHeight: 800,
      }, null, 2))
      for (const detection of result.detections) {
        context.strokeStyle = detection.kind === 'pedestrian' ? '#52e3bd' : '#ffdd63'
        context.lineWidth = 1.5
        context.strokeRect(detection.x - 2, detection.y - 2, detection.width + 4, detection.height + 4)
        context.fillStyle = context.strokeStyle
        context.font = '10px sans-serif'
        context.fillText(`${detection.kind} ${detection.id}`, detection.x, detection.y - 5)
      }
      writeFileSync(join(process.env.VISION_SNAPSHOT_DIR, 'vision-detected.png'), canvas.toBuffer('image/png'))
    }
    // The second row physically hides four heads. Recognition reports visible
    // people instead of fabricating the simulator's hidden count.
    expect(result.observation.pedestrians).toBeGreaterThanOrEqual(12)
    expect(result.observation.pedestrians).toBeLessThanOrEqual(20)
  })
})
