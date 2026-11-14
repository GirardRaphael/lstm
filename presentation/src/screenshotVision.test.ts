import { describe, expect, it } from 'vitest'
import { analyzeScreenshot, type ScreenshotPixels, type VisionFrame } from './screenshotVision'

type RGB = [number, number, number]
const S = 800
function fill(frame: ScreenshotPixels, x: number, y: number, w: number, h: number, color: RGB) {
  for (let yy = Math.max(0, Math.round(y)); yy < Math.min(S, y + h); yy++)
    for (let xx = Math.max(0, Math.round(x)); xx < Math.min(S, x + w); xx++) {
      const i = (yy * S + xx) * 4
      frame.data[i] = color[0]; frame.data[i + 1] = color[1]; frame.data[i + 2] = color[2]; frame.data[i + 3] = 255
    }
}
function disc(frame: ScreenshotPixels, x: number, y: number, radius: number, color: RGB) {
  for (let yy = Math.floor(y - radius); yy <= y + radius; yy++)
    for (let xx = Math.floor(x - radius); xx <= x + radius; xx++)
      if ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) fill(frame, xx, yy, 1, 1, color)
}
function road(): ScreenshotPixels {
  const frame = { width: S, height: S, data: new Uint8ClampedArray(S * S * 4) }
  fill(frame, 0, 0, S, S, [44, 53, 62])
  fill(frame, 310, 0, 180, S, [26, 35, 44])
  fill(frame, 0, 310, S, 180, [26, 35, 44])
  // Realistic road paint deliberately shares brightness with white vehicles.
  for (let i = 0; i < 8; i++) {
    fill(frame, 320 + i * 14, 312, 8, 16, [190, 185, 177])
    fill(frame, 320 + i * 14, 472, 8, 16, [190, 185, 177])
    fill(frame, 312, 320 + i * 14, 16, 8, [190, 185, 177])
    fill(frame, 472, 320 + i * 14, 16, 8, [190, 185, 177])
  }
  fill(frame, 397, 0, 3, 310, [135, 99, 23])
  fill(frame, 403, 0, 3, 310, [135, 99, 23])
  fill(frame, 316, 304, 78, 4, [182, 177, 169])
  return frame
}
function car(frame: ScreenshotPixels, x: number, y: number, vertical = true, color: RGB = [231, 76, 60]) {
  fill(frame, x - (vertical ? 8 : 14), y - (vertical ? 14 : 8), vertical ? 16 : 28, vertical ? 28 : 16, color)
  fill(frame, x - 6, y - 6, 12, 12, color.map(c => Math.max(0, c - 76)) as RGB)
  fill(frame, vertical ? x - 5 : x + 4, vertical ? y + 4 : y - 5, vertical ? 10 : 5, vertical ? 5 : 10, [80, 110, 140])
}
function truck(frame: ScreenshotPixels, x: number, y: number, vertical = true, stopped = false) {
  fill(frame, x - (vertical ? 11 : 20), y - (vertical ? 20 : 11), vertical ? 22 : 40, vertical ? 40 : 22,
    stopped ? [151, 63, 14] : [231, 139, 8])
  fill(frame, vertical ? x - 9 : x + 10, vertical ? y + 10 : y - 9, vertical ? 18 : 10, vertical ? 10 : 18, [31, 41, 55])
}
function pedestrian(frame: ScreenshotPixels, x: number, y: number, crossing = false, color: RGB = [52, 152, 219]) {
  fill(frame, x - 5, y - 2, 10, 9, color)
  disc(frame, x, y - 6, 4.2, crossing ? [243, 224, 200] : [232, 212, 176])
}

describe('screenshot pixel recognition', () => {
  it('rejects missing, transparent, blank and wrongly sized images; an empty calibrated road is valid', () => {
    const blank = { width: S, height: S, data: new Uint8ClampedArray(S * S * 4) }
    expect(analyzeScreenshot(blank, 0).valid).toBe(false)
    expect(analyzeScreenshot(blank, 0).boxOccupied).toBe(true)
    fill(blank, 0, 0, S, S, [44, 53, 62])
    expect(analyzeScreenshot(blank, 0).valid).toBe(false)
    expect(analyzeScreenshot({ ...blank, width: 400 }, 0).valid).toBe(false)
    const detected = analyzeScreenshot(road(), 0)
    expect(detected.valid).toBe(true)
    expect(detected.detections).toHaveLength(0)
    expect(detected.boxOccupied).toBe(false)
    expect(detected.observation.queues).toEqual({ north: 0, south: 0, east: 0, west: 0 })
  })
  it('reads inbound cars and left lanes from pixels and excludes outbound vehicles', () => {
    const frame = road()
    car(frame, 387, 210)
    car(frame, 445, 600, true, [236, 240, 241])
    car(frame, 610, 355, false, [26, 188, 156])
    car(frame, 190, 413, false, [155, 89, 182])
    car(frame, 445, 180) // northbound departure on the upper road
    const detected = analyzeScreenshot(frame, 1)
    expect(detected.observation.queues).toEqual({ north: 1, south: 1, east: 1, west: 1 })
    expect(detected.observation.leftQueues).toEqual({ north: 1, south: 0, east: 0, west: 1 })
    expect(detected.observation.cars).toBe(4)
    expect(detected.detections.filter(d => d.kind === 'car')).toHaveLength(5)
    expect(detected.boxOccupied).toBe(false)
  })
  it('recognizes both bright and dark truck bodies and weights them twice', () => {
    const frame = road()
    truck(frame, 355, 160)
    truck(frame, 630, 387, false, true)
    const detected = analyzeScreenshot(frame, 1)
    expect(detected.observation.trucks).toBe(2)
    expect(detected.observation.cars).toBe(0)
    expect(detected.observation.queues.north).toBe(2)
    expect(detected.observation.queues.east).toBe(2)
  })
  it('distinguishes waiting pedestrians from crossing people and colored shirts from cars', () => {
    const frame = road()
    pedestrian(frame, 296, 296)
    pedestrian(frame, 282, 296, false, [231, 76, 60])
    pedestrian(frame, 504, 504, false, [236, 240, 241])
    pedestrian(frame, 350, 322, true)
    const detected = analyzeScreenshot(frame, 1)
    expect(detected.observation.pedestrians).toBe(3)
    expect(detected.crossingPedestrians).toBe(1)
    expect(detected.observation.cars).toBe(0)
  })
  it('estimates stopping and pedestrian waiting time by tracking successive screenshots', () => {
    const frame = road()
    car(frame, 355, 240)
    pedestrian(frame, 296, 296)
    let result: VisionFrame = analyzeScreenshot(frame, 0)
    expect(result.observation.stopped.ns).toBe(0)
    for (const time of [0.4, 0.8, 1.2]) result = analyzeScreenshot(frame, time, result)
    expect(result.observation.stopped.ns).toBe(1)
    expect(result.observation.oldestPedestrian).toBeCloseTo(1.2)
    const moving = road()
    car(moving, 355, 255)
    pedestrian(moving, 296, 296)
    result = analyzeScreenshot(moving, 1.6, result)
    expect(result.observation.stopped.ns).toBe(0)
    expect(result.observation.oldestPedestrian).toBeCloseTo(1.6)
  })
  it('resets wait estimates after a camera gap or clock reset', () => {
    const frame = road()
    pedestrian(frame, 296, 296)
    const first = analyzeScreenshot(frame, 1)
    const later = analyzeScreenshot(frame, 2, first)
    expect(later.observation.oldestPedestrian).toBe(1)
    expect(analyzeScreenshot(frame, 10, later).observation.oldestPedestrian).toBe(0)
    expect(analyzeScreenshot(frame, 0, later).observation.oldestPedestrian).toBe(0)
  })
  it('holds clearance for junction vehicles and uncertain colored fragments but permits stopped bumpers', () => {
    const stopped = road()
    car(stopped, 387, 291)
    truck(stopped, 355, 291, true, true)
    expect(analyzeScreenshot(stopped, 1).boxOccupied).toBe(false)
    const occupied = road()
    car(occupied, 355, 360)
    expect(analyzeScreenshot(occupied, 1).boxOccupied).toBe(true)
    const uncertain = road()
    fill(uncertain, 410, 410, 10, 10, [231, 76, 60])
    const result = analyzeScreenshot(uncertain, 1)
    expect(result.boxOccupied).toBe(true)
    expect(result.observation.cars).toBe(0)
  })
  it('does not mistake lit signal bulbs for vehicles', () => {
    const frame = road()
    disc(frame, 286, 324, 7, [46, 196, 182])
    disc(frame, 476, 286, 5, [255, 209, 102])
    disc(frame, 514, 496, 7, [255, 90, 95])
    const result = analyzeScreenshot(frame, 1)
    expect(result.detections).toHaveLength(0)
    expect(result.boxOccupied).toBe(false)
  })
  it('responds to changed pixels without access to scene objects', () => {
    const north = road(), east = road()
    car(north, 355, 210)
    car(east, 600, 355, false)
    expect(analyzeScreenshot(north, 1).observation.queues.north).toBe(1)
    expect(analyzeScreenshot(east, 1).observation.queues.north).toBe(0)
    expect(analyzeScreenshot(east, 1).observation.queues.east).toBe(1)
  })
})
