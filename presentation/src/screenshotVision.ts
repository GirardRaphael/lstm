import type { TrafficObservation } from './trafficController'

/** Calibrated computer vision for this app's unmodified 800px canvas artwork.
 * The input is a screenshot's RGBA pixels, never simulator objects or IDs.
 * This is color/shape recognition, not a model trained on aerial photographs.
 */
export type ScreenshotPixels = { width: number; height: number; data: Uint8ClampedArray }
export type VisionApproach = 'north' | 'south' | 'east' | 'west'
export type VisionDetection = {
  id: number
  kind: 'car' | 'truck' | 'pedestrian'
  x: number
  y: number
  width: number
  height: number
  approach?: VisionApproach
  lane?: 'left' | 'straight' | 'right'
  crossing: boolean
  stopped: boolean
  firstSeen: number
  stationarySince: number
}
export type VisionFrame = {
  valid: boolean
  reason: string
  width: number
  height: number
  capturedAt: number
  observation: TrafficObservation
  detections: VisionDetection[]
  boxOccupied: boolean
  crossingPedestrians: number
  nextTrackId: number
}

type Component = { x: number; y: number; width: number; height: number; area: number }
type RGB = readonly [number, number, number]
const SIZE = 800
const ROAD_MIN = 310
const ROAD_MAX = 490
const CAR_PALETTE: readonly RGB[] = [
  [231, 76, 60], [52, 152, 219], [243, 156, 18],
  [236, 240, 241], [155, 89, 182], [26, 188, 156],
]

function emptyObservation(): TrafficObservation {
  return { queues: { north: 0, south: 0, east: 0, west: 0 },
    leftQueues: { north: 0, south: 0, east: 0, west: 0 },
    stopped: { ns: 0, ew: 0 }, cars: 0, trucks: 0,
    pedestrians: 0, oldestPedestrian: 0 }
}
function inRoad(x: number, y: number): boolean {
  return (x >= ROAD_MIN && x < ROAD_MAX) || (y >= ROAD_MIN && y < ROAD_MAX)
}
function nearColor(r: number, g: number, b: number, color: RGB, tolerance: number): boolean {
  return Math.abs(r - color[0]) <= tolerance && Math.abs(g - color[1]) <= tolerance &&
    Math.abs(b - color[2]) <= tolerance
}
function onGradient(r: number, g: number, b: number, a: RGB, z: RGB): boolean {
  const dr = z[0] - a[0], dg = z[1] - a[1], db = z[2] - a[2]
  const t = Math.max(0, Math.min(1,
    ((r - a[0]) * dr + (g - a[1]) * dg + (b - a[2]) * db) / (dr * dr + dg * dg + db * db)))
  return Math.abs(r - a[0] - t * dr) < 13 && Math.abs(g - a[1] - t * dg) < 13 &&
    Math.abs(b - a[2] - t * db) < 13
}
function bodyPixel(r: number, g: number, b: number): boolean {
  if (Math.max(r, g, b) < 90) return false
  for (const color of CAR_PALETTE) {
    // Canvas body and roof use the same RGB shade offset, including clamping.
    const shade = Math.max(color[0] - r, color[1] - g, color[2] - b)
    if (shade < -6 || shade > 82) continue
    if (Math.abs(Math.max(0, color[0] - shade) - r) < 10 &&
        Math.abs(Math.max(0, color[1] - shade) - g) < 10 &&
        Math.abs(Math.max(0, color[2] - shade) - b) < 10) return true
  }
  return onGradient(r, g, b, [245, 158, 11], [217, 119, 6]) ||
    onGradient(r, g, b, [180, 83, 9], [124, 45, 18])
}
function inSignalHousing(x: number, y: number): boolean {
  return (x >= 270 && x <= 302 && y >= 248 && y <= 340) ||
    (x >= 498 && x <= 530 && y >= 460 && y <= 552) ||
    (x >= 460 && x <= 552 && y >= 270 && y <= 302) ||
    (x >= 248 && x <= 340 && y >= 498 && y <= 530)
}
function inPedestrianSignal(x: number, y: number): boolean {
  return (Math.abs(x - 238) <= 11 || Math.abs(x - 562) <= 11) &&
    (Math.abs(y - 238) <= 11 || Math.abs(y - 562) <= 11)
}

/** Eight-connected components preserve diagonal edges of turning vehicles. */
function components(mask: Uint8Array): Component[] {
  const found: Component[] = []
  const stack: number[] = []
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start]) continue
    mask[start] = 0
    stack.push(start)
    let minX = SIZE, maxX = 0, minY = SIZE, maxY = 0, area = 0
    while (stack.length) {
      const index = stack.pop()!
      const x = index % SIZE, y = Math.floor(index / SIZE)
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y); area++
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy
        if (yy < 0 || yy >= SIZE) continue
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx
          if ((!dx && !dy) || xx < 0 || xx >= SIZE) continue
          const adjacent = yy * SIZE + xx
          if (!mask[adjacent]) continue
          mask[adjacent] = 0
          stack.push(adjacent)
        }
      }
    }
    found.push({ x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1, area })
  }
  return found
}
function calibrationValid(frame: ScreenshotPixels): boolean {
  // Four known, unobstructed sidewalk patches distinguish empty road from a
  // missing/black frame; road samples also reject a flat sidewalk-colored image.
  let sidewalk = 0, road = 0
  for (const x of [20, 779]) for (const y of [20, 779]) {
    const i = (y * SIZE + x) * 4
    if (frame.data[i + 3] >= 250 && nearColor(frame.data[i], frame.data[i + 1], frame.data[i + 2], [44, 53, 62], 8)) sidewalk++
  }
  for (const [x, y] of [[312, 100], [487, 700], [100, 312], [700, 487]]) {
    const i = (y * SIZE + x) * 4
    if (frame.data[i + 3] >= 250 && nearColor(frame.data[i], frame.data[i + 1], frame.data[i + 2], [26, 35, 44], 12)) road++
  }
  return sidewalk >= 3 && road >= 2
}
function vehicleApproach(c: Component): { approach?: VisionApproach; lane?: VisionDetection['lane'] } {
  const x = c.x + c.width / 2, y = c.y + c.height / 2
  let approach: VisionApproach | undefined
  let transverse = 0
  if (y < ROAD_MIN && x < 400) { approach = 'north'; transverse = x }
  else if (y > ROAD_MAX && x > 400) { approach = 'south'; transverse = 800 - x }
  else if (x < ROAD_MIN && y > 400) { approach = 'west'; transverse = 800 - y }
  else if (x > ROAD_MAX && y < 400) { approach = 'east'; transverse = y }
  if (!approach) return {}
  const lane = Math.abs(transverse - 387) <= 15 ? 'left' : Math.abs(transverse - 355) <= 15 ? 'straight' : 'right'
  return { approach, lane }
}
function pedestrianApproach(x: number, y: number): VisionApproach {
  if (x < 400) return y < 400 ? 'north' : 'west'
  return y < 400 ? 'east' : 'south'
}
function overlapsInterior(c: Component): boolean {
  // Leave room for a legal stopped vehicle's nose beyond the painted stop line.
  // Unknown colored fragments in this interior also hold clearance conservatively.
  if (!(c.x + c.width > 306 && c.x < 494 && c.y + c.height > 306 && c.y < 494)) return false
  const x = c.x + c.width / 2, y = c.y + c.height / 2
  // An approaching vehicle whose center remains behind the stop boundary may
  // have a few bumper pixels on the zebra. Turning vehicles can hug the four
  // corners, so the rest of the full crossing area must stay protected.
  if ((x < 400 && y <= 296) || (x > 400 && y >= 504) ||
      (y > 400 && x <= 296) || (y < 400 && x >= 504)) return false
  return true
}

export function analyzeScreenshot(
  frame: ScreenshotPixels,
  capturedAtSeconds: number,
  previous?: VisionFrame,
): VisionFrame {
  const result: VisionFrame = {
    valid: false, reason: '', width: frame.width, height: frame.height, capturedAt: capturedAtSeconds,
    observation: emptyObservation(), detections: [], boxOccupied: true,
    crossingPedestrians: 0, nextTrackId: previous?.nextTrackId ?? 1,
  }
  if (frame.width !== SIZE || frame.height !== SIZE || frame.data.length !== SIZE * SIZE * 4) {
    result.reason = 'Camera requires the calibrated 800 × 800 intersection canvas.'
    return result
  }
  if (!Number.isFinite(capturedAtSeconds) || capturedAtSeconds < 0 || !calibrationValid(frame)) {
    result.reason = 'Camera image is blank or does not match the intersection calibration.'
    return result
  }
  result.valid = true
  result.reason = 'Calibrated screenshot: color and shape detection; counts are estimates.'
  result.boxOccupied = false
  const heads = new Uint8Array(SIZE * SIZE)
  const body = new Uint8Array(SIZE * SIZE)
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const p = y * SIZE + x, i = p * 4
    if (frame.data[i + 3] < 250) continue
    const r = frame.data[i], g = frame.data[i + 1], b = frame.data[i + 2]
    if (r >= 226 && g >= 206 && b >= 170 && !inPedestrianSignal(x, y) &&
      (nearColor(r, g, b, [232, 212, 176], 6) || nearColor(r, g, b, [243, 224, 200], 6))) heads[p] = 1
    if (inRoad(x, y) && !inSignalHousing(x, y) && bodyPixel(r, g, b)) body[p] = 1
  }
  const makeDetection = (kind: VisionDetection['kind'], c: Component, crossing: boolean,
    approach?: VisionApproach, lane?: VisionDetection['lane']): VisionDetection => ({
    id: 0, kind, x: c.x, y: c.y, width: c.width, height: c.height, approach, lane, crossing,
    stopped: false, firstSeen: capturedAtSeconds, stationarySince: capturedAtSeconds,
  })
  for (const head of components(heads)) {
    if (head.area < 10 || head.area > 180 || head.width < 3 || head.height < 3 || head.width > 24 || head.height > 16) continue
    const cx = head.x + head.width / 2, cy = head.y + head.height / 2 + 6
    const crossing = cx >= 305 && cx <= 495 && cy >= 305 && cy <= 495
    const pedestrian = { x: Math.max(0, cx - 6), y: Math.max(0, cy - 11), width: 12, height: 23, area: head.area }
    result.detections.push(makeDetection('pedestrian', pedestrian, crossing, pedestrianApproach(cx, cy)))
    if (crossing) result.crossingPedestrians++
    // A colored shirt must not be counted as a small car or an unknown obstacle.
    for (let y = Math.max(0, Math.floor(cy - 11)); y <= Math.min(799, Math.ceil(cy + 12)); y++)
      for (let x = Math.max(0, Math.floor(cx - 7)); x <= Math.min(799, Math.ceil(cx + 7)); x++) body[y * SIZE + x] = 0
  }
  for (const c of components(body)) {
    const shorter = Math.min(c.width, c.height), longer = Math.max(c.width, c.height)
    // Ignore thin lane paint and zebra stripes, but preserve uncertain objects
    // in the junction as occupancy instead of declaring an empty crossing.
    if (c.area >= 30 && shorter >= 9 && overlapsInterior(c)) result.boxOccupied = true
    const touchesEdge = c.x === 0 || c.y === 0 || c.x + c.width >= SIZE || c.y + c.height >= SIZE
    if (c.area < (touchesEdge ? 70 : 125) || shorter < (touchesEdge ? 7 : 11) || longer < 18 || longer > 65) continue
    if (c.area / (c.width * c.height) < 0.28) continue
    const { approach, lane } = vehicleApproach(c)
    // On an approach, the body's short axis distinguishes 22px trucks from
    // 16px cars even when their dark cab/window pixels split the long axis.
    const kind = shorter >= 18 && c.area >= 300 ? 'truck' : 'car'
    result.detections.push(makeDetection(kind, c, !approach, approach, lane))
  }
  const elapsed = previous?.valid ? capturedAtSeconds - previous.capturedAt : NaN
  const canTrack = Number.isFinite(elapsed) && elapsed > 0 && elapsed <= 2
  const available = new Set(canTrack ? previous!.detections.map(d => d.id) : [])
  for (const detection of result.detections) {
    const cx = detection.x + detection.width / 2, cy = detection.y + detection.height / 2
    let match: VisionDetection | undefined, best = Math.max(14, (elapsed || 0) * 150)
    for (const candidate of canTrack ? previous!.detections : []) {
      if (!available.has(candidate.id) || candidate.kind !== detection.kind || candidate.crossing !== detection.crossing ||
        (candidate.approach && detection.approach && candidate.approach !== detection.approach) ||
        (candidate.lane && detection.lane && candidate.lane !== detection.lane)) continue
      const distance = Math.hypot(cx - candidate.x - candidate.width / 2, cy - candidate.y - candidate.height / 2)
      if (distance < best) { best = distance; match = candidate }
    }
    if (match) {
      available.delete(match.id)
      detection.id = match.id
      detection.firstSeen = match.firstSeen
      if (best / elapsed <= 3) detection.stationarySince = match.stationarySince
      detection.stopped = capturedAtSeconds - detection.stationarySince >= 0.65
    } else detection.id = result.nextTrackId++
    if (detection.kind === 'pedestrian') {
      if (!detection.crossing) {
        result.observation.pedestrians++
        result.observation.oldestPedestrian = Math.max(result.observation.oldestPedestrian, capturedAtSeconds - detection.firstSeen)
      }
    } else if (detection.approach) {
      const weight = detection.kind === 'truck' ? 2 : 1
      result.observation.queues[detection.approach] += weight
      if (detection.lane === 'left') result.observation.leftQueues[detection.approach] += weight
      if (detection.stopped) result.observation.stopped[detection.approach === 'north' || detection.approach === 'south' ? 'ns' : 'ew'] += weight
      if (detection.kind === 'truck') result.observation.trucks++
      else result.observation.cars++
    }
  }
  return result
}
