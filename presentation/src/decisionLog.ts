import type { Approach } from './simulation'
import type { DecisionThought, ModelSource } from './decisionBrain'

export type Axis = 'ns' | 'ew'
export type LogTrigger = 'phase' | 'jam' | 'forecast' | 'override'
export type ReviewStatus = 'looks_right' | 'needs_review' | 'you_corrected'
export type ForecastQuality = 'good' | 'poor' | 'unavailable'

export type QueueSnapshot = {
  north: number
  south: number
  east: number
  west: number
}

export type LiveCounts = {
  cars: number
  trucks: number
  pedestrians: number
}

/** Metro Interstate naive last-hour MAE (`reports/model_comparison.json`). */
export const NAIVE_LAST_HOUR_MAE = 585.6

export const MAX_LEDGER_ROWS = 48

export type MatchedHour = {
  index: number
  timestamp: string
  actual: number
}

export type ForecastVerdict = {
  predicted: number | null
  actual: number | null
  residual: number | null
  absResidual: number | null
  quality: ForecastQuality
  vsNaive: 'beats_naive' | 'worse_than_naive' | 'unavailable'
}

export type HeuristicVerdict = {
  servedAxis: Axis
  longestApproach: Approach
  servedLongest: boolean
  queuesAtDecision: QueueSnapshot
  queuesAtPhaseEnd: QueueSnapshot | null
  servedQueueBefore: number
  servedQueueAfter: number | null
  queueShrunk: boolean | null
}

export type DecisionSnapshot = {
  trigger: LogTrigger
  phase: string
  phaseLabel: string
  tick: number
  jamActive: boolean
  queues: QueueSnapshot
  liveCounts: LiveCounts
  heaviest: Approach
  heaviestPcu: number
  recommendedAxis: Axis
  recommendedGreen: number
  forecastVehicles: number
  matchedHour: MatchedHour | null
  source: ModelSource
  userOverride: Axis | null
  adaptive: boolean
  atMs: number
}

export type DecisionRow = {
  id: number
  atMs: number
  tick: number
  trigger: LogTrigger
  changeKey: string
  phase: string
  phaseLabel: string
  sensed: {
    heaviest: Approach
    pcu: number
    queues: QueueSnapshot
    liveCounts: LiveCounts
  }
  matchedHour: MatchedHour | null
  forecastVehicles: number | null
  source: ModelSource
  heuristic: {
    axis: Axis
    greenSeconds: number
    why: string
    userOverride: Axis | null
  }
  forecastVerdict: ForecastVerdict
  heuristicVerdict: HeuristicVerdict
  status: ReviewStatus
  statusLocked: boolean
  narrative: string
}

export type DecisionLogState = {
  rows: DecisionRow[]
  nextId: number
  lastKeys: Partial<Record<LogTrigger, string>>
  pendingShrinkIds: number[]
  openGreenPhase: string | null
  /** Consumed by the next non-override logged recommendation. */
  pendingAxisOverride: Axis | null
}

export function emptyDecisionLog(): DecisionLogState {
  return {
    rows: [],
    nextId: 1,
    lastKeys: {},
    pendingShrinkIds: [],
    openGreenPhase: null,
    pendingAxisOverride: null,
  }
}

export function axisFromApproach(approach: Approach): Axis {
  return approach === 'north' || approach === 'south' ? 'ns' : 'ew'
}

export function oppositeAxis(axis: Axis): Axis {
  return axis === 'ns' ? 'ew' : 'ns'
}

export function axisShort(axis: Axis): string {
  return axis === 'ns' ? 'NS' : 'EW'
}

export function axisLong(axis: Axis): string {
  return axis === 'ns' ? 'North–South' : 'East–West'
}

export function queueOnAxis(queues: QueueSnapshot, axis: Axis): number {
  return axis === 'ns' ? queues.north + queues.south : queues.east + queues.west
}

export function peakOnAxis(queues: QueueSnapshot, axis: Axis): number {
  return axis === 'ns'
    ? Math.max(queues.north, queues.south)
    : Math.max(queues.east, queues.west)
}

export function servedLongestQueue(served: Axis, heaviest: Approach): boolean {
  return served === axisFromApproach(heaviest)
}

export function forecastResidual(
  predicted: number | null | undefined,
  actual: number | null | undefined,
): number | null {
  if (!Number.isFinite(predicted) || !Number.isFinite(actual)) return null
  return (predicted as number) - (actual as number)
}

export function gradeForecast(
  predicted: number | null | undefined,
  actual: number | null | undefined,
  naiveMae = NAIVE_LAST_HOUR_MAE,
): ForecastVerdict {
  const residual = forecastResidual(predicted, actual)
  if (residual == null) {
    return {
      predicted: Number.isFinite(predicted) ? (predicted as number) : null,
      actual: Number.isFinite(actual) ? (actual as number) : null,
      residual: null,
      absResidual: null,
      quality: 'unavailable',
      vsNaive: 'unavailable',
    }
  }
  const absResidual = Math.abs(residual)
  const vsNaive = absResidual <= naiveMae ? 'beats_naive' : 'worse_than_naive'
  return {
    predicted: predicted as number,
    actual: actual as number,
    residual,
    absResidual,
    quality: vsNaive === 'beats_naive' ? 'good' : 'poor',
    vsNaive,
  }
}

export function isGreenPhase(phase: string): boolean {
  return phase === 'ns-green' || phase === 'ew-green'
}

export function changeKey(snap: DecisionSnapshot): string {
  const hour = snap.matchedHour?.index ?? 'none'
  const forecast = Number.isFinite(snap.forecastVehicles)
    ? Math.round(snap.forecastVehicles)
    : 'na'
  switch (snap.trigger) {
    case 'phase':
      return `phase:${snap.phase}`
    case 'jam':
      return `jam:${snap.jamActive ? 1 : 0}`
    case 'forecast':
      return `forecast:${hour}:${forecast}`
    case 'override':
      return `override:${snap.userOverride ?? 'none'}:${snap.atMs}:${snap.tick}`
  }
}

export function shouldAppend(log: DecisionLogState, snap: DecisionSnapshot): boolean {
  if (snap.trigger === 'override') return true
  if (snap.trigger === 'jam' && !snap.jamActive) return false
  const key = changeKey(snap)
  const prev = log.lastKeys[snap.trigger]
  if (prev === undefined) {
    // Seed phase/forecast fingerprints so the first live frame is not a fake decision.
    // A jam that is already active on first observation is a real "jam created".
    return snap.trigger === 'jam'
  }
  return prev !== key
}

export function heuristicWhy(snap: DecisionSnapshot): string {
  const axis = axisShort(snap.recommendedAxis)
  const seconds = snap.recommendedGreen.toFixed(1)
  if (snap.userOverride) {
    return `user correction: serve ${axis} (${seconds}s) next — the LSTM did not learn from this`
  }
  const forecastBit = Number.isFinite(snap.forecastVehicles)
    ? `; extend ${axis} because forecast ~${Math.round(snap.forecastVehicles)}`
    : ''
  return `serve heaviest queue; ${axis} ${seconds}s because ${snap.heaviest} queue ${snap.heaviestPcu} PCU${forecastBit}`
}

export function buildNarrative(row: Omit<DecisionRow, 'narrative'> | DecisionRow): string {
  const axis = axisShort(row.heuristic.axis)
  const seconds = row.heuristic.greenSeconds.toFixed(1)
  const heuristicBit = row.heuristic.userOverride
    ? `User correction: serve ${axis} (${seconds}s) next — the model did not learn from this.`
    : `Heuristic served ${axis} (${seconds}s) because ${row.sensed.heaviest} queue ${row.sensed.pcu} PCU.`

  let forecastBit = ''
  if (row.forecastVerdict.quality === 'unavailable') {
    forecastBit = ' Forecast unavailable — heuristic-only row.'
  } else if (row.matchedHour && row.forecastVerdict.absResidual != null) {
    const pred = Math.round(row.forecastVerdict.predicted ?? 0)
    const actual = Math.round(row.matchedHour.actual)
    const abs = Math.round(row.forecastVerdict.absResidual)
    const dir =
      row.forecastVerdict.residual === 0
        ? 'exact'
        : (row.forecastVerdict.residual ?? 0) < 0
          ? `under by ${abs}`
          : `over by ${abs}`
    const grade =
      row.forecastVerdict.quality === 'poor'
        ? 'Forecast is weak on this hour.'
        : `Forecast beats naive last-hour MAE (${Math.round(NAIVE_LAST_HOUR_MAE)}) on this hour.`
    forecastBit = ` Keras replay ${row.matchedHour.timestamp} forecast ${pred} vs actual ${actual} — ${dir}. ${grade}`
  }

  const lightBit = row.heuristicVerdict.servedLongest
    ? ' Light rule matched longest queue.'
    : ` Light rule missed the longest queue (served ${axis}, heaviest was ${row.heuristicVerdict.longestApproach}).`

  let shrinkBit = ''
  if (row.heuristicVerdict.queueShrunk === true) {
    shrinkBit = ` After that green the served queue shrank (${row.heuristicVerdict.servedQueueBefore} → ${row.heuristicVerdict.servedQueueAfter} PCU).`
  } else if (row.heuristicVerdict.queueShrunk === false) {
    shrinkBit = ` After that green the served queue did not shrink (${row.heuristicVerdict.servedQueueBefore} → ${row.heuristicVerdict.servedQueueAfter} PCU).`
  }

  return `${heuristicBit}${forecastBit}${lightBit}${shrinkBit}`
}

function autoStatus(row: Omit<DecisionRow, 'narrative' | 'status' | 'statusLocked'>): ReviewStatus {
  if (row.heuristic.userOverride) return 'you_corrected'
  if (row.forecastVerdict.quality === 'poor' || !row.heuristicVerdict.servedLongest) {
    return 'needs_review'
  }
  return 'looks_right'
}

export function buildDecisionRow(id: number, snap: DecisionSnapshot): DecisionRow {
  const forecastVehicles = Number.isFinite(snap.forecastVehicles) ? snap.forecastVehicles : null
  const forecastVerdict = gradeForecast(forecastVehicles, snap.matchedHour?.actual)
  const servedLongest = servedLongestQueue(snap.recommendedAxis, snap.heaviest)
  const partial: Omit<DecisionRow, 'narrative'> = {
    id,
    atMs: snap.atMs,
    tick: snap.tick,
    trigger: snap.trigger,
    changeKey: changeKey(snap),
    phase: snap.phase,
    phaseLabel: snap.phaseLabel,
    sensed: {
      heaviest: snap.heaviest,
      pcu: snap.heaviestPcu,
      queues: { ...snap.queues },
      liveCounts: { ...snap.liveCounts },
    },
    matchedHour: snap.matchedHour ? { ...snap.matchedHour } : null,
    forecastVehicles,
    source: snap.source,
    heuristic: {
      axis: snap.recommendedAxis,
      greenSeconds: snap.recommendedGreen,
      why: heuristicWhy(snap),
      userOverride: snap.userOverride,
    },
    forecastVerdict,
    heuristicVerdict: {
      servedAxis: snap.recommendedAxis,
      longestApproach: snap.heaviest,
      servedLongest,
      queuesAtDecision: { ...snap.queues },
      queuesAtPhaseEnd: null,
      servedQueueBefore: queueOnAxis(snap.queues, snap.recommendedAxis),
      servedQueueAfter: null,
      queueShrunk: null,
    },
    status: 'needs_review',
    statusLocked: false,
  }
  partial.status = autoStatus(partial)
  return { ...partial, narrative: buildNarrative(partial) }
}

function closePendingShrink(
  log: DecisionLogState,
  queues: QueueSnapshot,
): DecisionLogState {
  if (log.pendingShrinkIds.length === 0) return log
  const pending = new Set(log.pendingShrinkIds)
  const rows = log.rows.map((row) => {
    if (!pending.has(row.id) || row.heuristicVerdict.queuesAtPhaseEnd) return row
    const after = queueOnAxis(queues, row.heuristicVerdict.servedAxis)
    const queueShrunk = after < row.heuristicVerdict.servedQueueBefore
    const heuristicVerdict: HeuristicVerdict = {
      ...row.heuristicVerdict,
      queuesAtPhaseEnd: { ...queues },
      servedQueueAfter: after,
      queueShrunk,
    }
    let status = row.status
    if (!row.statusLocked && status !== 'you_corrected') {
      if (!heuristicVerdict.servedLongest || queueShrunk === false || row.forecastVerdict.quality === 'poor') {
        status = 'needs_review'
      } else if (row.forecastVerdict.quality !== 'unavailable') {
        status = 'looks_right'
      }
    }
    const updated: Omit<DecisionRow, 'narrative'> = {
      ...row,
      heuristicVerdict,
      status,
    }
    return { ...updated, narrative: buildNarrative(updated) }
  })
  return { ...log, rows, pendingShrinkIds: [] }
}

export function ingestSnapshot(log: DecisionLogState, snap: DecisionSnapshot): DecisionLogState {
  const key = changeKey(snap)
  let next = log

  if (snap.trigger === 'phase') {
    const leavingGreen =
      next.openGreenPhase != null && next.openGreenPhase !== snap.phase
    if (leavingGreen) {
      next = closePendingShrink(next, snap.queues)
      next = { ...next, openGreenPhase: isGreenPhase(snap.phase) ? snap.phase : null }
    } else if (isGreenPhase(snap.phase)) {
      next = { ...next, openGreenPhase: snap.phase }
    }
  }

  const append = shouldAppend(next, snap)
  const lastKeys =
    snap.trigger === 'override'
      ? next.lastKeys
      : { ...next.lastKeys, [snap.trigger]: key }

  if (snap.trigger === 'jam' && !snap.jamActive) {
    return { ...next, lastKeys }
  }

  if (!append) {
    return { ...next, lastKeys }
  }

  const row = buildDecisionRow(next.nextId, snap)
  const pendingShrinkIds =
    next.openGreenPhase || isGreenPhase(snap.phase)
      ? [...next.pendingShrinkIds, row.id]
      : next.pendingShrinkIds

  let pendingAxisOverride = next.pendingAxisOverride
  if (snap.trigger !== 'override' && snap.userOverride) {
    pendingAxisOverride = null
  }

  return {
    ...next,
    rows: [row, ...next.rows].slice(0, MAX_LEDGER_ROWS),
    nextId: next.nextId + 1,
    lastKeys,
    pendingShrinkIds,
    pendingAxisOverride,
  }
}

export function markRowCorrect(log: DecisionLogState, rowId: number): DecisionLogState {
  return {
    ...log,
    rows: log.rows.map((row) =>
      row.id === rowId
        ? { ...row, status: 'looks_right', statusLocked: true }
        : row,
    ),
  }
}

export function applyUserOverride(
  log: DecisionLogState,
  rowId: number,
  nextAxis: Axis,
  live: DecisionSnapshot,
): DecisionLogState {
  const marked: DecisionLogState = {
    ...log,
    rows: log.rows.map((row) =>
      row.id === rowId
        ? { ...row, status: 'you_corrected', statusLocked: true }
        : row,
    ),
    pendingAxisOverride: nextAxis,
  }
  const queueForGreen = peakOnAxis(live.queues, nextAxis)
  const recommendedGreen = live.adaptive
    ? Math.min(14, Math.max(3.5, 3.5 + queueForGreen * 0.55))
    : live.recommendedGreen
  const snap: DecisionSnapshot = {
    ...live,
    trigger: 'override',
    userOverride: nextAxis,
    recommendedAxis: nextAxis,
    recommendedGreen,
    atMs: live.atMs,
  }
  return ingestSnapshot(marked, snap)
}

export function snapshotFromLive(args: {
  trigger: LogTrigger
  phase: string
  phaseLabel: string
  tick: number
  jamActive: boolean
  queues: QueueSnapshot
  liveCounts: LiveCounts
  thought: DecisionThought
  adaptive: boolean
  atMs?: number
}): DecisionSnapshot {
  const { thought } = args
  return {
    trigger: args.trigger,
    phase: args.phase,
    phaseLabel: args.phaseLabel,
    tick: args.tick,
    jamActive: args.jamActive,
    queues: { ...args.queues },
    liveCounts: { ...args.liveCounts },
    heaviest: thought.congested,
    heaviestPcu: thought.congestedQueue,
    recommendedAxis: thought.recommendedAxis,
    recommendedGreen: thought.recommendedGreen,
    forecastVehicles: thought.forecastVehicles,
    matchedHour: thought.matchedHour,
    source: thought.source,
    userOverride: thought.userOverride,
    adaptive: args.adaptive,
    atMs: args.atMs ?? Date.now(),
  }
}

export function formatQueues(queues: QueueSnapshot): string {
  return `N${queues.north} S${queues.south} E${queues.east} W${queues.west}`
}

export function formatCounts(counts: LiveCounts): string {
  return `${counts.cars}c / ${counts.trucks}t / ${counts.pedestrians}p`
}

export function statusLabel(status: ReviewStatus): string {
  if (status === 'looks_right') return 'Looks right'
  if (status === 'you_corrected') return 'You corrected it'
  return 'Needs review'
}
