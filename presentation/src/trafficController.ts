import type { Approach, SimState } from './simulation'

export type Axis = 'ns' | 'ew'
export type TrafficObservation = {
  queues: Record<Approach, number>
  leftQueues: Record<Approach, number>
  stopped: Record<Axis, number>
  cars: number
  trucks: number
  pedestrians: number
  oldestPedestrian: number
}
export type SignalDecision = {
  action: Axis | 'walk'
  movement?: 'left'
  seconds: number
  reason: string
  observation: TrafficObservation
  redWait: Record<Axis, number>
  scores: Record<Axis, number>
  tick: number
  automatic: boolean
}

/** Inputs are rendered scene objects and observed elapsed time, never future arrivals,
 * hidden off-screen queues, the jam flag, or historical motorway forecasts. */
export function observeTraffic(state: SimState): TrafficObservation {
  const result: TrafficObservation = {
    queues: { north: 0, south: 0, east: 0, west: 0 },
    leftQueues: { north: 0, south: 0, east: 0, west: 0 },
    stopped: { ns: 0, ew: 0 }, cars: 0, trucks: 0,
    pedestrians: 0, oldestPedestrian: 0,
  }
  for (const v of state.vehicles) {
    if (v.crossed || !Number.isFinite(v.progress) || v.progress < 0 || v.progress >= 1) continue
    const weight = v.kind === 'truck' ? 2 : 1
    result.queues[v.approach] += weight
    if (v.intent === 'left') result.leftQueues[v.approach] += weight
    if (v.waiting) result.stopped[v.approach === 'north' || v.approach === 'south' ? 'ns' : 'ew'] += weight
    if (v.kind === 'truck') result.trucks++
    else result.cars++
  }
  for (const p of state.pedestrians) {
    if (!p.waiting || p.done) continue
    result.pedestrians++
    result.oldestPedestrian = Math.max(result.oldestPedestrian, p.waitTime || 0)
  }
  return result
}

export function decideTraffic(state: SimState, allowWalk = true): SignalDecision {
  const observation = observeTraffic(state)
  const q = observation.queues
  const pressure = { ns: q.north + q.south, ew: q.east + q.west }
  const redWait = { ...state.redWait }
  const scores = {
    ns: pressure.ns * 2 + observation.stopped.ns + redWait.ns * 0.3,
    ew: pressure.ew * 2 + observation.stopped.ew + redWait.ew * 0.3,
  }
  const base = { observation, redWait, scores, tick: state.tick, automatic: state.adaptive }
  const manualWalk = state.crosswalkRequest.ns || state.crosswalkRequest.ew
  const waitingWalk = state.adaptive && state.pedWalkCooldown <= 0 &&
    (observation.oldestPedestrian >= 20 || (observation.pedestrians >= 6 && observation.oldestPedestrian >= 12))
  if (allowWalk && (manualWalk || waitingWalk)) {
    return { ...base, action: 'walk', seconds: 6,
      reason: manualWalk ? 'Crosswalk request: let pedestrians cross after vehicles clear.' :
        `${observation.pedestrians} pedestrians waiting; longest wait ${observation.oldestPedestrian.toFixed(0)}s.` }
  }
  let action: Axis = state.nextAxis
  let reason = 'Fixed cycle: alternate directions.'
  if (state.adaptive) {
    const overdue = (['ns', 'ew'] as Axis[]).filter(a => pressure[a] > 0 && redWait[a] >= 30)
    if (overdue.length) {
      action = overdue.sort((a, b) => redWait[b] - redWait[a])[0]
      reason = `${action.toUpperCase()} has waited ${redWait[action].toFixed(0)}s; serve it to prevent starvation.`
    } else if (state.controllerOverride) {
      action = state.controllerOverride
      reason = `Operator requested ${action.toUpperCase()} at the next clear junction.`
    } else {
      if (pressure.ns === 0 && pressure.ew > 0) action = 'ew'
      else if (pressure.ew === 0 && pressure.ns > 0) action = 'ns'
      else if (scores.ns !== scores.ew) action = scores.ns > scores.ew ? 'ns' : 'ew'
      reason = `Visible traffic score: NS ${scores.ns.toFixed(1)}, EW ${scores.ew.toFixed(1)}. ` +
        `${action.toUpperCase()} selected from queue size, stopped traffic and waiting time.`
    }
  }
  const seconds = state.adaptive ? Math.min(16, Math.max(4, 4 + pressure[action] * 0.5)) : 6
  return { ...base, action, seconds, reason }
}
