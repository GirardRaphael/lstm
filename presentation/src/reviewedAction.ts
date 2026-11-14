import { pressCrosswalkButton, type SimState } from './simulation'

export type ReviewedAction = 'ns' | 'ew' | 'walk' | 'hold'

/** Queue a human-reviewed choice; the normal phase and camera checks still apply. */
export function applyReviewedActionToState(state: SimState, action: ReviewedAction): SimState {
  if (action === 'hold') return {
    ...state,
    operatorHold: true,
    controllerOverride: null,
    crosswalkRequest: { ns: false, ew: false },
  }
  const released = { ...state, operatorHold: false }
  if (action === 'walk') return pressCrosswalkButton(released, 'ns')
  // Fixed cycles intentionally ignore axis overrides. An explicit reviewed axis
  // request therefore resumes adaptive operation before queuing that request.
  return { ...released, adaptive: true, controllerOverride: action }
}
