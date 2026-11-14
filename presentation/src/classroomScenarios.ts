import {
  createInitialState, createTrafficJam, pressCrosswalkButton, spawnAgents,
  type Approach, type SimState, type SpawnConfig,
} from './simulation'

export type DemoScenarioId = 'balanced' | 'east-jam' | 'pedestrians' | 'camera-loss' | 'fixed'

export const DEMO_SCENARIOS: Record<DemoScenarioId, {
  label: string; description: string; steps: string[]
}> = {
  balanced: {
    label: 'Balanced traffic',
    description: 'A repeatable starting scene with 12 cars and 4 trucks spread across all four approaches.',
    steps: [
      'Watch the single camera view show each captured intersection image with its detection boxes.',
      'Read the NS and EW scores, selected direction, and reason in the controller panel.',
      'Observe green, yellow, and all-red transitions. New arrivals continue during the demonstration.',
    ],
  },
  'east-jam': {
    label: 'East-side congestion',
    description: 'The same starting traffic plus a visible queue on the east approach and higher east-side arrival demand.',
    steps: [
      'Point out the east-side queue in the image and the measured EW score.',
      'Watch the controller select its next direction from captured pixels after the intersection clears.',
      'Explain that demand changes simulated arrivals; the controller reads the visible queue rather than that hidden setting.',
    ],
  },
  pedestrians: {
    label: 'Pedestrian request',
    description: 'Twelve pedestrians wait at the corners, with a crosswalk button request already queued.',
    steps: [
      'Point out the waiting pedestrians and the queued crosswalk request.',
      'Watch all vehicle lights remain red while the pedestrian phase runs after fresh clear camera frames.',
      'Explain that people arriving after a walking group starts wait for a later crossing.',
    ],
  },
  'camera-loss': {
    label: 'Camera interruption',
    description: 'Begin with a working camera, then turn it off during a green to demonstrate the missing-input response.',
    steps: [
      'Let the camera establish a clear view and wait for a vehicle green.',
      'Turn Camera off. The image freezes; read the current phase banner and controller as green finishes through yellow before all lights hold red.',
      'Turn Camera on. Recovery needs fresh clear images and the all-red interval before a new green can start.',
    ],
  },
  fixed: {
    label: 'Fixed-cycle comparison',
    description: 'Exactly the balanced starting scene, with fixed timing and alternating directions instead of adaptive scoring.',
    steps: [
      'Compare this starting image with Balanced traffic: initial cars, trucks, and positions match.',
      'Watch the fixed controller alternate directions with six-second through greens; camera clearance rules still apply.',
      'Describe this as a live illustration. Later random arrivals differ, so these are not a controlled performance benchmark.',
    ],
  },
}

// Only the initial scene is seeded. Later arrivals use the existing simulation.
function scenarioRandom(seed: number): () => number {
  let value = seed >>> 0
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0
    return value / 0x100000000
  }
}

export function createClassroomScenario(id: DemoScenarioId): {
  state: SimState; cameraEnabled: boolean; counts: SpawnConfig; jamApproach: Approach
} {
  const rng = scenarioRandom(20261114)
  let state = spawnAgents(createInitialState(id !== 'fixed', 'camera'), {
    cars: 12, trucks: 4, pedestrians: id === 'pedestrians' ? 12 : 0,
  }, rng)
  if (id === 'east-jam') state = createTrafficJam(state, 'east', rng)
  if (id === 'pedestrians') state = pressCrosswalkButton(state, 'ns')
  // Stable scene-local IDs also reproduce the pedestrian shirt colors. No
  // observations are inserted: the camera must measure the rendered pixels.
  state = {
    ...state,
    vehicles: state.vehicles.map((vehicle, index) => ({ ...vehicle, id: index + 1 })),
    pedestrians: state.pedestrians.map((person, index) => ({ ...person, id: state.vehicles.length + index + 1 })),
  }
  return {
    state, cameraEnabled: true, jamApproach: 'east',
    counts: {
      cars: state.vehicles.filter(vehicle => vehicle.kind === 'car').length,
      trucks: state.vehicles.filter(vehicle => vehicle.kind === 'truck').length,
      pedestrians: state.pedestrians.length,
    },
  }
}
