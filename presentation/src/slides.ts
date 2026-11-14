export type Slide = {
  id: string
  kicker?: string
  title: string
  body?: string
  bullets?: string[]
  metricRows?: { label: string; value: string; note?: string }[]
  tables?: {
    caption: string
    headers: string[]
    rows: string[][]
  }[]
  footer?: string
  variant?: 'hero' | 'content' | 'dark-metric' | 'warning'
}

export const SLIDES: Slide[] = [
  {
    id: 'title',
    variant: 'hero',
    kicker: 'Traffic Observatory · LSTM demo',
    title: 'Forecast the next hour.\nOpen the network.\nCompare the tree.',
    body: 'A stacked LSTM on Metro Interstate traffic, scored next to XGBoost, plus a 2D intersection that is a simulation — not city signals.',
    footer: 'Raphael Girard · presentation deck · interactive simulation',
  },
  {
    id: 'problem',
    variant: 'content',
    kicker: '01 · The problem',
    title: 'Traffic is a time series with memory.',
    body: 'How many vehicles arrive in the next hour depends on the hours before it. An LSTM walks through that history and chooses what to keep.',
    bullets: [
      'Supervised regression: predict vehicles / hour',
      'Input: last 24 hours → stacked LSTM 64 → 32 → Dense 16',
      'Honest baselines: “last hour” and “same hour yesterday”',
    ],
  },
  {
    id: 'why-lstm',
    variant: 'content',
    kicker: '02 · Why LSTM was still studied',
    title: 'Sequential hours, gates, inspectable state.',
    body: 'The exploratory legacy motorway MAE favored XGBoost. The LSTM remains useful for inspecting sequence memory; these scores do not establish a general winner.',
    bullets: [
      'Ordered hours: rush builds over a day; a cell state can keep or discard that history at each step',
      'Gates are explicit valves (forget / input / output) — introspect.py replays them from trained weights',
      'The historical forecast panel shows that state; a separate scene-based controller operates the simulated lights',
    ],
  },
  {
    id: 'pipeline',
    variant: 'content',
    kicker: '03 · Pipeline honesty',
    title: 'Chronological split. Scaler never sees the future.',
    bullets: [
      'No shuffled train/test — production only has the past',
      'v2 causal pipeline: split raw timestamps before fitting transforms',
      'Legacy limitation: 28.7% of v1 training windows spanned a break; v2 requires complete windows',
      'New results: read reports/REPORT.md and its linked models/v2/<run>/manifest.json packages',
    ],
  },
  {
    id: 'xgb-vs-lstm',
    variant: 'dark-metric',
    kicker: '04 · Say this out loud',
    title: 'Legacy scores are exploratory.',
    body: 'Legacy motorway MAE favored XGBoost. The tiny bike MAE differences do not establish an LSTM advantage; its MAPE was worse. Select models using validation and report all relevant metrics.',
    tables: [
      {
        caption: 'Legacy UCI Metro Interstate MAE (vehicles / hour)',
        headers: ['Model', 'MAE', 'Note'],
        rows: [
          ['XGBoost + calendar', '154.3', 'lowest legacy MAE shown'],
          ['LSTM + calendar', '201.4', 'lowest legacy LSTM MAE shown'],
          ['LSTM univariate', '228.8', 'past traffic only'],
          ['Naive last-hour', '585.6', 'floor to beat'],
        ],
      },
      {
        caption: 'Legacy UCI Bike Sharing MAE (rentals / hour) — exploratory',
        headers: ['Setup', 'LSTM', 'XGBoost', 'Interpretation'],
        rows: [
          ['Univariate', '41.3', '43.1', 'No established LSTM advantage'],
          ['+ weather & calendar', '38.5', '38.6', 'Tiny MAE gap; LSTM MAPE worse'],
        ],
      },
    ],
    footer:
      'Historical values: reports/LEGACY_REPORT.md. New results: reports/REPORT.md and linked models/v2/<run>/manifest.json packages are the source of truth. Compare MAE within a dataset only.',
  },
  {
    id: 'results',
    variant: 'dark-metric',
    kicker: '05 · Legacy motorway MAE (vehicles/h)',
    title: 'Historical motorway comparison.',
    metricRows: [
      { label: 'XGBoost + calendar', value: '154.3', note: 'lowest legacy MAE shown' },
      { label: 'LSTM + calendar', value: '201.4', note: '−12% vs univariate LSTM' },
      { label: 'LSTM traffic only', value: '228.8', note: 'archived exploratory run' },
      { label: 'Naive “last hour”', value: '585.6', note: 'must beat this' },
    ],
    footer:
      'Exploratory legacy results, not an untouched confirmatory test. The historical bike MAE gap does not establish an LSTM advantage; its MAPE was worse. Use v2 manifests for new results.',
  },
  {
    id: 'neurons',
    variant: 'content',
    kicker: '06 · Interpretability',
    title: 'Inspect the forecast neuron by neuron.',
    bullets: [
      'introspect.py replays every LSTM gate in NumPy from the trained weights',
      'Checked against Keras each export — max abs error ~1e-7',
      'Historical forecast panel: nearest real Metro Interstate hour, then a real forward pass — not a quiet/rush blend',
      'The historical forecast is separate from the algorithm controlling the simulated signals',
    ],
  },
  {
    id: 'product',
    variant: 'warning',
    kicker: '07 · Product boundary',
    title: 'Simulation only. Not city signals.',
    body: 'A deterministic controller reads visible vehicles, waiting pedestrians, and waiting times to choose the simulated lights. It is not connected to signal cabinets, roadside hardware, or a city feed.',
    bullets: [
      'The separate LSTM panel forecasts next-hour demand on stored motorway hours; its forecast does not control these lights',
      'The scene-based controller applies green directions and durations after clearance; the log shows the actual choices',
      'Operator requests wait for a safe transition; pedestrian and waiting-time priority can go first. Requests do not train the model',
      'Real-world signal control and field validation remain outside this demo; it does not establish deployment readiness',
    ],
  },
  {
    id: 'live',
    variant: 'hero',
    kicker: '08 · Live demo',
    title: 'Jam the intersection.\nWatch the lights respond.',
    body: 'Add cars, trucks, and pedestrians. Force a jam. Watch the controller choose the next green and record why. Expand the separate historical LSTM panel to inspect gates and neurons from the closest real test hour.',
    footer: 'Press Live intersection → in the top bar · simulation only',
  },
  {
    id: 'talk-track',
    variant: 'content',
    kicker: '09 · 3-minute talk track',
    title: 'Say this, then click Live.',
    bullets: [
      '“The controller chooses these simulated lights from visible traffic and waiting times. The historical LSTM forecast is separate, and no city signals are connected.”',
      '“The exploratory legacy Metro Interstate scores were XGBoost+calendar MAE 154.3, LSTM+calendar 201.4, univariate LSTM 228.8, and last-hour 585.6. New results come from v2 manifests.”',
      '“The historical bike MAE scores were 41.3 vs 43.1, and 38.5 vs 38.6 with weather+calendar. These small gaps do not establish an LSTM advantage; its MAPE was worse.”',
      '“We still studied the LSTM for sequential hours and inspectable gates — not because it is the universal winner.”',
    ],
    footer: 'Keep the product-boundary slide if someone asks about real signals.',
  },
]
