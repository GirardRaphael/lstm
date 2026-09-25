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
    footer: 'Raphael Girard · presentation deck · simulation labelled read-only',
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
    body: 'Not “LSTM is the winner.” XGBoost was stronger on the motorway. The LSTM stays because the mechanism is visible.',
    bullets: [
      'Ordered hours: rush builds over a day; a cell state can keep or discard that history at each step',
      'Gates are explicit valves (forget / input / output) — introspect.py replays them from trained weights',
      'The live panel shows that state; it does not prove the LSTM should run a city’s lights',
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
      'Gaps are disclosed, not papered over (28.7% of training windows spanned a break)',
      'Archived v1 artifacts stay readable; new runs default to v2',
    ],
  },
  {
    id: 'xgb-vs-lstm',
    variant: 'dark-metric',
    kicker: '04 · Say this out loud',
    title: 'The verdict flips by dataset. Keep both.',
    body: 'XGBoost won on motorway MAE. LSTM won (narrowly) on bikes. That is why both stay.',
    tables: [
      {
        caption: 'UCI Metro Interstate (vehicles / hour) — XGBoost won',
        headers: ['Model', 'MAE', 'Note'],
        rows: [
          ['XGBoost + calendar', '154.3', 'best on this set'],
          ['LSTM + calendar', '201.4', 'best LSTM on this set'],
          ['LSTM univariate', '228.8', 'past traffic only'],
          ['Naive last-hour', '585.6', 'floor to beat'],
        ],
      },
      {
        caption: 'UCI Bike Sharing (rentals / hour) — LSTM won, narrowly',
        headers: ['Setup', 'LSTM', 'XGBoost', 'Winner'],
        rows: [
          ['Univariate', '41.3', '43.1', 'LSTM by 4.1%'],
          ['+ weather & calendar', '38.5', '38.6', 'LSTM by 0.3%'],
        ],
      },
    ],
    footer:
      'Sources: reports/model_comparison.json, reports/REPORT.md. MAE units differ across datasets — compare within a row only.',
  },
  {
    id: 'results',
    variant: 'dark-metric',
    kicker: '05 · Motorway MAE (vehicles/h)',
    title: 'Calendar helps the LSTM. Weather mostly costs.',
    metricRows: [
      { label: 'XGBoost + calendar', value: '154.3', note: 'best on this dataset' },
      { label: 'LSTM + calendar', value: '201.4', note: '−12% vs univariate LSTM' },
      { label: 'LSTM traffic only', value: '228.8', note: 'documented vault run' },
      { label: 'Naive “last hour”', value: '585.6', note: 'must beat this' },
    ],
    footer:
      'XGBoost won on the motorway. On bike-sharing the LSTM won (41.3 vs 43.1 univariate; 38.5 vs 38.6 multivariate). Verdict flips. Keep both.',
  },
  {
    id: 'neurons',
    variant: 'content',
    kicker: '06 · Interpretability',
    title: 'The decision is visible neuron by neuron.',
    bullets: [
      'introspect.py replays every LSTM gate in NumPy from the trained weights',
      'Checked against Keras each export — max abs error ~1e-7',
      'Live panel: nearest real Metro Interstate hour, then a real forward pass — not a quiet/rush blend',
      'That is inspection of a forecast, not closed-loop control of signals',
    ],
  },
  {
    id: 'product',
    variant: 'warning',
    kicker: '07 · Product boundary',
    title: 'Simulation only. Not city signals.',
    body: 'This intersection is a visual. The cartoon lights are not a closed-loop controller and are not connected to signal cabinets, roadside hardware, or a city feed.',
    bullets: [
      'The LSTM forecasts next-hour demand on stored motorway hours',
      'Recommended green in the demo is a queue heuristic driven by that forecast — the LSTM did not choose the light',
      'Street-data preflight and versioned forecast packages exist for later corridor work, with an authorized partner dataset',
    ],
  },
  {
    id: 'live',
    variant: 'hero',
    kicker: '08 · Live demo',
    title: 'Jam the intersection.\nWatch a real hour replay.',
    body: 'Randomize cars, trucks, and pedestrians. Force a jam. The brain shows gates and neurons from baseline_univariate.keras on the closest real test hour.',
    footer: 'Press Live intersection → in the top bar · simulation only',
  },
  {
    id: 'talk-track',
    variant: 'content',
    kicker: '09 · 3-minute talk track',
    title: 'Say this, then click Live.',
    bullets: [
      '“Simulation only — these cartoon lights are not a closed-loop controller and are not connected to city signals.”',
      '“On Metro Interstate, XGBoost+calendar MAE 154.3 beat LSTM+calendar 201.4 and univariate LSTM 228.8. Last-hour was 585.6.”',
      '“On bike-sharing the LSTM won narrowly: 41.3 vs XGBoost 43.1, and 38.5 vs 38.6 with weather+calendar. The verdict flips. That is why both stay.”',
      '“We still studied the LSTM for sequential hours and inspectable gates — not because it is the universal winner.”',
    ],
    footer: 'Keep the product-boundary slide if someone asks about real signals.',
  },
]
