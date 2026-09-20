export type Slide = {
  id: string
  kicker?: string
  title: string
  body?: string
  bullets?: string[]
  metricRows?: { label: string; value: string; note?: string }[]
  footer?: string
  variant?: 'hero' | 'content' | 'dark-metric' | 'warning'
}

export const SLIDES: Slide[] = [
  {
    id: 'title',
    variant: 'hero',
    kicker: 'Traffic Observatory · LSTM demo',
    title: 'Forecast the next hour.\nSee every neuron fire.',
    body: 'A stacked LSTM trained on Metro Interstate traffic — plus a live 2D intersection that reacts when demand spikes.',
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
    id: 'pipeline',
    variant: 'content',
    kicker: '02 · Pipeline honesty',
    title: 'Chronological split. Scaler never sees the future.',
    bullets: [
      'No shuffled train/test — production only has the past',
      'v2 causal pipeline: split raw timestamps before fitting transforms',
      'Gaps are disclosed, not papered over (28.7% of training windows spanned a break)',
      'Archived v1 artifacts stay readable; new runs default to v2',
    ],
  },
  {
    id: 'results',
    variant: 'dark-metric',
    kicker: '03 · Motorway results (MAE, vehicles/h)',
    title: 'Calendar helps. Weather mostly costs.',
    metricRows: [
      { label: 'XGBoost + calendar', value: '154.3', note: 'best on this dataset' },
      { label: 'LSTM + calendar', value: '201.4', note: '−12% vs univariate LSTM' },
      { label: 'LSTM traffic only', value: '228.8', note: 'documented vault run' },
      { label: 'Naive “last hour”', value: '585.6', note: 'must beat this' },
    ],
    footer: 'Verdict flips on bike-sharing: LSTM wins by 0.3%. One dataset cannot settle the tool choice.',
  },
  {
    id: 'neurons',
    variant: 'content',
    kicker: '04 · Interpretability',
    title: 'The Obsidian vault is the network, not a drawing of one.',
    bullets: [
      'introspect.py replays every LSTM gate in NumPy from the trained weights',
      'Checked against Keras each export — max abs error ~1e-7',
      'Rush-hour vs quiet-night canvases: different units light up',
      'Streamlit workbench trains, forecasts, and opens the same neurons',
    ],
  },
  {
    id: 'product',
    variant: 'warning',
    kicker: '05 · Product boundary',
    title: 'Traffic Observatory — observation, not control.',
    body: 'This demo can adapt green time inside a simulator. It does not connect to signal cabinets, roadside hardware, or a city feed.',
    bullets: [
      'Street-data preflight: cadence, gaps, duplicates, freshness',
      'Forecast packages are versioned and hash-verified',
      'Any live corridor work needs an authorized partner dataset',
    ],
  },
  {
    id: 'live',
    variant: 'hero',
    kicker: '06 · Live demo',
    title: 'Open the 2D intersection.',
    body: 'Spawn cars, trucks, and pedestrians. Jam one approach. Watch queues grow — and watch adaptive green time respond in simulation.',
    footer: 'Press Live intersection → in the top bar',
  },
]
