import { useId, useState } from 'react'
import type { Slide } from './slides'

function IntersectionDiagram({ boxes = false }: { boxes?: boolean }) {
  return <svg viewBox="0 0 400 330" role="img" aria-label={boxes ? 'Illustration of vehicles with estimated detection boxes' : 'Diagram of a four-way signalized intersection'}>
    <rect width="400" height="330" rx="18" fill="#102b29" />
    <path d="M144 0H256V330H144Z M0 109H400V221H0Z" fill="#283742" />
    <path d="M200 0V98M200 232V330M0 165H133M267 165H400" stroke="#f0c868" strokeWidth="2" strokeDasharray="9 8" />
    <rect x="145" y="110" width="110" height="110" rx="8" fill="#384952" />
    {[0, 1, 2, 3, 4, 5].map(n => <g key={n} fill="#b7c2c4"><rect x={150 + n * 17} y="88" width="10" height="13" /><rect x={150 + n * 17} y="230" width="10" height="13" /><rect x="123" y={114 + n * 17} width="13" height="10" /><rect x="265" y={114 + n * 17} width="13" height="10" /></g>)}
    {[[168, 34], [168, 65], [217, 260]].map(([x, y]) => <g key={`${x}-${y}`}><rect x={x} y={y} width="14" height="24" rx="3" fill="#e4bc70" />{boxes ? <rect x={x - 4} y={y - 4} width="22" height="32" fill="none" stroke="#62e4d3" strokeWidth="2" /> : null}</g>)}
    {[[294, 128], [329, 128], [364, 128], [65, 185]].map(([x, y]) => <g key={`${x}-${y}`}><rect x={x} y={y} width="24" height="14" rx="3" fill="#dc8a73" />{boxes ? <rect x={x - 4} y={y - 4} width="32" height="22" fill="none" stroke="#62e4d3" strokeWidth="2" /> : null}</g>)}
    <circle cx="130" cy="90" r="6" fill="#ff6e74" /><circle cx="270" cy="241" r="6" fill="#ff6e74" />
    <circle cx="274" cy="94" r="6" fill="#62e4d3" /><circle cx="126" cy="239" r="6" fill="#62e4d3" />
    <g fill="#f4efe6" fontSize="12" fontFamily="system-ui" textAnchor="middle"><text x="228" y="20">N</text><text x="173" y="317">S</text><text x="387" y="205">E</text><text x="15" y="128">W</text><text x="200" y="161">Clearance</text><text x="200" y="178">zone</text></g>
  </svg>
}

function ScoreExplorer() {
  const [eastQueue, setEastQueue] = useState(10)
  const [northWait, setNorthWait] = useState(10)
  const id = useId()
  const ns = 2 * 4 + 3 + .3 * northWait
  const ew = 2 * eastQueue + 6 + .3 * 5
  const selected = ns > ew ? 'North / south' : ns < ew ? 'East / west' : 'Tie: alternate direction'
  return <div className="score-explorer">
    <p className="visual-eyebrow">Try the scoring rule</p>
    <label htmlFor={`${id}-q`}>EW weighted queue <b>{eastQueue}</b></label>
    <input id={`${id}-q`} type="range" min="6" max="24" value={eastQueue} onChange={e => setEastQueue(Number(e.target.value))} />
    <label htmlFor={`${id}-w`}>NS seconds waiting <b>{northWait}</b></label>
    <input id={`${id}-w`} type="range" min="0" max="29" value={northWait} onChange={e => setNorthWait(Number(e.target.value))} />
    <div className="score-pair"><div><span>NS score</span><strong>{ns.toFixed(1)}</strong><small>2 × 4 + 3 + 0.3 × {northWait}</small></div><div><span>EW score</span><strong>{ew.toFixed(1)}</strong><small>2 × {eastQueue} + 6 + 0.3 × 5</small></div></div>
    <p className="visual-result">{selected}</p>
    <p className="visual-caption">Teaching example, with fixed stopped weights NS 3 / EW 6. Pedestrians, overdue traffic, operator requests and clearance take priority in the live controller.</p>
  </div>
}

const architecture = [
  ['01', 'Live simulation', 'Canvas pixels → calibrated recognizer → traffic rules → simulated lights', 'Working control loop'],
  ['02', 'Real-image recognition', 'Street image → EfficientDet boxes + separate anomaly classifier', 'Measured, limited accuracy'],
  ['03', 'Learned preferences', 'Frozen image features + your labels → separate decision model', 'Requires human examples'],
  ['04', 'Historical forecasting', '24 hourly counts → stacked LSTM → next-hour volume', 'Separate from live lights'],
]

export function SlideVisual({ kind }: { kind: NonNullable<Slide['visual']> }) {
  if (kind === 'score') return <ScoreExplorer />
  if (kind === 'intersection' || kind === 'perception') return <figure className="intersection-diagram"><IntersectionDiagram boxes={kind === 'perception'} /><figcaption>Explanatory illustration · open the live demo for actual screenshot detections.</figcaption></figure>
  if (kind === 'architecture') return <div className="architecture-lanes">{architecture.map(([number, title, path, status]) => <div key={number}><span>{number}</span><section><h3>{title}</h3><p>{path}</p><small>{status}</small></section></div>)}</div>
  if (kind === 'signal-flow') return <div className="signal-flow"><p className="visual-eyebrow">A request is not an instant switch</p>{[
    ['green', 'Active green', 'Serve the selected movement'], ['yellow', 'Yellow · 1.4 s', 'End the vehicle movement'], ['red', 'All red · ≥ 0.8 s', 'Wait for two fresh clear observations'], ['green', 'Next through green or walk', 'Apply the permitted decision after clearance'],
  ].map(([color, title, text], i) => <div key={title}><span className={`phase-dot ${color}`} /><section><h3>{title}</h3><p>{text}</p></section>{i < 3 ? <span className="flow-arrow" aria-hidden="true">↓</span> : null}</div>)}<p className="visual-caption">An occupied junction or unavailable camera extends the hold. Protected left turns use a separate branch after yellow and clearance, before all-red.</p></div>
  if (kind === 'training') return <div className="training-diagram"><p className="visual-eyebrow">One recording stays in one partition</p><div className="split-bar"><span>Fit ~50%</span><span>Calibrate ~25%</span><span>Evaluate ~25%</span></div>{[
    ['Fit', 'Learn classifier weights from explicit human actions.'], ['Calibrate', 'Adjust probability scale on different recording groups.'], ['Evaluate', 'Compare predictions with held-out human labels.'],
  ].map(([label, text]) => <div className="training-step" key={label}><b>{label}</b><p>{text}</p></div>)}<p className="visual-caption">These are the decision-model group proportions. The aerial anomaly experiment uses its own 2,550 / 225 / 225 image split.</p></div>
  if (kind === 'metrics') return <div className="metric-visual"><p className="visual-eyebrow">Measured detector benchmark · 500 images</p>{[
    ['Precision', 80.98, '1,933 correct / 2,387 evaluated predictions'], ['Recall', 48.18, '1,933 found / 4,012 labeled non-crowd objects'], ['F1', 60.42, 'Balances precision and recall'],
  ].map(([label, value, detail]) => <div key={String(label)}><div className="metric-bar-label"><b>{label}</b><strong>{Number(value).toFixed(1)}%</strong></div><div className="metric-track"><span style={{ width: `${value}%` }} /></div><small>{detail}</small></div>)}<p className="visual-caption">General COCO photographs. This does not measure intersection decisions or prove aerial-road reliability.</p></div>
  return <div className="roadmap-visual">{[
    ['Now', 'Demonstrate and inspect', 'Working simulation, pretrained detections, human labeling.'], ['Next', 'Build the right labels', 'Signalized recordings, corrected object boxes, reviewed actions.'], ['Then', 'Test new recordings', 'Measure misses, false alarms, coverage and delay across conditions.'], ['Later', 'Integrate and assess', 'Tracking, signal-state input and independent control evaluation.'],
  ].map(([stage, title, text]) => <div key={stage}><span>{stage}</span><section><h3>{title}</h3><p>{text}</p></section></div>)}</div>
}
