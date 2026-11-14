# Traffic LSTM live intersection

Run `./run_app.ps1` from the repository root. The live view opens automatically
at http://127.0.0.1:43125/. The Slideshow button opens the presentation deck.
Use Ctrl+C in the launcher terminal to stop both services.

The launcher uses `.venv/Scripts/python.exe` or the existing
`~/.venvs/traffic_lstm/Scripts/python.exe`. Install the root requirements first.
Node.js 22.12+ is required; this copy was checked with Node 24.14.

## Manual startup

From `presentation/`, run `npm ci --ignore-scripts`, then
`npm run dev` in one terminal and `python server/sidecar.py` in another using
the project Python environment. Both services bind only to localhost.
The page is on port 43125; the model service is on port 43126.

## Checks

Run `npm test` and `npm run build` from `presentation/`.
The API health endpoint is http://127.0.0.1:43125/api/health.

## Source and scope

Imported from `origin/observatory/phase2-v2-wiring`, commit
`0b21861719c6529f16b70401ca948b67ec5680e9` (September 25, 2026).
`DecisionLog.tsx` was renamed `DecisionLogPanel.tsx` to avoid colliding with
`decisionLog.ts` on Windows. The live view is now the default.

The model and data dependencies are the existing root files:
`models/baseline_univariate.keras`, `models/baseline_univariate_artifacts.json`,
and `data/raw/Metro_Interstate_Traffic_Volume.csv`.

This is a simulation. A scene-based controller chooses the lights from visible
traffic and waiting times. The separate historical forecast replays the Keras
weights for the closest stored Metro Interstate hour and does not control the
lights. Operator requests are queued overrides, not model training.


The bundled `Traffic_LSTM_Presentation.pptx` is an archived deck and may contain
superseded model-comparison and controller claims. Use the in-app Slideshow for
the updated explanation. Its legacy scores are exploratory; new results come
from `reports/REPORT.md` and the linked `models/v2/<run>/manifest.json` packages.

## Live controller

Automatic control is on by default. `trafficController.ts` reads scene objects:
visible approaching vehicles (progress 0 to 1), whether they are stopped,
vehicle type, waiting pedestrians, and elapsed waiting time. It does not use
hidden incoming demand, vehicles outside the approach, or historical forecasts.
A truck counts as two cars. Direction scores are twice the queue weight plus
stopped-vehicle weight plus 0.3 times seconds waiting for green. Empty directions
are skipped when the other direction has demand. Ties use the alternating direction.

At each cleared junction, directions waiting at least 30 simulated seconds take
priority (longest first), then an operator request, then the traffic score.
Greens last 4–16 simulated seconds based on observed volume. This is priority
at a decision boundary, not a guaranteed 30-second maximum wait. Yellow and
clearance must finish first. Protected left turns also wait for the box to clear.
Pedestrian requests receive exclusive walk; a lone pedestrian waiting 20 seconds
can trigger walk, subject to the existing post-walk cooldown. A queued manual
vehicle request remains pending if pedestrian or waiting-time priority goes first.

The panel distinguishes the last applied decision from a changing preview;
the log records observations and durations for actual through, left-turn and
walk phases. Disabling automatic control uses alternating six-second through
greens (turn and pedestrian phases still operate). The historical LSTM panel is
separate and does not drive these lights. This controller is a deterministic
algorithm, not an image detector or trained neural policy. No real infrastructure
is connected and no claim of optimal performance is made.

Tests include direction changes under swapped traffic, actual applied durations,
ignored hidden data, waiting-time priority, manual requests, fixed-mode comparison,
clearance before crossing/turns, and pedestrian service.
