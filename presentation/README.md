# Traffic intersection classroom presentation

Run `./run_app.ps1` from the repository root. The Train & review view opens automatically
at http://127.0.0.1:43125/. Select **Slideshow** for the classroom presentation,
**Live intersection** for the operating simulation, or **Train & review** for real-image
inspection and human teaching labels. **Real traffic videos** opens the observation-only video monitor.
Use Ctrl+C in the launcher terminal to stop both services.

The launcher uses `.venv/Scripts/python.exe` or the existing
`~/.venvs/traffic_lstm/Scripts/python.exe`. Install the root requirements first.
Node.js 22.12+ is required; this copy was checked with Node 24.14.

## Classroom presentation

Use the in-app deck for the current project explanation. It provides chapter navigation,
interactive diagrams, speaker notes, local readiness information and prepared live demo
scenarios. The balanced, east-jam, pedestrian, camera-loss and fixed-cycle presets make
behavior easier to demonstrate. Use **Return to slide** to resume your place after
a demo. Seeded scenes are illustrative scenarios, not quantitative proof of reduced delay.

The deck links to the [classroom presenter guide](../reports/CLASSROOM_PRESENTER_GUIDE.md),
including a 15–20 minute talk, 5/15 minute cuts, exact controls, worked equations,
training/calibration/evaluation explanations, common questions and troubleshooting.
Rehearse the detector once before class because initial model loading can take time.

The essential distinction is between four tasks: object recognition, anomaly recognition,
human light-choice learning, and simulated signal operation. The historical LSTM forecast
is a separate branch. The live camera controller uses graphics-specific screenshot rules;
the real-image lab uses pretrained EfficientDet-Lite2. Human action learning starts
untrained and does not autonomously control physical signals.

Present the actual evidence: detector precision **80.98%**, recall **48.18%** on 500 general
COCO images; the expanded aerial anomaly baseline detected **0 of 45** test anomalies.
The 3,000-image aerial collection supplies binary anomaly labels, not correct signal
choices or verified object boxes. A 90% recommendation-score cutoff is separate from
measured reliability. See the guide and linked reports before quoting a result.

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
`decisionLog.ts` on Windows. **Train & review** is the current default view; the
slideshow and live simulation are separate modes in the same app.

The model and data dependencies are the existing root files:
`models/baseline_univariate.keras`, `models/baseline_univariate_artifacts.json`,
and `data/raw/Metro_Interstate_Traffic_Volume.csv`.

This is a simulation. A screenshot-driven controller chooses the lights from detected
traffic and tracked waiting times. The separate historical forecast replays the Keras
weights for the closest stored Metro Interstate hour and does not control the
lights. Operator requests are queued overrides, not model training.


The bundled `Traffic_LSTM_Presentation.pptx` is an archived deck and may contain
superseded model-comparison and controller claims. Use the in-app Slideshow for
the updated explanation. Its legacy scores are exploratory; new results come
from `reports/REPORT.md` and the linked `models/v2/<run>/manifest.json` packages.

## Live controller

Automatic control and screenshot input are on by default. A hidden 800 × 800
intersection canvas supplies raw captures every 250ms; one annotated camera view
displays the matching image and detections. `screenshotVision.ts` receives only
RGBA pixels, the capture clock and prior pixel detections. Color/shape components
identify cars, trucks and pedestrian heads; calibrated image regions identify
approaches and left lanes. Frame-to-frame matching estimates motion and waiting.
The controller receives these measured counts, estimated waits and visible
occupancy. It does not receive simulator vehicles, pedestrian timers, hidden
arrival demand or historical forecasts in camera mode. Scene mode remains in
the code for regression comparisons; the app always starts in camera mode.
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
rule-based light policy following a calibrated pixel detector, not a trained neural policy. No real infrastructure
is connected and no claim of optimal performance is made.

Tests include direction changes under swapped traffic, actual applied durations,
ignored hidden data, waiting-time priority, manual requests, fixed-mode comparison,
clearance before crossing/turns, and pedestrian service.

## Screenshot input and limits

**What the camera sees** is the single intersection visualization. It shows the
exact captured image with matching detection boxes added after raw capture.
The annotated preview is never fed back into recognition. The raw source stays
unobscured so labels cannot hide vehicles from the recognizer. Use **Download screenshot** and
**Download detections** to save a raw PNG and its estimated boxes/counts as JSON.
Use **Download sample** for one JSON containing both the PNG data URL
and matching detections from the same capture. These are predicted annotations,
not independently verified training labels.

Switch **Camera on** off to test loss of input. The camera image stays frozen at
its last capture; read the current phase banner and controller panel to follow
the signal response. A frame older than one simulation
second, missing frame, blank frame or failed calibration prevents new green/walk
phases. Active green ends via the full yellow interval, then all-red. Recovery
requires fresh frames, two consecutive clear observations and the all-red
interval. Requests cannot bypass this hold. Pausing the simulation also pauses
its control clock. The frame timestamp is recorded with each applied decision.

This is real image processing of screenshots of a simulated scene. The detector
is calibrated to this app's colors, scale and road layout, so ordinary aerial
photos or differently cropped/resized screenshots are not supported. Occlusion,
merged crowds and partial vehicles can produce missed or misclassified objects;
unknown colored fragments inside the box conservatively hold clearance. No
accuracy claim on real traffic, violation classification or street deployment
is implied. The UIT-ADrone photographs remain a separate local dataset, and the
historical traffic-count LSTM is not an image-recognition model.

Checks include raw-pixel recognition, the app's actual canvas renderer, queue
changes affecting applied signals, hidden-state mutations having no control
effect, camera loss and recovery, and image-only crossing/left-lane decisions.

## Train & review

See [the teaching guide](../reports/DECISION_TRAINING_GUIDE.md). The lab accepts real images/video frames and frozen app screenshots, stores explicit human action labels, trains separate models by image domain, and reports held-out group results. The initial street anomaly experiment was trained and evaluated on real images; its failures are visible in the lab and [report](../reports/STREET_VISION_EVALUATION.md). Neither a fitted calibration nor a high score is proof of road-ready decisions.

## Real video demonstration

1. Select **Real traffic videos** and the included FLUID example. Watch the original recording to establish what is visible.
2. Pause at a moment, then choose **Analyze this frame**. The overlay and results refer to that frozen frame and its timestamp. Detector counts are partial observations, not a verified traffic census.
3. Choose **Observe video** to step through spaced frames. This processes sequentially at model speed; it is not a live camera feed. **Stop** cancels the sequence.
4. Collect spaced pictures, then open a saved frame to review it. Return to the monitor without losing the selected video. Saved frames remain unlabelled until a human supplies an action.
5. Download the review package for original pictures and provenance. Keep all frames from the recording in one data split.

The sample is a 60-second, CC BY 4.0 FLUID signalized-intersection excerpt and 60 extracted pictures. Run `python scripts/setup_signalized_video.py` from the repository root to install it on a fresh checkout. [Source, licensing and reproduction](../reports/REAL_TRAFFIC_VIDEO_SOURCE.md).

The pretrained detector can miss many tiny aerial vehicles. It does not read traffic-light color or infer right-of-way. Anomaly scores come from a separate aerial classifier and do not establish any particular violation. The latest retraining failed its validation gate and remains inactive; [actual training results](../reports/STREET_ANOMALY_RETRAINING.md). No physical signal is connected.

A subsequent 320px visual fine-tuning experiment trained the last MobileNetV3 blocks on publisher-labelled images while preserving the separate action encoder. It reduced retrospective test false alerts to 10/180 normal frames, but still detected only 6/45 anomalies and failed the separate development gate. It remains inactive. [Actual fine-tuning method, checks and results](../reports/STREET_ANOMALY_FINETUNING.md). Neither experiment trained the light-choice model.
