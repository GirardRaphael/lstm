# Classroom presenter guide: seeing traffic, explaining decisions

This guide accompanies **Slideshow**, **Live intersection**, and **Train & review**. It supports a 15–20 minute talk, a shorter demonstration, and detailed questions afterward. Implementation and measured results are from November 14, 2026; the lab's live status is the authority for labels or models added later.

## Your central message

> “My project separates what a camera can observe, how a controller chooses a light, and how we test whether a learned model deserves trust. I can demonstrate an operating intersection simulation, inspect real images with a pretrained detector, and collect human examples for a separate light-choice model. The results show useful behavior and important limitations.”

This describes a functioning classroom research prototype. It does not imply that the learned street policy is already trained, that the simulator is a real road, or that 90% reliability has been achieved. A strong presentation explains what works, shows it working, and names the next experiment. Showing that 80% accuracy can miss every anomaly demonstrates scientific judgment rather than hiding a failed experiment.

## What the project actually contains

| Component | Input | Output | Method and status |
|---|---|---|---|
| Live screenshot perception | Pixels from the 800 × 800 simulated intersection | Estimated traffic, queues, waiting and occupied regions | Color/shape recognition and tracking, specialized to these graphics |
| Live signal operation | Observations, requests and elapsed waiting | Phase, duration and explanation | Explicit rules plus a phase/clearance controller; demonstrated operation, not measured real-road superiority |
| Real-image object recognition | Photograph or selected video frame | Boxes, classes and raw scores | COCO-pretrained EfficientDet-Lite2; 80.98% precision and 48.18% recall on a 500-image general benchmark |
| Real-image anomaly experiment | Real aerial frame | Binary normal/anomaly probability | Frozen MobileNetV3Small features and trained logistic classifier; 0 of 45 test anomalies detected at the preset threshold |
| Human light-choice learning | Image features, observed counts and explicit action labels | Suggested action or abstention | Separate classifier, calibration and group-held-out evaluation; initially untrained |
| Historical LSTM forecast | Stored sequence of hourly traffic volumes | Next-volume estimate | Separate recurrent network experiment; it does not choose the intersection lights |

For a spoken map, group these into four tasks: **recognize objects, recognize anomalies, learn human light choices, operate simulated lights**. Show the historical LSTM as a separate forecasting branch. This prevents the word “algorithm” from incorrectly referring to every component at once.

## Before class: practical rehearsal

1. Start `./run_app.ps1` from the repository root and leave its launcher running. Open `http://127.0.0.1:43125/` in the browser you will project.
2. Open **Slideshow** and check local readiness. Confirm the learning service, detector, encoder and dataset status; a fresh checkout does not automatically contain downloaded assets.
3. Open **Train & review** → **Roundabouts · recognition only** → select a thumbnail → **Analyze this image**. First inference may take longer while models load. Confirm the source image, boxes and evidence panels appear.
4. Open **Live intersection**. Confirm traffic moves, **Camera on** is checked, screenshots arrive, and the applied-decision log gains an entry.
5. Rehearse the balanced, east-jam, pedestrian, camera-loss and fixed-cycle demo presets. Use **Return to slide** to keep your place in the talk.
6. Prepare a readable signalized-intersection video or still that you may use. If its visible state cannot justify a light choice, skip the frame instead of inventing a label. The existing roundabout gallery is for recognition inspection, not signal teaching.
7. Decide whether to save a genuine teaching label during class. Saving changes the local collection. Analysis alone does not save an action or train a model.
8. Download this guide, rehearse at the projected resolution, and check readability from the back of the room. Keep detailed notes for speaking rather than reading every line on screen.
9. Keep a captured successful analysis and decision log as a fallback. Describe them as captured results if used; do not present a screenshot as live execution.

The simulation progresses while **Live intersection** is active. Slides and the learning lab pause its progression. Presets make scenarios easier to repeat, but browser timing and image sampling can affect the exact instant of a decision. Seeded scenarios demonstrate behavior; they are not a paired quantitative delay benchmark.

## A 15–20 minute core talk

Use the deck's chapter navigation rather than memorizing slide numbers. These segments are a speaking plan, not a requirement to spend equal time on every slide.

### 0:00–1:30 — Define the problem and the claim

> “A fixed light cycle can serve an empty approach while traffic waits elsewhere. My prototype asks whether image observations can support a transparent choice of the next phase. I also built a separate path for real recognition and human-reviewed learning. I will show the chain from image to explanation and distinguish software behavior from model accuracy.”

Point to the architecture diagram. Name the four tasks and historical forecast. State that live traffic is simulated and real-image detection uses a different perception system. Avoid “This AI is 81% accurate”: precision, recall, anomaly classification and light-choice agreement answer different questions.

### 1:30–3:30 — Explain the observation boundary

Launch the balanced demo. Point first to the intersection, then **What the camera sees**.

> “The renderer produces the scene. The controller's camera input receives screenshot pixels, a capture time and previous pixel detections. It does not receive the hidden vehicle list or future arrivals. It identifies objects, estimates their approaches and tracks them between frames.”

The canvas is sampled approximately every 250 milliseconds. This recognizer is designed for the app's colors, dimensions and layout. Boxes are drawn on a separate display layer, so they do not feed back into recognition. Pixel processing is real, but this particular recognizer is not a general pretrained neural detector.

### 3:30–6:00 — Make the light decision explainable

Launch the east-jam demo. Point to detected queues and **If deciding now**, then wait for **Last applied decision** and the log to update.

> “The rule scores each direction using its visible queue, stopped traffic and waiting time. Higher demand usually receives service, but pedestrian requests, long waits and clearance can take priority. The preview changes continuously; actual changes occur at a permitted phase boundary.”

Use the numerical example below. Explain truck weight two. A request queues a choice; it does not instantly force green. Show a log entry containing phase, actual duration, reason and source-image timestamp. That proves what was applied rather than what a changing preview might select.

### 6:00–8:00 — Show pedestrians and missing input

Launch the pedestrian and camera-loss demos. Show **Camera on** off and then restored; allow recovery and clearance to finish.

> “Scoring is one layer. The phase controller handles yellow, all-red, protected turns and exclusive pedestrian service. Missing observations prevent new green or walk phases. An active green ends through yellow before the red hold.”

The last image may remain visible while stale input is rejected. This demonstrates a specified response to input loss, not proof that every real pedestrian or hazard will be detected.

### 8:00–10:30 — Show real recognition and interpret 81%

Open **Train & review**, select a real gallery frame and analyze it. Point out a visible detection and, where present, a missed object or questionable box.

> “This is a real photograph processed by pretrained EfficientDet-Lite2. It finds objects, not the best light action. On 500 fixed benchmark images, precision was 80.98% and recall 48.18%. Most accepted boxes were correct, but roughly half the labeled objects were missed.”

The benchmark is general COCO imagery containing vehicles, not an independent test of this aerial intersection. Box scores are raw model outputs, not guarantees of correctness.

### 10:30–12:30 — Explain why more data alone is insufficient

Show data partitions and anomaly results.

> “I expanded the aerial collection from 1,500 to 3,000 distinct real images, keeping whole recordings separate across training and evaluation. Their labels mean normal or anomalous; they are not object boxes or correct light actions. The expanded anomaly model still missed all 45 test anomalies. Its 80% accuracy matched always predicting normal.”

Point to the confusion matrix. Ranking average precision improved retrospectively, but the fixed decision threshold remained ineffective. This is why overall accuracy alone is inadequate.

### 12:30–15:30 — Teach an action and explain training

Use a prepared compatible real image/video, or explicitly demonstrate the simulation domain. Analyze the exact frozen frame, explain the visible evidence, select an action and **Save teaching example** only if it is a genuine defensible preference. Show label and recording-group counts.

> “The frame supplies input features and my explicit choice supplies the supervised target. Viewing an image does not train the model; saving does not immediately retrain it. Fitting learns the classifier, calibration uses separate recordings, and evaluation measures agreement on other recordings.”

The initial action model is untrained. If training is disabled, explain the requirement rather than fabricating examples: at least 80 labels, 12 independent groups and eight examples per action overall, plus partition requirements. These minimums permit an experiment, not a dependable street system.

### 15:30–17:30 — Separate confidence, evidence and the LSTM

Explain that a **90% score gate** selects which recommendations to show, while **measured reliability** requires held-out agreement, coverage and uncertainty.

> “A 90% cutoff cannot manufacture 90% reliability. We must compare recommendations with reviewed labels and report how often the model declines to answer.”

If time permits, open **Historical LSTM forecast — separate from light control**. It predicts traffic volume from a historical sequence. It does not recognize objects or decide the current light. Forecast errors do not belong in the vision accuracy claims.

### 17:30–20:00 — State the contribution and next experiment

> “The contribution is an inspectable prototype: pixels become observations, observations become an explained control decision, and reviewed real examples can train a separate policy. Errors are visible rather than hidden behind one number. My next experiment needs independently labeled signalized-intersection recordings, controlled comparisons and calibration.”

End with the evidence and roadmap. Invite the class to distinguish a software test, model benchmark and real-world validation claim.

## Exact demonstration paths

### A. Queue response

Use the deck's east-jam preset, or:

1. Open **Live intersection**; enable automatic control and **Camera on**.
2. Select **East** under **Jam approach** and press **Create traffic jam**.
3. Point to the queue estimate, current phase and preview.
4. Wait for yellow/clearance and a new applied decision. Pedestrian or long-wait priority may legitimately go first.
5. Press **Pause** to explain a stable log entry and **Resume** to continue.

Sliders and number fields set the next spawn counts. Moving them alone does not add traffic; press **Spawn traffic**. **Randomize & spawn** selects new counts and spawns them. **Random jam** chooses an approach.

### B. Pedestrian service

Use the pedestrian preset, or create a visible crowd with the pedestrian count and **Spawn traffic**, then press a **Crosswalk** button, **NS** or **EW**. Observe the request, vehicle clearance, exclusive walk and return to vehicle service. It is a queued request, not an immediate green override.

### C. Missing camera input

Start a populated live scene and wait for a green phase. Turn **Camera on** off in **What the camera sees**. Show the camera hold, completed yellow and all-red. Turn the camera back on and allow fresh readable observations and clearance to arrive. Keep the simulation running: pausing also pauses the clock, so it cannot demonstrate elapsed stale-input time.

### D. Fixed-cycle comparison

Use the fixed-cycle preset, or uncheck **Automatic traffic controller (off = fixed cycle)**. Through greens alternate at six simulated seconds; pedestrian/turn phases and clearance still exist. Re-enable automatic control to inspect demand-based durations.

Describe the behavior you can see. Do not claim a percentage reduction in waiting time from this demonstration. Quantitative comparison needs matched arrival demand and seeds, repeated runs, consistent warm-up and horizon, and recorded delay, throughput, pedestrian wait and maximum queue.

### E. Real-image inspection

Open **Train & review** → **Roundabouts · recognition only** → select a thumbnail → **Analyze this image**. The gallery uses training images. Read the predicted boxes separately from measured evidence. Roundabout labels support binary anomaly experiments and are deliberately unavailable for signal-action teaching.

### F. Save a real teaching label

Open **Your image or video** and choose a compatible signalized-intersection image or video. For video, seek and press **Capture this video frame**. For extracted stills, preserve the same **Recording name for still images** across frames from one recording. Analyze, choose an action, and **Save teaching example**.

North/south means the vertical road in this view; east/west means horizontal. A rotated or complex layout may not support this mapping. A still may omit signal state, trajectories, elapsed wait or hidden traffic. Skip unsupported choices; select hold only when it is a defensible reviewed preference. Arbitrary labels added merely to enable training damage the experiment.

### G. Review a simulation screenshot

Open **Live intersection**, wait for a captured frame, then press **Teach a decision from this screenshot**. The lab automatically opens **App screenshot** with the captured frame frozen. Press **Analyze this image**. Save a genuine preference if appropriate. **Apply my reviewed choice in the simulation** sends that reviewed request through the existing controller; it does not enable an autonomous learned street policy.

Street and simulation labels train separate models. One simulation episode is one group. Many frames in one episode are not independent groups; resetting the same prepared scene repeatedly also does not create new environmental diversity.

## Technical explanation and speaking notes

### 1. How screenshots become traffic observations

Raw RGBA pixels enter the recognizer. The known palette and shapes identify connected components representing cars, trucks and pedestrian heads. Calibrated regions assign them to approaches and turning lanes. Matching detections between frames estimates motion and waiting. Visible junction/crossing occupancy participates in clearance.

A truck contributes two passenger-car units to queue/stopped weights; a car contributes one. This is a modeling simplification, not a measured capacity ratio for every vehicle. The queue is visible approaching demand, not an independently audited physical queue length in metres. Occlusion and merged shapes can change estimates.

This narrow recognizer expects the app's 800 × 800 drawing and color conventions. Reading actual pixels is meaningful, but it does not imply generalization to arbitrary photographs, angles or resizing. The pretrained street detector addresses a different image domain.

### 2. The rule and a worked calculation

For each axis, combine its two approaches:

`score = 2 × queue_weight + stopped_weight + 0.3 × red_wait_seconds`

Automatic through-green duration is:

`green_seconds = clamp(4 + 0.5 × selected_queue_weight, 4, 16)`

`clamp` limits the result to four through sixteen simulated seconds. Suppose no priority exception applies:

| Quantity | North/south | East/west |
|---|---:|---:|
| Queue weight | 4 | 9 |
| Stopped weight | 2 | 7 |
| Waiting for green | 10 seconds | 4 seconds |
| Score | 2×4 + 2 + 0.3×10 = **13** | 2×9 + 7 + 0.3×4 = **26.2** |

East/west wins and receives `4 + 0.5×9 = 8.5` simulated seconds of through green. This illustrative calculation is not a promise about the next live scene, whose observations and priority conditions change.

The coefficients are an explicit engineering rule. They were not learned by the LSTM, detector or 3,000 anomaly images. Empirical justification would require controlled comparison of controller variants.

### 3. Priority, clearance and time

The phase controller determines when a choice can be applied. At an eligible all-red boundary, an accepted pedestrian request comes first. Automatic service can be requested when a detected pedestrian has waited at least 20 seconds, or at least six have waited at least 12 seconds, subject to post-walk cooldown. Manual crosswalk requests can also be queued.

For vehicle directions in automatic mode:

1. A direction with visible demand waiting at least 30 seconds gets priority; the longer wait wins if both qualify.
2. Otherwise consider a queued operator direction request.
3. Otherwise compare scores. Skip an empty axis when the other has demand; equal scores use the alternating next axis.

The 30-second rule is priority at a decision boundary, not a guaranteed maximum wait. Yellow, occupancy and committed phases still need to finish.

Simulation parameters include 1.4 seconds of yellow, 0.8 seconds of all-red and a six-second exclusive walk. Protected left phases have separate bounded durations and wait for clearance. These are demonstration parameters in simulated time, not recommended street timing specifications. The animation advances simulation time faster than wall-clock time.

A missing, unreadable, future-dated or more-than-one-simulation-second-old frame cannot authorize a new phase. Recovery requires fresh input, clear observations and the all-red interval. Unknown fragments or missed/merged objects can still affect perception; clearance logic does not prove all real hazards are covered.

### 4. What EfficientDet-Lite2 contributes

The official float32 EfficientDet-Lite2 model supplies visual knowledge pretrained on COCO. The local app runs it through MediaPipe and reports people, bicycles, cars, motorcycles, buses, trucks, traffic lights and stop signs. A traffic-light box identifies an object; it does not establish its active color or the movement it governs.

The implementation uses a whole-image pass and overlapping crops for larger images, maps boxes back to image coordinates and merges same-class duplicates. This can preserve small-object detail. Its score cutoff is 0.3. Scores are uncalibrated outputs. A large aerial training image took about 2.38 seconds for nine passes locally; the general 500-image benchmark averaged 280 milliseconds per image. Neither is a guaranteed end-to-end video frame rate.

Advantages include existing visual knowledge, a relatively compact local runtime and immediate object-level output. Limitations include small objects, false positives, occlusion, domain shift and crop latency. The detector was not fine-tuned locally: the aerial anomaly labels lack verified boxes.

Inference runs locally. The app does not upload images or teaching labels to a hosted model service. MediaPipe may report SDK usage metrics; the project links Google's statement that input images are not sent. Do not promise a telemetry-free SDK.

### 5. Precision, recall and the actual benchmark

The fixed class-matched box-overlap threshold is 0.5. Recorded counts are 1,933 true-positive detections, 454 false-positive detections and 2,079 missed ordinary objects. The dataset contains 49 crowd annotations excluded from the ordinary recall denominator; 93 predictions overlapping qualifying crowds are ignored under the stated protocol.

`precision = TP / (TP + FP) = 1933 / 2387 = 80.98%`

`recall = TP / (TP + FN) = 1933 / 4012 = 48.18%`

`F1 = 2 × precision × recall / (precision + recall) = 60.42%`

Precision asks whether reported boxes are correct. Recall asks whether labeled objects were found. Each match needs the correct class and sufficient overlap. These statistics are not COCO mean average precision, classification accuracy, or the probability that a light change is safe.

The 500 vehicle-containing COCO validation images were selected before inference. This does not measure all-negative scenes. The publisher may have used COCO validation during model development, so it is a public baseline rather than a fresh private intersection holdout.

The recognition target is at least 90% precision **and** recall for every displayed road class, on at least 500 images and with at least 20 ground-truth instances per class. It is not met.

### 6. Aerial anomaly data and training

The collection expanded from 1,500 to 3,000 distinct real aerial images. Complete source recordings are separated:

| Partition | Images | Normal | Anomalous | Recordings |
|---|---:|---:|---:|---:|
| Fit | 2,550 | 1,412 | 1,138 | 35 |
| Calibration/validation | 225 | 180 | 45 | 8 |
| Test | 225 | 180 | 45 | 8 |

New images came only from the original training recordings; all 450 held-out images remained unchanged. All image hashes are unique. Frame spacing reduces near duplication, but frames in one video remain correlated. More frames from 35 recordings do not create more cameras or environments.

The original subset and both holdouts retain 80% normal / 20% anomaly. Expanded training is approximately 55.37% normal / 44.63% anomaly. Describing the entire expanded collection as uniformly 80/20 would be inaccurate.

Training uses frozen ImageNet MobileNetV3Small features, standardization fitted on training data only, and balanced L2 logistic regression with fixed `C=0.1`. A sigmoid calibrator uses validation scores. The threshold remains 0.5; convolutional weights are not fine-tuned.

| Actual label | Predicted normal | Predicted anomaly |
|---|---:|---:|
| Normal | 180 | 0 |
| Anomaly | 45 | 0 |

Accuracy is 80%, anomaly recall 0%, and balanced accuracy 50%. Probabilities are nearly constant around 20%. Calibration can make a base-rate prediction look plausible without making individual cases distinguishable.

A separate higher-resolution candidate found 42 of 45 anomalies but flagged 161 of 180 normal frames: 93.3% recall with only 20.7% precision. Quoting recall alone would hide false alarms. The expanded baseline remains the documented default. Repeated comparisons used an inspected test set, so they are retrospective; untouched recordings are needed for confirmation.

### 7. Human light-choice learning

The human selects one target: `ns`, `ew`, `walk`, or `hold`. A saved example includes the exact frozen image identity, feature version, domain and recording group. Duplicate images cannot silently count as new independent examples. Correcting a label replaces it and invalidates the old model until retraining.

Street input combines 576 frozen scene features with 72 object features: eight classes counted in each cell of a 3 × 3 grid, transformed with `log(1 + count)`. This describes approximate distribution and appearance; it is not a full road map, trajectory history or verified signal state.

Training fits a small regularized logistic classifier to human labels. Standardization uses fitting groups only. Temperature calibration adjusts probabilities using different groups. Evaluation uses reserved test groups. Initially, approximately half the recordings go to fitting and one quarter each to calibration and test. Existing assignments persist rather than being shuffled until results improve.

Overall requirements are 80 labels, 12 groups and eight examples per action. Fitting also needs at least 40 images, three groups and three labels per action; calibration and test each need at least 16 images, three groups and two per action. Meeting headline totals can still leave one partition short. Add independent examples rather than renaming a recording to evade grouping.

The model learns agreement with reviewed preferences. Incorrect or inconsistent labels can teach errors. A written rubric, multiple reviewers and adjudication are useful next steps. Human agreement does not demonstrate optimal throughput, lawful control or safety.

### 8. Calibration and the two 90% concepts

Training fits the mapping to labels. Calibration adjusts scores against outcomes in separate data. Evaluation measures behavior on data excluded from both steps.

A recommendation needs a calibrated score of at least 0.90 and a minimum similarity to training features. Otherwise the model abstains. Similarity is a coarse unfamiliar-input check, not comprehensive distribution-shift detection.

The separate evidence target needs at least 40 accepted test predictions across three held-out groups, at least five examples of every action, and an approximate 95% Wilson agreement interval whose lower bound reaches 90%. It also reports **coverage**, the proportion of test frames receiving a recommendation. Declining nearly every case is not automatically useful because a few accepted cases are correct.

Nearby frames are correlated, so a frame-based interval is descriptive and may overstate independent evidence. Repeatedly inspecting and retraining against the same test groups is retrospective. New recording-level confirmation remains necessary.

### 9. Why the project includes an LSTM

An LSTM is a recurrent network with gates controlling how sequence information is retained, updated and exposed. It is reasonable to investigate for traffic-volume forecasting because future counts can depend on recent counts and recurring temporal patterns.

The historical baseline artifact uses 24 observations, stacked LSTM layers with 64 and 32 units, dropout, a 16-unit dense layer and a next-volume output. The app displays an example selected from stored traffic data and either replays saved results or runs extracted weights locally. Scene intensity helps choose the historical example; it is not a live motorway sensor or direct forecast of the simulated intersection.

The source data has missing hours and duplicate timestamps. The newer forecast evaluation considers cadence and baselines. Do not describe the old 24-observation baseline as a flawless 24-consecutive-hour experiment.

The newer retrospective motorway comparison selected XGBoost, with MAE 153.84 vehicles/hour versus 277.98 for LSTM. The tree used practical features including seasonal lags, so this is not an architecture-only comparison. It illustrates why the claim should be “a tested candidate for this task,” not “LSTM is universally best.” Neither forecast result validates detection or light control.

## Common classroom questions

**Does it really control the lights?**

Yes, it controls simulated phases and durations from screenshot observations. The learned street-action model is separate and initially untrained. No physical signal equipment is connected.

**Is all the data fictional?**

The live scene is simulated, and camera mode processes its pixels rather than its hidden vehicle objects. The lab separately processes real photographs and video frames. Each domain has separate evidence and limitations.

**Is 81% reliability good?**

The number is 80.98% detection precision on a particular benchmark. Recall is 48.18%. That is a useful baseline for inspection, not 81% overall reliability or proof of suitability for control.

**Why not call it 90% reliable when the threshold is 90%?**

A threshold selects which answers to show. Reliability must be measured against held-out labels, with coverage and uncertainty. An incorrect or miscalibrated model can still output a high score.

**Why did doubling the images not solve anomalies?**

Examples help only if their labels and representations capture the relevant information. Tiny actors, motion, camera differences and label ambiguity remain difficult. New frames came from existing training recordings, and the frozen whole-image representation still failed at the fixed threshold.

**Did you train EfficientDet on those 3,000 images?**

No. The detector is pretrained. Those images trained a separate binary anomaly head. Detector fine-tuning requires verified object boxes and classes, which the anomaly dataset does not supply.

**Can I correct a vehicle box in the app to improve recognition?**

Not yet. The current interface saves human light-action labels. Manual box editing, corrected-object export and detector fine-tuning remain future work. Predicted boxes are not verified ground truth.

**Can it detect jaywalking or red-light violations?**

Not as a validated reliable capability. Finding a person or car differs from establishing a violation. Many events require trajectories, crossing geometry, signal state, timestamps and explicit event labels. The current anomaly model is inadequate.

**Why logistic regression rather than a larger neural policy?**

A small regularized classifier on frozen features is reproducible and inexpensive to retrain when reviewed labels are scarce. It is a useful baseline, but can miss complex temporal/spatial relationships. Stronger candidates need comparable data and evaluation.

**Why not reinforcement learning?**

It could be investigated with an explicit reward, realistic simulator and careful evaluation. This project currently uses human-supervised preferences and operating rules; it has not trained a trial-and-reward agent. A useful reward and realistic environment are substantial research work.

**Do passing tests mean good traffic decisions?**

No. Tests verify specified software behavior such as request handling, grouping, persistence and camera-loss transitions. Model benchmarks measure predictions. Controlled traffic experiments measure delay or throughput. Deployment needs a further engineering validation process.

**Which algorithm is the best?**

No universal winner is demonstrated. “Best” requires a specific task, metric, resource budget and comparable held-out experiment. Each current choice has practical reasons and measurable limitations.

## A credible route to stronger performance

For the next milestone, prioritize a complete experiment over a larger headline dataset:

1. Define one camera view and target task. Separate object detection, temporal violations and human action preferences in the labels and evaluation.
2. Collect diverse, permitted signalized-intersection recordings across traffic levels, lighting, occlusion and weather. Preserve source identity and reserve new sites/recordings before tuning.
3. Produce verified object boxes/classes for detector fine-tuning. Add trajectories, road geometry and signal-state labels when needed. Build or integrate a real annotation/correction workflow; it is not implemented in the current lab.
4. Use a written action rubric and multiple competent reviewers. Resolve disagreements and retain ambiguity. Do not treat the model's own guesses as unquestioned correct labels.
5. Fit candidates using training recordings and separate calibration. Compare frozen versus fine-tuned recognition, temporal/tracking approaches and simple baselines with the same data and declared compute budget.
6. Freeze settings before the independent test. Report per-class precision/recall, missed pedestrians, false alarms, latency, calibration and abstention coverage across conditions, including failures.
7. Compare control policies using repeated simulation runs with matched arrivals. Measure delay, throughput, queue length, pedestrian waiting, starvation and sensor-failure behavior. Do not infer traffic outcomes from detection precision.
8. Before physical deployment, involve traffic-control professionals, implement an independent fail-safe operating layer, validate hardware/timing interfaces, address monitoring/security/data governance, and perform staged observation-only trials and formal acceptance review. The current app does not implement those deployment requirements.

More data helps when it adds relevant, well-labeled diversity. Raising an interface threshold, duplicating nearby frames or selecting favorable results cannot establish 90% reliability.

## Troubleshooting during the talk

| Symptom | Meaning or first check | Presenter action |
|---|---|---|
| App opens in learning lab | Intended default | Choose **Slideshow** |
| Traffic does not move | Not in Live intersection, or paused | Open live mode; press **Resume** if shown |
| Slider has no immediate effect | It sets next spawn counts | Press **Spawn traffic** |
| Requested direction is delayed | Priority/clearance may precede it | Explain current phase and applied log; avoid repeated clicks |
| Lights hold red | Missing camera, occupied clearance or reviewed hold | Inspect status, camera and clock; use a prepared preset for a fresh scene if needed |
| Old image remains after camera off | Retained preview is not fresh input | Show camera status and hold |
| Gallery is absent | Local data not in fresh checkout | Upload an available image and disclose missing local gallery |
| Detector unavailable | Service, runtime or weights missing | Read readiness result; use recorded benchmark while preparing assets |
| First inference is slow | Initialization or multiple crops | Let it finish; avoid duplicate submissions |
| Video fails | Codec/file incompatibility | Try a supported MP4 or still from the same recording |
| Training is disabled | Labels/groups/actions insufficient | Show requirements; do not fabricate examples |
| Partition shortfall after training attempt | Overall counts do not guarantee split coverage | Add independent groups/actions without changing existing identities |
| No recommendation | Untrained model or abstention | Explain why withholding an unsupported choice is intentional |
| No real-image boxes | Nothing passed detector threshold | Do not infer empty road; inspect missed objects |
| Anomaly probability near 20% | Known default experiment failure | Show confusion matrix and always-normal baseline |

## Shorter versions

### Fifteen minutes

Use two minutes for scope, four for queue/pedestrian/camera demos, three for real detection and metrics, three for labeling/calibration, and three for limitations/questions. Leave LSTM architecture and detailed training equations for questions. Save a label only if prepared and defensible.

### Five minutes

1. **0:00–0:45:** Separate simulation control, real recognition and reviewed learning.
2. **0:45–2:00:** East-jam demo; show an applied decision with input and reason.
3. **2:00–2:45:** Camera-loss hold and recovery.
4. **2:45–3:45:** Real-image analysis; explain 80.98% precision and 48.18% recall.
5. **3:45–5:00:** Label interface, untrained status if applicable, and the next independent-data experiment.

Skip unprepared live training. Showing how a defensible label enters the pipeline is stronger than rushing to fit inadequate data.

## Evidence and implementation references

- [Decision training guide](evidence/DECISION_TRAINING_GUIDE.md): workflow, grouping, calibration and evidence targets.
- [Detector evaluation](evidence/PRETRAINED_DETECTOR_EVALUATION.md): fixed sample, matching, class results and reproduction.
- [Aerial data expansion](evidence/STREET_VISION_DATA_EXPANSION.md): counts, recordings and unchanged holdouts.
- [Expanded anomaly evaluation](evidence/STREET_VISION_EXPANDED_EVALUATION.md): training configuration, actual confusion matrix and limitations.
- [Higher-resolution candidate](https://github.com/GirardRaphael/lstm/blob/main/reports/STREET_VISION_CANDIDATE_EVALUATION.md): sensitivity/false-alarm tradeoff; not the default model.
- [Hourly forecasting evidence](evidence/REPORT.md): separate historical comparisons and retrospective scope.
- [Presentation README](evidence/LIVE_INTERSECTION.md): startup and exact live behavior.
- [Traffic scoring](https://github.com/GirardRaphael/lstm/blob/main/presentation/src/trafficController.ts), [phase control](https://github.com/GirardRaphael/lstm/blob/main/presentation/src/simulation.ts), [screenshot perception](https://github.com/GirardRaphael/lstm/blob/main/presentation/src/screenshotVision.ts), [human learning](https://github.com/GirardRaphael/lstm/blob/main/presentation/server/decision_learning.py): implementation details.


## The roundabout training obstacle

The built-in roundabout gallery is now labeled **Roundabouts · recognition only**. Those publisher anomaly labels cannot teach traffic-light actions. The lab opens on **Your image or video**, with a visible **Create a signalized practice scene** button. That button creates a new simulated intersection episode. After the scene develops, **Teach a decision from this screenshot** automatically selects App screenshot and freezes the captured image. Press **Analyze this image**, choose your own action, then **Save teaching example**. No action is preselected and no label is saved automatically.

Practice labels train the simulation model only. Real-world action training still needs compatible signalized footage and reviewed labels. A model remains untrained until its data requirements are met: 80 human-labeled images, 12 recording/session groups, at least eight of each action overall, and enough action coverage in each partition. A rehearsal should demonstrate one saved correction and the remaining requirements honestly if the class begins before data collection is complete.

We checked the official [SinD signalized-intersection dataset](https://github.com/SOTIF-AVLab/SinD) as a better-matched data source. Its public repository provides trajectory samples and annotated visual previews; full data access requires an educational-email application under its [access instructions](https://github.com/SOTIF-AVLab/SinD/blob/main/DATASETS.md). No raw footage from that source has been added to this project, no access request has been sent, and annotated preview images were not used as recognition training examples.
