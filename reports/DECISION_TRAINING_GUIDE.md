# Teaching decisions from street footage

Open **Train & review** in the intersection app. It is now the default view.

This workspace separates four tasks:

- **Object recognition:** Google's COCO-pretrained EfficientDet-Lite2 locates people, vehicles, traffic lights and stop signs in actual image pixels. Its uncalibrated detection scores are not signal-action confidence. See [the measured object benchmark](PRETRAINED_DETECTOR_EVALUATION.md).
- **Street anomaly recognition:** a trained image classifier estimates the publisher's binary normal/anomaly class. Its measured performance is shown; fitting a calibrator does not establish reliability.
- **Learning your light choices:** a separate model learns only from actions you explicitly label. It cannot inherit correct signal actions from anomaly labels.
- **Simulated light operation:** the existing screenshot-based rules and clearance phases remain available under Live intersection. Learned recommendations require explicit human review.

## Use the lab

1. Inspect a real aerial training example to see an actual model output, or choose **Your image or video**.
2. For a video, seek and capture a frame. The video stays in the browser; only your selected image goes to the local service. All frames from the same file share a recording group.
3. Use a signalized intersection recording for light-choice teaching. North/south means the vertical road in that view; east/west means the horizontal road. State, visibility and missing motion may make a frame ambiguous.
4. Analyze the frozen image, choose **North / south**, **East / west**, **Pedestrian walk** or **Hold all red**, then **Save teaching example**. Selecting a model suggestion still requires your explicit save.
5. Use the same recording name for extracted stills from one video. Do not invent different groups to reach the training requirement.
6. After collecting sufficient labels and groups, press **Train decision model**. Inspect the held-out results before treating its recommendations as useful.

For simulation teaching, visit Live intersection, let a screenshot arrive, and choose **Teach a decision from this screenshot**. Reset starts a new simulation episode/group. All screenshots from one episode stay together. Street and simulation feedback train separate models.

Applying a reviewed simulation choice sends an operator request through the existing clearance controller. Hold ends active green through yellow and then holds red. Releasing hold still requires current camera input and clearance. Uploaded street images do not operate the simulator or any physical signals.

## What is trained

The street anomaly experiment uses an ImageNet-pretrained MobileNetV3Small encoder and a supervised, regularized logistic classifier. Real UIT-ADrone frames supply normal/anomaly labels. Training, probability calibration and evaluation use disjoint source recordings. See [the actual experiment report](STREET_VISION_EVALUATION.md), including failures and limitations. This is image classification, not a trained vehicle-box detector, and still images cannot establish all movement-dependent violations.

The street light-choice model uses 576 frozen image features plus 72 observed-object counts: eight road-object classes in a 3-by-3 image grid. Counts are transformed with log(1 + count). Thus detected object locations feed the model, alongside scene appearance and your four-class action labels. The detector's weights and feature format are included in the feature-version identity; incompatible feedback cannot silently mix. Simulated graphics have their own model and do not use the real-image detector. The small classifier is intentionally easy to retrain and evaluate; it is not demonstrated to be the best architecture. StandardScaler fits only the training groups. A fixed logistic-regression classifier is fitted there, scalar temperature calibration uses separate groups, and remaining groups provide evaluation. Confidence estimates agreement with human preferences; it is not the probability that an action is safe, lawful, or optimal.

Initial group assignment reserves approximately half the groups for fitting and a quarter each for calibration and evaluation. Assignments persist. Requirements are at least 80 human-labeled images from 12 independent recording/session groups, with at least eight labels of each action overall. Each partition must also contain enough groups and action examples; the service explains any shortfall. These minimums make a small experiment possible, not a road-readiness claim.

A fixed 90% probability threshold and a coarse feature-similarity check cause uncertain recommendations to abstain. Similarity is not a guarantee of detecting unfamiliar scenes. Repeated training against already inspected evaluation groups is marked retrospective. Correcting a label invalidates the previous model until retraining. Identical images cannot be placed in different groups or silently counted twice.

A separate **90% reliability evidence target** measures agreement with held-out human labels; it is not set merely by changing a probability cutoff. The gate requires at least 40 recommended test frames across three held-out recording groups, at least five examples of each action, and an approximate 95% Wilson interval lower bound of 90%. Nearby video frames remain correlated, so the frame interval is descriptive and does not establish independent recording-level confidence. Coverage includes both probability and image-similarity abstention. The app shows failure to meet the target honestly.

## Why these algorithms

EfficientDet-Lite2 is a pretrained object detector that shares a feature network across objects and runs locally through MediaPipe. Reusing its learned visual representation is more practical than learning vehicles from a few local frames. This project measures actual local latency; a vendor benchmark on a different device is not our frame rate. Its larger 448-pixel input is useful for detail, but small aerial objects, occlusion and domain shift can still cause missed detections.

For images larger than 960pixels on their longest side, the detector adds overlapping 640-pixel crops with 512-pixel spacing to its whole-image pass. It maps boxes back to the original image and merges same-class duplicates above 0.5 overlap. This preserves more detail for small aerial objects. One 1920-by-1080 training frame required nine passes and about 2.38 seconds; the 500-image general benchmark averaged 280 milliseconds per image. These are local observations, not a real-time video guarantee.

Regularized logistic regression provides a small, reproducible supervised baseline for both anomaly labels and action preferences. Frozen visual features reduce the number of trainable parameters. Training on human corrections is genuine supervised learning: fitting changes classifier weights. Temperature/sigmoid calibration then fits probabilities on separate data. The poor aerial anomaly results show why interpretable training and honest measurement are necessary; model complexity or a pretrained name alone does not establish accuracy.

The existing LSTM forecasts hourly traffic volume. It does not recognize objects or learn the best light phase from this footage. Motion-dependent violations need sequences, tracking, visible signal state and appropriate labels. A single frame cannot establish waiting duration or whether a vehicle crossed while a light was red.

## Data and reproducibility

- The original 1,500-frame aerial subset remains under ignored `data/intersection_anomalies/`. The gallery shows its training partition; the report covers held-out evaluation. Publisher anomaly labels are never converted to light labels. The roundabout examples are therefore disabled for signal-action teaching.
- Human feedback, frozen snapshots and local decision models persist under ignored `var/traffic_learning/`. Inference runs locally. The MediaPipe binary may report SDK usage metrics; [Google states that input images are not sent](https://github.com/google-ai-edge/mediapipe/issues/6291). There is no official SDK switch to disable those metrics. The app itself does not upload images, human labels or snapshots to a model service.
- `GET /api/learning/export?domain=street` (or `simulation`) exports the human labels, image hashes, embedding version and grouping. Corresponding PNGs remain in the local snapshots folder.
- Run `python scripts/setup_street_encoder.py` to prepare the frozen scene encoder without needing the aerial dataset. This alone does not train an anomaly classifier or a light policy.
- Run `python scripts/setup_pretrained_detector.py --install-runtime` to install the pinned local detector runtime and download verified public model assets. Runtime files stay under ignored `.runtime/`; setup does not replace the main Python environment.
- Frozen public image-encoder and anomaly model packages can be retained under `models/street_vision/`; new experiments default to ignored `.runtime/street_vision/`. When present, `.runtime/street_vision_expanded/` and `data/intersection_anomalies_expanded/` are the lab defaults. The original dataset and baseline are retained for audit. The expanded dataset contains 3,000 real images; see [its counts and split audit](STREET_VISION_DATA_EXPANSION.md) and [actual retraining result](STREET_VISION_EXPANDED_EVALUATION.md). `TRAFFIC_STREET_MODEL_DIR` selects a specific package, and `TRAFFIC_IMAGE_DATASET` selects the local image folder.
- Run `python scripts/train_street_vision.py --data-dir <dataset folder>` to reproduce the declared baseline in a new output directory. Existing evaluated artifacts cannot be overwritten silently.

The local service binds localhost and rejects requests from unrelated web origins. Uploads accept bounded PNG/JPEG/WebP images. A fresh checkout without image/model assets reports their absence rather than fabricating predictions.

## Evidence limits

Software tests check persistence, image identity, domain separation, group isolation, calibration independence from test labels, abstention and hold behavior. They do not demonstrate lower traffic delay or safe street deployment. The learned light model starts untrained because no human signal-action labels have been supplied. Real signalized footage and reviewed action labels are still needed to evaluate those decisions.

The public pretrained detector is not fine-tuned on the local aerial anomaly labels: those labels contain no object boxes. The expanded aerial experiment trains the anomaly head, while the object benchmark measures the frozen pretrained detector. Neither dataset supplies correct light actions. Claiming that it did would invent supervision.
