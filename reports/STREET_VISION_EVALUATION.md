# Real aerial-frame anomaly experiment

Generated 2026-11-14T15:58:53.053746+00:00. Model `92efc8827f43e940-13fe8d9793e4`.

**Outcome: inadequate at the preset threshold.** The model missed all 45 held-out anomaly frames and predicted normal for all 225 test frames. Its probabilities were effectively constant at 20%; the 80% accuracy is the always-normal baseline, not useful anomaly detection. It must not be used to justify confidence in signal decisions.

This experiment actually trained a binary classifier on real UIT-ADrone aerial images. It does **not** validate object detection or traffic-light decisions. The current screenshot detector and light controller are separate systems.

## Fixed experiment

ImageNet-pretrained MobileNetV3Small was frozen; each full image was resized to 224x224 using bilinear sampling. Its pooled visual embedding was standardized using training data only. L2 logistic regression (C=0.1, balanced training class weights, seed 42) learned the binary publisher anomaly labels on the training split. A second one-variable logistic model (C=1.0, natural class prevalence) fitted sigmoid probability calibration on validation scores only. The review threshold was fixed at 0.5 before test evaluation. No hyperparameter search or threshold selection used test data. No full neural-network fine-tuning was performed.

| Split | Images | Normal | Anomaly | Source recordings |
|---|---:|---:|---:|---:|
| train | 1050 | 840 | 210 | 35 |
| validation | 225 | 180 | 45 | 8 |
| test | 225 | 180 | 45 | 8 |

Whole source recordings are disjoint across these splits. Exact image hashes were also checked for cross-split duplication. These are new project splits of the selected publisher data, not a claim to reproduce the original paper benchmark. Nearby frames within a recording are correlated, so 225 test frames are not 225 independent street environments.

## Held-out result

| Metric | Calibrated model | Always normal (p=0) | Training prevalence (p=0.2) |
|---|---:|---:|---:|
| Accuracy | 0.8000 | 0.8000 | 0.8000 |
| Anomaly precision | 0.0000 | 0.0000 | 0.0000 |
| Anomaly recall | 0.0000 | 0.0000 | 0.0000 |
| Anomaly F1 | 0.0000 | 0.0000 | 0.0000 |
| Balanced accuracy | 0.5000 | 0.5000 | 0.5000 |
| Average precision | 0.3099 | 0.2000 | 0.2000 |
| ROC AUC | 0.6005 | 0.5000 | 0.5000 |
| Brier score (lower is better) | 0.1600 | 0.2000 | 0.1600 |
| 10-bin ECE (lower is better) | 0.0498 | 0.2000 | 0.0000 |

Confusion matrix, rows=true [normal, anomaly], columns=predicted [normal, anomaly]: `[[180, 0], [45, 0]]`. Undefined positive precision for an always-normal classifier is reported as zero. AP uses ranked probabilities; it does not use the 0.5 threshold.

The model's Brier score is lower than the constant training-prevalence predictor, and its balanced accuracy is not higher than the always-normal baseline. Fitting a calibrator is **not proof of reliable probabilities**. The table and bins are the measured evidence, and no real-road deployment acceptance criterion has been met.

## Test reliability bins

| Estimated probability interval | Frames | Mean predicted probability | Observed anomaly fraction |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 198 | 0.2000 | 0.1717 |
| 0.2-0.3 | 27 | 0.2000 | 0.4074 |
| 0.3-0.4 | 0 | - | - |
| 0.4-0.5 | 0 | - | - |
| 0.5-0.6 | 0 | - | - |
| 0.6-0.7 | 0 | - | - |
| 0.7-0.8 | 0 | - | - |
| 0.8-0.9 | 0 | - | - |
| 0.9-1.0 | 0 | - | - |

ECE is the frame-weighted absolute difference between the final two columns across ten equal-width bins. Small bins and correlated frames make this estimate unstable; no independent multi-site calibration claim is made.

## Limitations and scope

- Publisher binary labels have not been independently adjudicated; anomalies have no verified subtype or boxes.
- Only 45 positive test frames from a small number of held-out recordings; nearby frames are correlated.
- Frozen 224px ImageNet features can miss small actors, motion-dependent violations, signal state, and other street details.
- Calibration is fitted on 225 validation frames and may fail under new cameras, geography, illumination, or synthetic graphics.
- This is a binary image classifier, not a vehicle detector, a learned light policy, or evidence that a signal action is safe.

A 'review' prediction asks a person to inspect a frame. It does not authorize a light change. Anomaly probability is not the probability that a selected signal action is correct. Real-footage policy labels and separate group-held-out policy validation are still required to establish decision quality.

## Reproduction and artifacts

Run `python scripts/train_street_vision.py --data-dir <local intersection_anomalies folder>` once. The script refuses to overwrite an evaluated experiment; a new experiment requires a new model directory and an explicit evaluation plan. It downloads official Keras ImageNet weights on first run. Local files under `.runtime/street_vision` include JSON coefficients/scaler/calibrator, frozen HDF5 weights, hashed embedding cache, per-image SHA256 records, test predictions, and the complete manifest. No pickle deserialization is used.

Dataset SHA256: `bbf4a8c97b3afbc0ea59289e55892bd205fdf451ee192eb77a0a5f635fd21385`. Label CSV SHA256: `ed330d47b16deefd5a9b9e85fa6b3f8ad8d853a1214c9c2f6d14dd0e4dc1d12a`. Embedding version: `13fe8d9793e4fa496f74e4d689a9235455d2f890c671ac60ae6e1930f758a625`. Parameters SHA256: `92efc8827f43e940397b5b945f2e4d8b6a95f15f33d75c5eb86e230438990eaa`. Frozen weights SHA256: `bd42160c365bcd5edbc2ad3aa9612bb2c074575e73ba05b309336b8caaf5bb4b`.

Preprocessing shared by training and inference: `PIL RGB; bilinear resize 224x224; float32 0..255; MobileNetV3 built-in rescaling`. The exact same frozen embedding is available for separately teaching decisions, but anomaly labels never become policy labels automatically.

For a fresh clone that only needs the image encoder for decision teaching, run `python scripts/setup_street_encoder.py`. It downloads the official Keras ImageNet backbone without requiring this dataset or training an anomaly head. Existing encoder weights and feature versions are verified and preserved. Object detection has its own `scripts/setup_pretrained_detector.py` setup.
