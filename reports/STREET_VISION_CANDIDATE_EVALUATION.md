# Spatial-feature follow-up on real aerial frames

Model `40569530247e6305-297073d3efb1`, generated 2026-11-14T16:06:52.324813+00:00.

**This is a retrospective follow-up.** The same 225-frame test set was previously scored by the [initial baseline](STREET_VISION_EVALUATION.md). Candidate selection and threshold selection use only training groups, but the changed experimental direction followed seeing baseline failure. A new untouched dataset is required for confirmation. This model is not approved for signal control.

**Outcome: inadequate for trusted anomaly alarms.** It detected 42 of 45 anomalies but incorrectly flagged 161 of 180 normal frames. Precision was only 20.7%, and balanced accuracy 51.9%. High recall here comes with an overwhelming false-alert burden, not reliable recognition.

## Predeclared plan and training

The frozen ImageNet MobileNetV3Small processes 448x448 RGB images. Spatial mean, maximum, and standard deviation are concatenated into 1728 features to retain stronger local activations than whole-image average pooling. No backbone fine-tuning occurred. Standardization is fitted separately within each training fold; no validation/test image trains the scaler or classifier.

The 1050 training images from 35 source recordings were divided by recording into three folds. Logistic C candidates were fixed before scoring: 0.001, 0.01, 0.1. Mean average precision selects C; ties retain the earlier/smaller C. Full training data refits the chosen model. Sigmoid calibration uses the separate 225-image/8-recording validation split. Review threshold selection uses calibrated training-group out-of-fold predictions to maximize specificity while retaining at least 80% anomaly recall. The threshold is therefore a review operating point, not 50% belief or a traffic-light safety threshold.

| C | Mean training-group CV AP | Fold AP values |
|---|---:|---|
| 0.001 | 0.2590 | [0.3793218248106845, 0.20345410674994419, 0.1943651569353365] |
| 0.01 | 0.2372 | [0.3206496900331289, 0.19472192634818242, 0.1961932724673035] |
| 0.1 | 0.2304 | [0.2674428503382041, 0.20284980011307965, 0.22081252092209605] |

Selected C: 0.001. Review threshold: 0.17880562. Threshold-selection recall: 0.8000; specificity: 0.2476. These development values are not independent evaluation. Candidate selection, probability calibration, and threshold were fixed before this experiment scored the reused test images.

## Retrospective evaluation

The 225 test images contain 180 normal and 45 anomalous frames across eight recordings. Source recordings remain disjoint from training and calibration, and all image/label hashes match the first experiment.

| Metric | Follow-up model | Always normal p=0 | Constant training prevalence p=0.2 |
|---|---:|---:|---:|
| accuracy | 0.2711 | 0.8000 | 0.8000 |
| precision | 0.2069 | 0.0000 | 0.0000 |
| recall | 0.9333 | 0.0000 | 0.0000 |
| f1 | 0.3387 | 0.0000 | 0.0000 |
| balanced_accuracy | 0.5194 | 0.5000 | 0.5000 |
| average_precision | 0.3066 | 0.2000 | 0.2000 |
| roc_auc | 0.5701 | 0.5000 | 0.5000 |
| brier | 0.1588 | 0.2000 | 0.1600 |
| ece_10_bins | 0.0280 | 0.2000 | 0.0000 |

Confusion matrix (rows true normal/anomaly; columns predicted normal/review): `[[19, 161], [3, 42]]`. This model asks for review on 203 of 225 frames, including 161 false alerts and 42 detected anomalies. It misses 3 anomalies. AP and ROC AUC assess ranking; Brier/ECE assess probabilities; thresholded recall alone does not establish a useful model.

## Reliability bins

| Probability interval | Frames | Mean probability | Observed anomaly fraction |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 121 | 0.1855 | 0.1570 |
| 0.2-0.3 | 104 | 0.2226 | 0.2500 |
| 0.3-0.4 | 0 | - | - |
| 0.4-0.5 | 0 | - | - |
| 0.5-0.6 | 0 | - | - |
| 0.6-0.7 | 0 | - | - |
| 0.7-0.8 | 0 | - | - |
| 0.8-0.9 | 0 | - | - |
| 0.9-1.0 | 0 | - | - |

## Limitations and artifacts

- Publisher binary labels have not been independently adjudicated; anomalies have no verified subtype or boxes.
- Only 45 positive test frames from a small number of held-out recordings; nearby frames are correlated.
- Frozen 448px ImageNet features can miss small actors, motion-dependent violations, signal state, and other street details.
- Calibration is fitted on 225 validation frames and may fail under new cameras, geography, illumination, or synthetic graphics.
- This is a binary image classifier, not a vehicle detector, a learned light policy, or evidence that a signal action is safe.
- Test images were already scored by the first baseline; this follow-up is retrospective, not an untouched confirmation.
- The review threshold targets recall using training-group out-of-fold scores. New recordings can have a different recall/false-alert tradeoff.

Artifacts: `.runtime/street_vision_candidate`. Reproduce with `python scripts/train_street_vision_candidate.py --data-dir <intersection_anomalies>`. The original baseline artifacts must exist and are retained. The script refuses to overwrite an evaluated candidate. `experiment_plan.json` is written before feature extraction; `selection.json` records training-only selection and threshold. No pickle is loaded.

Dataset hash: `bbf4a8c97b3afbc0ea59289e55892bd205fdf451ee192eb77a0a5f635fd21385`. Embedding version: `297073d3efb138e69960eb02105f81872821c6650dce80bc53234523d652a316`. Parameter hash: `40569530247e6305f334beb73b3d5d943fdeea27c5dd0e946045877048b5a61e`.
