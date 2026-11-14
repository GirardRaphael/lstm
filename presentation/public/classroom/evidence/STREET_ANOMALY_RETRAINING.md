# Real aerial anomaly retraining

Created 2026-11-14T17:15:17.456906+00:00. Model `656f973ad495ec31-review`.

**Result: inadequate: development review gate failed; do not activate as a reliable anomaly recognizer.** This run actually trained a new binary anomaly classifier. It does not train the object detector, create signalized-intersection anomaly ground truth, or learn safe signal decisions.

The pretrained MobileNetV3Small encoder stayed frozen and unchanged (224×224 average-pooled features). The script checked all 3,000 actual image hashes, labels, cached features, and encoder identity before training. The final classifier fitted **2,550 real publisher-labelled frames from 35 recordings**: 1,412 normal and 1,138 anomalous. Validation and test each retain 225 frames from eight disjoint recordings.

## Predeclared selection

The experiment plan was saved before fitting in `.runtime/street_vision_review/experiment_plan.json`. Six balanced classifiers competed only on three training-recording folds. Standardization was fitted inside each training fold. Mean average precision selected `logistic`, C=0.001, with mean training-CV AP 0.7386. This CV result is a development selection score, not external validation.

The 225 validation frames were divided by recording into **106 calibration frames** (24 anomalous) and **119 threshold-selection frames** (21 anomalous). The sigmoid only fitted the first subset. The threshold only used the second subset, maximizing balanced accuracy with a 20% normal-frame false-alert ceiling. Its value is 0.23795149; this is a review operating point, not a reliability percentage.

Development activation required recall >=40%, precision >=30%, false-alert rate <=20%, and a positive sigmoid slope. Accepted: **False**. Calibration slope: 0.031609. Threshold-subset precision 20.0%, recall 4.8%, false alerts 4/98.

## Retrospective test result

The test set was previously viewed in older experiments. It was not used for this run's fitting, model choice, calibration, threshold choice, or activation gate. Parameters and the development decision were saved before this test evaluation. No choice was changed in response to its outcome.

| Metric | New candidate | Previous expanded classifier |
|---|---:|---:|
| Anomaly precision | 18.75% | 0.00% |
| Anomaly recall | 13.33% | 0.00% |
| F1 | 0.1558 | 0.0000 |
| Balanced accuracy | 49.44% | 50.00% |
| Average precision | 0.2077 | 0.3752 |
| Overall accuracy | 71.11% | 80.00% |

Confusion matrix rows=true normal/anomaly, columns=predicted normal/review: `[[154, 26], [39, 6]]`. The model found 6 of 45 anomalies, missed 39, and raised 26 false alerts among 180 normal frames. Always predicting normal gives 80% accuracy but 0% anomaly recall; therefore accuracy alone is not success.

## Limits and next evidence

- Training labels are publisher binary UIT-ADrone frame masks, not independently adjudicated anomalies or verified violation subtypes.
- UIT-ADrone includes roundabout and other aerial scenes; these outcomes do not validate FLUID traffic-light footage or signal decisions.
- The 225-frame test set was already exposed by earlier experiments. These results are retrospective, not an untouched confirmation.
- Only 45 positive test frames and eight test recordings; nearby frames are correlated and apparent frame-level rates are uncertain.
- Calibration and operating-threshold subsets each contain only four validation recordings. The development gate is not independent reliability evidence.
- Frozen 224px image features cannot reliably establish temporal events, right-of-way, or red-light violations from a single frame.
- No pretrained detector weights or traffic-light decision policy were trained in this experiment.

No 90% reliability or real-road safety claim is supported. Obtain independently reviewed anomaly sequences from multiple actual signalized intersections and evaluate on new recording groups before such claims.

Reproduce with `python scripts/train_street_anomaly_review.py --model-dir .runtime/street_vision_review_new` after installing the original expanded dataset and frozen feature artifacts. Artifacts use safe JSON/NumPy formats, without pickle execution.
