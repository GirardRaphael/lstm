# Pretrained detector: measured real-image benchmark

Model: EfficientDet-Lite2, official Google float32 version1. SHA256 `ad2abbf2b4e10585e15176fd7b5ef03c28dda959ae26fc142549fdd1814db91d`. Inference contract `efficientdet-lite2-fp32-v1-ad2abbf2b4e1-69874f8e0a11`.

Fixed 500-image vehicle-containing COCO val2017 subset; general real-image recognition, not aerial intersection or signal-decision validation.

Score threshold 0.3; class-matched IoU threshold 0.5. Selection seed 20261114; sample fixed before inference.

**90% target met: False.** Precision 80.98%; recall 48.18%; F1 60.42%.

| Class | Ground truth | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| person | 1742 | 85.91% | 51.44% | 64.34% |
| bicycle | 101 | 64.41% | 37.62% | 47.50% |
| car | 1195 | 79.12% | 46.61% | 58.66% |
| motorcycle | 239 | 84.78% | 48.95% | 62.07% |
| bus | 180 | 85.43% | 71.67% | 77.95% |
| truck | 233 | 67.36% | 41.63% | 51.46% |
| traffic light | 295 | 65.41% | 29.49% | 40.65% |
| stop sign | 27 | 80.00% | 44.44% | 57.14% |

Observed average local inference: 280.0 ms/image. This is not an end-to-end video frame rate.

At least 90% precision AND recall for every displayed road class, with >=20 ground-truth instances/class and >=500 images.

## Interpretation

- Publisher may have used COCO validation while developing the pretrained model; this is a public benchmark, not a fresh private test.
- Vehicle-containing images were sampled; this does not measure false alarms on all-negative scenes.
- COCO images are not a recording-disjoint aerial intersection dataset.
- Crowd annotations are excluded from recall; predictions overlapping a same-class crowd at >=0.5 intersection/predicted area are ignored.
- Fixed-threshold precision/recall is not COCO mAP. No detector weights or thresholds were fitted using these evaluation images.

The pretrained detector is used without local weight fine-tuning. Additional aerial examples train the separate anomaly head; reviewed human actions train a separate light-choice model. Those are three distinct learning tasks.

## Reproduction

`python scripts/evaluate_pretrained_detector.py prepare --annotations <official annotation ZIP>`
`python scripts/evaluate_pretrained_detector.py evaluate`

Sources: [Google model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector), [COCO](https://cocodataset.org/#download).
