# Supervised visual anomaly fine-tuning

**Rejected: separate recording validation gate failed; not reliable anomaly recognition.** Model `24954ab96fc2254e-finetuned320`. This experiment actually changed pretrained visual convolution weights; it did not change the original action-teaching encoder or the pretrained vehicle detector.

## Training design

The predeclared plan used a separate 320×320 MobileNetV3Small, global mean and maximum pooling, a regularized 64-unit head and dropout. The early image stem was frozen and cached. Three head warm-up epochs preceded five low-learning-rate epochs updating the final two inverted residual blocks and final convolution. Batch-normalization weights and statistics remained frozen. Two views per gradient-fit image were used: original pixels and deterministic mild brightness/contrast adjustments; no geometric transformations or invented labels.

Of 2,550 training-set images, **2082 images from 27 recordings** supplied gradients; **468 from 8 different recordings** selected the best fine-tuning checkpoint. The separate validation set supplied **106 calibration images from four recordings** and **119 threshold-selection images from four other recordings**. All 3,000 image hashes were verified against the original feature provenance. Best checkpoint training-development AP: 0.7637.

The operating threshold 0.25823751 maximized balanced accuracy under a 20% false-alert ceiling using only the threshold-validation subset. Activation required recall >=40%, precision >=30%, false-alert rate <=20%, and positive calibration slope. Threshold-validation result: precision 50.00%, recall 4.76%, false alerts 1/98. Gate accepted: **False**.

Calibration slope: -0.09308911. The raw anomaly scores ranked the calibration labels in the wrong direction, which independently failed the predeclared acceptance rule. Labels came from the publisher's frame annotations, not predictions created by this model. They have not received a new human audit in this project.

## Retrospective test

Weights, calibration, threshold and development acceptance were saved before the 225-image test was scored. This test had been viewed in previous experiments, so the result is retrospective and cannot establish fresh reliability.

| Metric | Result |
|---|---:|
| Anomaly precision | 37.50% |
| Anomaly recall | 13.33% |
| F1 | 0.1967 |
| Balanced accuracy | 53.89% |
| Average precision | 0.2810 |
| Overall accuracy | 78.22% |

Detected 6 of 45 anomalies, missed 39, and raised 10 false alerts among 180 normal frames. Confusion matrix (true normal/anomaly rows; predicted normal/review columns): `[[170, 10], [39, 6]]`.

## Limits

Publisher UIT-ADrone labels include roundabout and other aerial scenes and have not been independently adjudicated here. No verified anomaly labels were supplied for the new FLUID signalized-intersection footage. A single image cannot reliably prove jaywalking, right-of-way, waiting time, wrong-way travel, or crossing a red light: those usually require tracked motion, mapped lanes and contemporaneous signal state. This appearance classifier is not a traffic violation detector or signal policy. Passing a small development gate would support research review only, not a 90% reliability claim or real-road deployment.

Original 224px action embeddings and active detector weights remain unchanged. The separate model can be loaded through `FineTunedAnomaly`; activation is a separate explicit integration choice. Run time: 11.9 minutes.

All four focused tests passed: deterministic training augmentation, failed-weight-load retry protection, recording separation and unchanged pretest artifact hashes, and fresh model reconstruction/loading that reproduced the saved probabilities for two fixed test frames within 0.0001. The completed run also verified in-memory pixel inference against its cached-stem path. No further fitting followed the test results.

Reproduce with `python scripts/finetune_street_anomaly.py --model-dir .runtime/street_anomaly_finetuned_new`. The official [TensorFlow transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning) supports training a new head before low-rate fine-tuning and retaining frozen batch-normalization behavior.
