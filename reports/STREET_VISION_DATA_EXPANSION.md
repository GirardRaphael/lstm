# Additional real aerial training frames

The existing UIT-ADrone index contains 1,500 additional eligible frames from the same 35 training recordings. They are distinct from the original 1,500 images and maintain the original minimum 60-source-frame spacing (nominally two seconds) across both classes. Frames near publisher-label transitions remain excluded by the original selection rules.

| Dataset portion | Original | Added | Expanded |
|---|---:|---:|---:|
| Training normal | 840 | 572 | 1,412 |
| Training anomalous | 210 | 928 | 1,138 |
| Validation normal/anomalous | 180/45 | 0 | 180/45 |
| Test normal/anomalous | 180/45 | 0 | 180/45 |
| Total images | 1,500 | 1,500 | 3,000 |

The expanded training set is 55.37% normal /44.63% anomalous. Adding all available spaced examples changes training balance; validation and test remain 80/20. Retaining 80/20 for the added examples would permit only 715 additions (572 normal /143 anomalous), so the full 1,500 additions were retained instead. Class weights and validation-only probability calibration account for the new training prevalence; no synthetic duplication or re-labeling was used.

All original examples and all 450 held-out rows remain unchanged. No held-out source recording contributes a new training image. The original dataset folder is preserved; the expanded copy is `data/intersection_anomalies_expanded`. Original image bytes are linked locally where the filesystem permits, while new images are downloaded from their recorded public Google Drive IDs. Binary labels remain publisher frame-mask labels, without independently verified boxes, violation subtypes, or signal-action labels.

More frames from the same 35 recordings do not add new environments or establish 90% reliability. This is a training-size experiment; the repeatedly inspected 225-frame test set is retrospective. Independent new recordings and action-level labels remain necessary for confirming signal-decision quality.

Reproduce selection with `python scripts/expand_street_vision_data.py --source <original-folder> --target data/intersection_anomalies_expanded`; append `--download` to retrieve the selected public images, then `--verify` to check all images and unchanged held-out rows. All images and local artifacts are ignored by Git.

The fixed baseline architecture and training settings are retained: 224px frozen ImageNet MobileNetV3Small features, standardized training features, balanced L2 logistic regression C=0.1, validation-only sigmoid calibration, fixed 0.5 review threshold. The original 1,500 image embeddings are reused only after image and encoder checksum verification. Retraining completed; results appear in [the expanded evaluation](STREET_VISION_EXPANDED_EVALUATION.md). All 3,000 images passed image decoding and file-size verification, and all 3,000 SHA256 hashes are unique. The fixed-threshold model still missed all 45 test anomalies; additional frames alone did not achieve 90% reliability.
