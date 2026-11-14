"""Train a separate 320px anomaly network with a frozen stem and learned tail.

Selection uses only training recording groups. Probability calibration and
operating-threshold validation use disjoint validation recordings. The historical
test is evaluated once after all choices are recorded and is retrospective.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
from finetuned_anomaly import INPUT_SIZE, STEM_LAYER, build_models, set_backbone_training
from street_vision import file_sha256, prepare_image
from train_street_anomaly_review import choose_review_threshold
from train_street_vision import metrics, write_json

PLAN = {
    "seed": 20261115, "input_size": [320, 320], "backbone": "local ImageNet MobileNetV3Small",
    "frozen_stem_through": STEM_LAYER, "fine_tuned_layers": "last two inverted residual blocks plus final 1x1 convolution; all BatchNormalization layers remain frozen",
    "pooling": "global average and global maximum concatenated",
    "head": "LayerNormalization; Dense64 relu with L2=.001; Dropout=.3; Dense1 logit",
    "augmentation": "Training fit groups only: original view plus one deterministic brightness/contrast view, factors independently uniform [.9,1.1], clipped to [0,255]; no geometric transforms or label changes",
    "fit_selection": "First five-fold StratifiedGroupKFold train split: four folds for gradients, one recording-disjoint fold for checkpoint selection",
    "warmup_epochs": 3, "warmup_learning_rate": .001, "fine_tune_epochs": 5, "fine_tune_learning_rate": .00001,
    "batch_size": 32, "class_weight": "inverse class frequency within gradient-fit groups",
    "checkpoint_selection": "Highest average precision on training-held-out recordings across the five fine-tuning epochs; first epoch wins ties",
    "calibration": "First half of validation recording groups only, same group split as earlier review experiment (seed 20261114); sigmoid logistic C=1",
    "threshold": "Other half of validation groups only: maximize balanced accuracy subject to false-positive rate <=.20; ties favor recall then higher threshold",
    "review_gate": "Threshold-validation recall >=.40, precision >=.30, FPR <=.20, positive calibration slope",
    "test": "Previously exposed test; score once after checkpoint/calibration/threshold/gate are frozen. Never use test to select weights or operating point.",
    "action_encoder": "Never changed. New weights live in a separate anomaly directory and inference class.",
}


def augmented_pixels(pixels: np.ndarray, sample_id: str) -> np.ndarray:
    seed = int(hashlib.sha256((str(PLAN["seed"]) + sample_id).encode()).hexdigest()[:16], 16)
    random = np.random.default_rng(seed)
    brightness, contrast = random.uniform(.9, 1.1, 2)
    center = pixels.mean(axis=(0, 1), keepdims=True)
    return np.clip((pixels - center) * contrast + center, 0, 255) * np.float32(brightness)


def predict_batches(model, features, indices, batch=64):
    return np.concatenate([np.asarray(model(np.asarray(features[indices[start:start + batch]]), training=False)).reshape(-1) for start in range(0, len(indices), batch)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/intersection_anomalies_expanded")
    parser.add_argument("--source-model-dir", type=Path, default=ROOT / ".runtime/street_vision_expanded")
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".runtime/street_anomaly_finetuned")
    args = parser.parse_args()
    data, source, target = args.data_dir.resolve(), args.source_model_dir.resolve(), args.model_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if (target / "experiment_plan.json").exists():
        raise SystemExit("An experiment already exists here. Use a new model directory; evaluated runs are immutable.")
    started = time.time()
    write_json(target / "experiment_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(), **PLAN})
    original = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    hashes = json.loads((source / "input_hashes.json").read_text(encoding="utf8"))
    if file_sha256(data / "labels.csv") != original["labels_sha256"] or file_sha256(source / "input_hashes.json") != original["artifact_sha256"]["input_hashes.json"]:
        raise ValueError("Labels or input hash manifest changed")
    if file_sha256(source / "backbone.weights.h5") != original["artifact_sha256"]["backbone.weights.h5"]:
        raise ValueError("Original pretrained weights changed")
    with (data / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in rows]
    hash_by_id = {entry["sample_id"]: entry["sha256"] for entry in hashes["images"]}
    if len(ids) != len(set(ids)) or set(ids) != set(hash_by_id) or len(ids) != len(hashes["images"]):
        raise ValueError("Label IDs and image hashes must correspond exactly")
    paths = [(data / row["image_path"]).resolve() for row in rows]
    for row, path in zip(rows, paths):
        if not path.is_relative_to(data) or file_sha256(path) != hash_by_id[row["sample_id"]]:
            raise ValueError("Actual image bytes changed")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    groups = np.asarray([row["source_video"] for row in rows])
    splits = {name: np.asarray([index for index, row in enumerate(rows) if row["split"] == name]) for name in ["train", "validation", "test"]}
    if sum(len(x) for x in splits.values()) != len(rows) or set(labels) != {0, 1}:
        raise ValueError("All rows require a declared split and binary label")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(groups[splits[a]]) & set(groups[splits[b]]):
            raise ValueError("Source recording leakage")
    train = splits["train"]
    fit_rel, dev_rel = next(StratifiedGroupKFold(5, shuffle=True, random_state=PLAN["seed"]).split(np.zeros(len(train)), labels[train], groups[train]))
    fit, dev = train[fit_rel], train[dev_rel]
    validation = splits["validation"]
    cal_rel, gate_rel = next(StratifiedGroupKFold(2, shuffle=True, random_state=20261114).split(np.zeros(len(validation)), labels[validation], groups[validation]))
    calibration, gate = validation[cal_rel], validation[gate_rel]
    partitions = {"gradient_fit": fit, "checkpoint_selection": dev, "calibration": calibration, "threshold_validation": gate, "test": splits["test"]}
    split_detail = {name: {"frames": len(index), "anomalies": int(labels[index].sum()), "recordings": sorted(set(groups[index]))} for name, index in partitions.items()}
    write_json(target / "partitions.json", split_detail)
    print("Verified 3,000 real images; partitions: " + json.dumps({name: {**value, "recordings": len(value["recordings"])} for name, value in split_detail.items()}), flush=True)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(PLAN["seed"])
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    backbone, stem, cached, full = build_models(tf, source / "backbone.weights.h5")
    set_backbone_training(tf, backbone, fine_tune=False)
    stem_shape = tuple(stem.output_shape[1:])
    feature_path = target / "stem_features.npy"
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(len(rows) + len(fit), *stem_shape))
    for start in range(0, len(rows) + len(fit), 16):
        end = min(start + 16, len(rows) + len(fit))
        batch = []
        for index in range(start, end):
            source_index = index if index < len(rows) else int(fit[index - len(rows)])
            with Image.open(paths[source_index]) as image:
                pixels = prepare_image(image, INPUT_SIZE)
            if index >= len(rows):
                pixels = np.clip(augmented_pixels(pixels, ids[source_index]), 0, 255)
            batch.append(pixels)
        features[start:end] = np.asarray(stem(np.stack(batch), training=False))
        if start % 160 == 0 or end == len(features):
            print(f"Frozen stem {end}/{len(features)} views; elapsed {time.time() - started:.1f}s", flush=True)
    features.flush()
    fit_indices = np.r_[fit, np.arange(len(rows), len(features))]
    fit_labels = np.r_[labels[fit], labels[fit]].astype(np.float32)
    counts = np.bincount(labels[fit], minlength=2)
    class_weights = len(fit) / (2 * counts)
    sample_weights = class_weights[fit_labels.astype(int)].astype(np.float32)
    random = np.random.default_rng(PLAN["seed"])
    history = []
    best_ap = -1.
    initial_tail_hash = hashlib.sha256(b"".join(np.asarray(w).tobytes() for layer in backbone.layers if layer.name.startswith("expanded_conv_9") for w in layer.weights)).hexdigest()
    for stage, epochs, learning_rate in [("warmup", 3, .001), ("fine_tune", 5, .00001)]:
        set_backbone_training(tf, backbone, fine_tune=stage == "fine_tune")
        cached.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        for epoch in range(epochs):
            order = random.permutation(len(fit_indices))
            losses = []
            epoch_started = time.time()
            for start in range(0, len(order), PLAN["batch_size"]):
                batch = order[start:start + PLAN["batch_size"]]
                loss = cached.train_on_batch(np.asarray(features[fit_indices[batch]]), fit_labels[batch, None], sample_weight=sample_weights[batch])
                losses.append(float(loss))
            dev_scores = predict_batches(cached, features, dev)
            ap = float(average_precision_score(labels[dev], dev_scores))
            entry = {"stage": stage, "epoch": epoch + 1, "loss": float(np.mean(losses)), "checkpoint_group_average_precision": ap, "seconds": time.time() - epoch_started}
            history.append(entry)
            if stage == "fine_tune" and ap > best_ap:
                best_ap = ap
                full.save_weights(str(target / "anomaly.weights.h5"))
                write_json(target / "checkpoint.json", entry)
            write_json(target / "training_history.json", history)
            print(json.dumps(entry), flush=True)
    full.load_weights(str(target / "anomaly.weights.h5"))
    final_tail_hash = hashlib.sha256(b"".join(np.asarray(w).tobytes() for layer in backbone.layers if layer.name.startswith("expanded_conv_9") for w in layer.weights)).hexdigest()
    if final_tail_hash == initial_tail_hash:
        raise RuntimeError("Fine-tuning did not update visual convolution weights")
    cal_scores = predict_batches(cached, features, calibration)
    calibrator = LogisticRegression(C=1., max_iter=3000, random_state=PLAN["seed"]).fit(cal_scores[:, None], labels[calibration])
    gate_scores = predict_batches(cached, features, gate)
    # Verify the deployable pixel path before publishing any activatable manifest.
    with Image.open(paths[int(gate[0])]) as image:
        pixels = prepare_image(image, INPUT_SIZE)[None]
    np.testing.assert_allclose(np.asarray(full(pixels, training=False)).reshape(-1), gate_scores[:1], rtol=1e-4, atol=1e-4)
    _, _, _, reloaded = build_models(tf)
    reloaded.load_weights(str(target / "anomaly.weights.h5"))
    np.testing.assert_allclose(np.asarray(reloaded(pixels, training=False)).reshape(-1), gate_scores[:1], rtol=1e-4, atol=1e-4)
    gate_probabilities = calibrator.predict_proba(gate_scores[:, None])[:, 1]
    threshold = choose_review_threshold(labels[gate], gate_probabilities)
    operating = metrics(labels[gate], gate_probabilities, threshold)
    tn, fp = operating["confusion_matrix"][0]
    accepted = bool(operating["recall"] >= .4 and operating["precision"] >= .3 and fp / (tn + fp) <= .2 and calibrator.coef_[0, 0] > 0)
    parameters = {"coef": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]), "threshold": threshold}
    write_json(target / "calibration.json", parameters)
    decision = {"checkpoint": json.loads((target / "checkpoint.json").read_text()), "development_metrics": operating, "review_candidate_accepted": accepted,
                "weights_sha256_before_test": file_sha256(target / "anomaly.weights.h5"), "calibration_sha256_before_test": file_sha256(target / "calibration.json"),
                "visual_weights_changed": True, "action_encoder_unchanged": file_sha256(source / "backbone.weights.h5") == original["artifact_sha256"]["backbone.weights.h5"]}
    write_json(target / "development_decision.json", decision)
    # Only now are the test scores calculated. No test outcome can change this run.
    test_scores = predict_batches(cached, features, splits["test"])
    test_probabilities = calibrator.predict_proba(test_scores[:, None])[:, 1]
    test_metrics = metrics(labels[splits["test"]], test_probabilities, threshold)
    manifest = {"model_version": file_sha256(target / "anomaly.weights.h5")[:16] + "-finetuned320", "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "Separate 320px MobileNetV3Small with supervised last-two-block fine-tuning, mean/max pooling and regularized dense head",
        "review_candidate_accepted": accepted, "validation_result": "Development review gate passed; research candidate, not independent reliability validation" if accepted else "Rejected: separate recording validation gate failed; not reliable anomaly recognition",
        "test_metrics": test_metrics, "development_metrics": operating, "partitions": split_detail,
        "dataset_sha256": original["dataset_sha256"], "labels_sha256": original["labels_sha256"], "source_encoder_sha256": original["artifact_sha256"]["backbone.weights.h5"],
        "plan": PLAN, "development_decision": decision, "evaluation_reused": True, "test_used_for_training_or_selection": False,
        "deployment_ready": False, "signal_control_validated": False, "separate_from_action_encoder": True,
        "artifact_sha256": {name: file_sha256(target / name) for name in ["anomaly.weights.h5", "calibration.json"]},
        "elapsed_seconds": time.time() - started, "versions": {"tensorflow": tf.__version__, "numpy": np.__version__}}
    write_json(target / "manifest.json", manifest)
    write_json(target / "test_predictions.json", [{"sample_id": ids[i], "source_video": str(groups[i]), "label": int(labels[i]), "anomaly_probability": float(p)} for i, p in zip(splits["test"], test_probabilities)])
    summary = {"schema_version": 1, "training_completed": True, "model_version": manifest["model_version"], "accepted": accepted,
        "active_model_changed": False, "outcome": manifest["validation_result"], "partitions": split_detail,
        "development_metrics": operating, "test_metrics": test_metrics, "evaluation_reused": True,
        "separate_from_action_encoder": True, "visual_weights_changed": True, "elapsed_seconds": manifest["elapsed_seconds"]}
    write_json(ROOT / "reports/street_anomaly_finetuning.json", summary)
    report = f"""# Supervised visual anomaly fine-tuning

**{manifest['validation_result']}.** Model `{manifest['model_version']}`. This experiment actually changed pretrained visual convolution weights; it did not change the original action-teaching encoder or the pretrained vehicle detector.

## Training design

The predeclared plan used a separate 320×320 MobileNetV3Small, global mean and maximum pooling, a regularized 64-unit head and dropout. The early image stem was frozen and cached. Three head warm-up epochs preceded five low-learning-rate epochs updating the final two inverted residual blocks and final convolution. Batch-normalization weights and statistics remained frozen. Two views per gradient-fit image were used: original pixels and deterministic mild brightness/contrast adjustments; no geometric transformations or invented labels.

Of 2,550 training-set images, **{len(fit)} images from {len(set(groups[fit]))} recordings** supplied gradients; **{len(dev)} from {len(set(groups[dev]))} different recordings** selected the best fine-tuning checkpoint. The separate validation set supplied **{len(calibration)} calibration images from four recordings** and **{len(gate)} threshold-selection images from four other recordings**. All 3,000 image hashes were verified against the original feature provenance. Best checkpoint training-development AP: {best_ap:.4f}.

The operating threshold {threshold:.8f} maximized balanced accuracy under a 20% false-alert ceiling using only the threshold-validation subset. Activation required recall >=40%, precision >=30%, false-alert rate <=20%, and positive calibration slope. Threshold-validation result: precision {operating['precision']:.2%}, recall {operating['recall']:.2%}, false alerts {fp}/{tn + fp}. Gate accepted: **{accepted}**.

Calibration slope: {parameters['coef']:.8f}. A negative slope means the raw anomaly scores ranked the calibration labels in the wrong direction, and independently fails the predeclared acceptance rule. Labels came from the publisher's frame annotations, not predictions created by this model. They have not received a new human audit in this project.

## Retrospective test

Weights, calibration, threshold and development acceptance were saved before the 225-image test was scored. This test had been viewed in previous experiments, so the result is retrospective and cannot establish fresh reliability.

| Metric | Result |
|---|---:|
| Anomaly precision | {test_metrics['precision']:.2%} |
| Anomaly recall | {test_metrics['recall']:.2%} |
| F1 | {test_metrics['f1']:.4f} |
| Balanced accuracy | {test_metrics['balanced_accuracy']:.2%} |
| Average precision | {test_metrics['average_precision']:.4f} |
| Overall accuracy | {test_metrics['accuracy']:.2%} |

Detected {test_metrics['confusion_matrix'][1][1]} of 45 anomalies, missed {test_metrics['confusion_matrix'][1][0]}, and raised {test_metrics['confusion_matrix'][0][1]} false alerts among 180 normal frames. Confusion matrix (true normal/anomaly rows; predicted normal/review columns): `{test_metrics['confusion_matrix']}`.

## Limits

Publisher UIT-ADrone labels include roundabout and other aerial scenes and have not been independently adjudicated here. No verified anomaly labels were supplied for the new FLUID signalized-intersection footage. A single image cannot reliably prove jaywalking, right-of-way, waiting time, wrong-way travel, or crossing a red light: those usually require tracked motion, mapped lanes and contemporaneous signal state. This appearance classifier is not a traffic violation detector or signal policy. Passing a small development gate would support research review only, not a 90% reliability claim or real-road deployment.

Original 224px action embeddings and active detector weights remain unchanged. The separate model can be loaded through `FineTunedAnomaly`; activation is a separate explicit integration choice. Run time: {manifest['elapsed_seconds'] / 60:.1f} minutes.

Reproduce with `python scripts/finetune_street_anomaly.py --model-dir .runtime/street_anomaly_finetuned_new`. The official [TensorFlow transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning) supports training a new head before low-rate fine-tuning and retaining frozen batch-normalization behavior.
"""
    (ROOT / "reports/STREET_ANOMALY_FINETUNING.md").write_text(report, encoding="utf8")
    del features
    feature_path.unlink()  # Verified task-owned cache; inference uses the saved complete weights.
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
