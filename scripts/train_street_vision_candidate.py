"""Predeclared spatial-feature follow-up; reused test is retrospective only."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
from street_vision import file_sha256, pool_features, prepare_image, score_embeddings, sigmoid
from train_street_vision import metrics, write_json

PLAN = {
    "seed": 42, "backbone": "MobileNetV3Small", "pretrained": "ImageNet", "backbone_frozen": True,
    "input_size": [448, 448], "pooling": "spatial_mean_max_std", "features": 1728,
    "classifier_C_candidates": [0.001, 0.01, 0.1], "classifier_class_weight": "balanced",
    "selection": "mean average precision over three source-recording-group folds within training only",
    "calibration": "validation-only sigmoid logistic regression C=1.0",
    "threshold_selection": "highest specificity satisfying >=80% recall on calibrated training-group out-of-fold scores",
    "evaluation_reused": True,
    "preprocessing": "PIL RGB; bilinear resize 448x448; float32 0..255; MobileNetV3 built-in rescaling; concatenate spatial mean/max/std",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / ".runtime" / "street_vision")
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".runtime" / "street_vision_candidate")
    args = parser.parse_args()
    directory, baseline, data = args.model_dir.resolve(), args.baseline_dir.resolve(), args.data_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise SystemExit("This candidate has already been evaluated; refusing to rerun its test evaluation.")
    write_json(directory / "experiment_plan.json", PLAN)
    original = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    hash_records = json.loads((baseline / "input_hashes.json").read_text(encoding="utf-8"))
    if file_sha256(data / "labels.csv") != original["labels_sha256"]:
        raise ValueError("Dataset labels changed since the baseline")
    with (data / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = sorted(list(csv.DictReader(handle)), key=lambda row: row["sample_id"])
    hashes = {record["sample_id"]: record["sha256"] for record in hash_records["images"]}
    for row in rows:
        path = (data / row["image_path"]).resolve()
        if not path.is_relative_to(data) or file_sha256(path) != hashes[row["sample_id"]]:
            raise ValueError("Dataset image changed since baseline")
    write_json(directory / "input_hashes.json", hash_records)
    weights = directory / "backbone.weights.h5"
    if not weights.exists():
        shutil.copy2(baseline / "backbone.weights.h5", weights)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    backbone = tf.keras.applications.MobileNetV3Small(input_shape=(448, 448, 3), include_top=False,
        weights=None, pooling=None, include_preprocessing=True)
    backbone.load_weights(str(weights))
    backbone.trainable = False
    weights_hash = file_sha256(weights)
    embedding_version = hashlib.sha256(json.dumps({"architecture": "MobileNetV3Small", "weights_sha256": weights_hash,
        "preprocessing": PLAN["preprocessing"], "pooling": PLAN["pooling"]}, sort_keys=True).encode()).hexdigest()
    write_json(directory / "backbone.json", {"embedding_version": embedding_version, "weights_sha256": weights_hash,
        "embedding_dimensions": 1728, "input_size": [448, 448], "pooling": PLAN["pooling"], "preprocessing": PLAN["preprocessing"]})
    ids = np.asarray([row["sample_id"] for row in rows])
    cache_path = directory / "embeddings.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["dataset_sha256"].item()) != original["dataset_sha256"] or str(cache["embedding_version"].item()) != embedding_version or not np.array_equal(ids, cache["sample_ids"]):
                raise ValueError("Invalid feature cache")
            features = cache["embeddings"]
    else:
        batches = []
        for start in range(0, len(rows), 16):
            pixels = []
            for row in rows[start:start + 16]:
                with Image.open(data / row["image_path"]) as image:
                    pixels.append(prepare_image(image, (448, 448)))
            batches.append(pool_features(np.asarray(backbone(np.stack(pixels), training=False)), PLAN["pooling"]))
            if start % 64 == 0:
                print(f"Spatially embedded {min(start + 16, len(rows))}/{len(rows)} real frames", flush=True)
        features = np.concatenate(batches)
        np.savez_compressed(cache_path, embeddings=features, sample_ids=ids,
            dataset_sha256=original["dataset_sha256"], embedding_version=embedding_version)
    masks = {name: np.asarray([row["split"] == name for row in rows]) for name in ("train", "validation", "test")}
    labels = np.asarray([int(row["label"]) for row in rows])
    train_x, train_y = features[masks["train"]], labels[masks["train"]]
    train_groups = np.asarray([row["source_video"] for row in rows])[masks["train"]]
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(train_x, train_y, train_groups))
    candidates, oof_scores = [], {}
    for c in PLAN["classifier_C_candidates"]:
        oof = np.zeros(len(train_y))
        scores = []
        for fit, tune in folds:
            assert not (set(train_groups[fit]) & set(train_groups[tune]))
            scaler = StandardScaler().fit(train_x[fit])
            classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42)
            classifier.fit(scaler.transform(train_x[fit]), train_y[fit])
            raw = classifier.decision_function(scaler.transform(train_x[tune]))
            oof[tune] = raw
            scores.append(float(average_precision_score(train_y[tune], raw)))
        candidates.append({"C": c, "fold_average_precision": scores, "mean_average_precision": float(np.mean(scores))})
        oof_scores[c] = oof
        print(f"Train-only grouped CV C={c}: AP={np.mean(scores):.4f}", flush=True)
    chosen = max(candidates, key=lambda candidate: candidate["mean_average_precision"])
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(C=chosen["C"], class_weight="balanced", max_iter=2000, random_state=42)
    classifier.fit(scaler.transform(train_x), train_y)
    val_raw = classifier.decision_function(scaler.transform(features[masks["validation"]]))
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(val_raw[:, None], labels[masks["validation"]])
    oof_p = calibrator.predict_proba(oof_scores[chosen["C"]][:, None])[:, 1]
    threshold_candidates = []
    for threshold in np.unique(oof_p):
        predicted = oof_p >= threshold
        recall = float(predicted[train_y == 1].mean())
        specificity = float((~predicted[train_y == 0]).mean())
        if recall >= 0.8:
            threshold_candidates.append((specificity, float(threshold), recall))
    specificity, threshold, recall = max(threshold_candidates)
    parameters = {"format_version": 1, "threshold": threshold, "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "classifier_coef": classifier.coef_[0].tolist(),
        "classifier_intercept": float(classifier.intercept_[0]), "calibrator_coef": float(calibrator.coef_[0, 0]),
        "calibrator_intercept": float(calibrator.intercept_[0]), "embedding_version": embedding_version}
    write_json(directory / "parameters.json", parameters)
    selection = {"candidates": candidates, "selected_C": chosen["C"], "threshold": threshold,
        "threshold_train_oof_recall": recall, "threshold_train_oof_specificity": specificity,
        "selection_used_test": False, "calibration_split": "validation"}
    write_json(directory / "selection.json", selection)
    # This is explicitly retrospective because the baseline already exposed this test set.
    test_p = score_embeddings(features[masks["test"]], parameters)
    test_y = labels[masks["test"]]
    evaluation = metrics(test_y, test_p, threshold)
    baseline_metrics = metrics(test_y, np.zeros(len(test_y)), 0.5)
    prior_metrics = metrics(test_y, np.full(len(test_y), float(train_y.mean())), 0.5)
    model_version = file_sha256(directory / "parameters.json")[:16] + "-" + embedding_version[:12]
    limitations = [item.replace("224px", "448px") for item in original["limitations"]] + ["Test images were already scored by the first baseline; this follow-up is retrospective, not an untouched confirmation.",
        "The review threshold targets recall using training-group out-of-fold scores. New recordings can have a different recall/false-alert tradeoff."]
    manifest = {**original, "created_utc": datetime.now(timezone.utc).isoformat(), "model_version": model_version,
        "config": PLAN, "embedding_version": embedding_version, "test_metrics": evaluation,
        "always_normal_baseline": baseline_metrics, "train_prevalence_baseline": prior_metrics,
        "selection": selection, "limitations": limitations, "evaluation_reused": True,
        "validation_result": f"retrospective research measurement: {evaluation['confusion_matrix'][0][1]} false alerts among 180 normal frames; not accepted for autonomous control",
        "deployment_ready": False, "signal_control_validated": False,
        "artifact_sha256": {name: file_sha256(directory / name) for name in ("parameters.json", "backbone.weights.h5", "backbone.json", "embeddings.npz", "input_hashes.json", "experiment_plan.json", "selection.json")}}
    write_json(directory / "manifest.json", manifest)
    test_rows = [row for row in rows if row["split"] == "test"]
    write_json(directory / "test_predictions.json", [{"sample_id": row["sample_id"], "source_video": row["source_video"],
        "label": int(row["label"]), "anomaly_probability": float(p)} for row, p in zip(test_rows, test_p)])
    metric_keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "average_precision", "roc_auc", "brier", "ece_10_bins"]
    table = "\n".join(f"| {key} | {evaluation[key]:.4f} | {baseline_metrics[key]:.4f} | {prior_metrics[key]:.4f} |" for key in metric_keys)
    cv_table = "\n".join(f"| {candidate['C']} | {candidate['mean_average_precision']:.4f} | {candidate['fold_average_precision']} |" for candidate in candidates)
    bins = "\n".join(f"| {b['lower']:.1f}-{b['upper']:.1f} | {b['count']} | {b['mean_probability']:.4f} | {b['anomaly_rate']:.4f} |" if b['count'] else f"| {b['lower']:.1f}-{b['upper']:.1f} | 0 | - | - |" for b in evaluation['reliability_bins'])
    report = f"""# Spatial-feature follow-up on real aerial frames

Model `{model_version}`, generated {manifest['created_utc']}.

**This is a retrospective follow-up.** The same 225-frame test set was previously scored by the [initial baseline](STREET_VISION_EVALUATION.md). Candidate selection and threshold selection use only training groups, but the changed experimental direction followed seeing baseline failure. A new untouched dataset is required for confirmation. This model is not approved for signal control.

## Predeclared plan and training

The frozen ImageNet MobileNetV3Small processes 448x448 RGB images. Spatial mean, maximum, and standard deviation are concatenated into 1728 features to retain stronger local activations than whole-image average pooling. No backbone fine-tuning occurred. Standardization is fitted separately within each training fold; no validation/test image trains the scaler or classifier.

The 1050 training images from 35 source recordings were divided by recording into three folds. Logistic C candidates were fixed before scoring: 0.001, 0.01, 0.1. Mean average precision selects C; ties retain the earlier/smaller C. Full training data refits the chosen model. Sigmoid calibration uses the separate 225-image/8-recording validation split. Review threshold selection uses calibrated training-group out-of-fold predictions to maximize specificity while retaining at least 80% anomaly recall. The threshold is therefore a review operating point, not 50% belief or a traffic-light safety threshold.

| C | Mean training-group CV AP | Fold AP values |
|---|---:|---|
{cv_table}

Selected C: {chosen['C']}. Review threshold: {threshold:.8f}. Threshold-selection recall: {recall:.4f}; specificity: {specificity:.4f}. These development values are not independent evaluation. Candidate selection, probability calibration, and threshold were fixed before this experiment scored the reused test images.

## Retrospective evaluation

The 225 test images contain 180 normal and 45 anomalous frames across eight recordings. Source recordings remain disjoint from training and calibration, and all image/label hashes match the first experiment.

| Metric | Follow-up model | Always normal p=0 | Constant training prevalence p=0.2 |
|---|---:|---:|---:|
{table}

Confusion matrix (rows true normal/anomaly; columns predicted normal/review): `{evaluation['confusion_matrix']}`. This model asks for review on {evaluation['confusion_matrix'][0][1] + evaluation['confusion_matrix'][1][1]} of 225 frames, including {evaluation['confusion_matrix'][0][1]} false alerts and {evaluation['confusion_matrix'][1][1]} detected anomalies. It misses {evaluation['confusion_matrix'][1][0]} anomalies. AP and ROC AUC assess ranking; Brier/ECE assess probabilities; thresholded recall alone does not establish a useful model.

## Reliability bins

| Probability interval | Frames | Mean probability | Observed anomaly fraction |
|---|---:|---:|---:|
{bins}

## Limitations and artifacts

""" + "\n".join(f"- {item}" for item in limitations) + f"""

Artifacts: `.runtime/street_vision_candidate`. Reproduce with `python scripts/train_street_vision_candidate.py --data-dir <intersection_anomalies>`. The original baseline artifacts must exist and are retained. The script refuses to overwrite an evaluated candidate. `experiment_plan.json` is written before feature extraction; `selection.json` records training-only selection and threshold. No pickle is loaded.

Dataset hash: `{original['dataset_sha256']}`. Embedding version: `{embedding_version}`. Parameter hash: `{manifest['artifact_sha256']['parameters.json']}`.
"""
    (ROOT / "reports" / "STREET_VISION_CANDIDATE_EVALUATION.md").write_text(report, encoding="utf-8")
    print(json.dumps({"model_version": model_version, "selection": selection, "test_metrics": evaluation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
