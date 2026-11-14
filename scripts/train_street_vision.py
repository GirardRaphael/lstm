"""Train once on real aerial images, calibrate on validation, then evaluate test.

Fixed configuration is declared below. Test outcomes do not select parameters.
Local data/model artifacts are ignored by Git; reports record reproducibility.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
from street_vision import PREPROCESSING, file_sha256, prepare_image, score_embeddings

CONFIG = {
    "seed": 42, "backbone": "MobileNetV3Small", "pretrained": "ImageNet",
    "backbone_frozen": True, "input_size": [224, 224], "pooling": "avg",
    "classifier": "L2 logistic regression", "classifier_C": 0.1,
    "classifier_class_weight": "balanced", "classifier_max_iter": 2000,
    "calibration": "sigmoid logistic regression on validation scores only",
    "calibration_C": 1.0, "threshold": 0.5, "preprocessing": PREPROCESSING,
}


def reliability_bins(labels, probabilities, count=10):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    bins = []
    for index in range(count):
        lo, hi = index / count, (index + 1) / count
        selected = (probabilities >= lo) & ((probabilities < hi) if index < count - 1 else (probabilities <= hi))
        n = int(selected.sum())
        bins.append({"lower": lo, "upper": hi, "count": n,
                     "mean_probability": float(probabilities[selected].mean()) if n else None,
                     "anomaly_rate": float(labels[selected].mean()) if n else None})
    return bins


def metrics(labels, probabilities, threshold=None):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    threshold = CONFIG["threshold"] if threshold is None else threshold
    predicted = (probabilities >= threshold).astype(int)
    bins = reliability_bins(labels, probabilities)
    ece = sum(row["count"] * abs(row["mean_probability"] - row["anomaly_rate"])
              for row in bins if row["count"]) / len(labels)
    return {"n": len(labels), "positives": int(labels.sum()), "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "brier": float(brier_score_loss(labels, probabilities)),
            "ece_10_bins": float(ece), "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            "reliability_bins": bins}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".runtime" / "street_vision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reuse-model-dir", type=Path, help="Reuse hash-matched frozen weights and existing image embeddings")
    parser.add_argument("--retrospective", action="store_true", help="Explicitly mark a reused-test follow-up")
    parser.add_argument("--report-name", default="STREET_VISION_EVALUATION.md")
    args = parser.parse_args()
    data_dir, model_dir = args.data_dir.resolve(), args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "manifest.json").exists():
        raise SystemExit("An evaluated model already exists. Choose a new model directory for another declared experiment; do not tune against this test result.")
    with (data_dir / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = sorted(list(csv.DictReader(handle)), key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "test")}
    if sum(map(len, splits.values())) != len(rows) or any(not group for group in splits.values()):
        raise ValueError("Every example requires a known nonempty split")
    groups = {name: sorted({row["source_video"] for row in group}) for name, group in splits.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[a]) & set(groups[b]):
            raise ValueError("Source recording leakage across splits")
    image_hashes = []
    for index, row in enumerate(rows):
        image_path = (data_dir / row["image_path"]).resolve()
        if not image_path.is_relative_to(data_dir):
            raise ValueError("Dataset image path escapes data directory")
        image_hashes.append({"sample_id": row["sample_id"], "sha256": file_sha256(image_path)})
        if (index + 1) % 300 == 0:
            print(f"Hashed {index + 1}/{len(rows)} input images", flush=True)
    split_for_id = {r["sample_id"]: r["split"] for r in rows}
    hashes_by_split = {name: {r["sha256"] for r in image_hashes if split_for_id[r["sample_id"]] == name} for name in splits}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if hashes_by_split[a] & hashes_by_split[b]:
            raise ValueError("Exact image duplication across splits")
    labels_hash = file_sha256(data_dir / "labels.csv")
    dataset_hash = hashlib.sha256(json.dumps({"labels_sha256": labels_hash, "images": image_hashes}, sort_keys=True).encode()).hexdigest()
    write_json(model_dir / "input_hashes.json", {"labels_sha256": labels_hash, "dataset_sha256": dataset_hash, "images": image_hashes})
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_HOME", str(model_dir / "keras_cache"))
    import tensorflow as tf
    tf.keras.utils.set_random_seed(CONFIG["seed"])
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    weights_path = model_dir / "backbone.weights.h5"
    if args.reuse_model_dir and not weights_path.exists():
        shutil.copy2(args.reuse_model_dir / "backbone.weights.h5", weights_path)
    backbone = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False,
             weights=None if weights_path.exists() else "imagenet", pooling="avg", include_preprocessing=True)
    if weights_path.exists():
        backbone.load_weights(str(weights_path))
    else:
        backbone.save_weights(str(weights_path))
    backbone.trainable = False
    weights_hash = file_sha256(weights_path)
    embedding_version = hashlib.sha256(json.dumps({"architecture": CONFIG["backbone"], "weights_sha256": weights_hash,
                         "preprocessing": PREPROCESSING, "pooling": "avg"}, sort_keys=True).encode()).hexdigest()
    write_json(model_dir / "backbone.json", {"embedding_version": embedding_version, "weights_sha256": weights_hash,
                "embedding_dimensions": int(backbone.output_shape[-1]), "preprocessing": PREPROCESSING})
    ids = np.asarray([r["sample_id"] for r in rows])
    cache_path = model_dir / "embeddings.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["dataset_sha256"].item()) != dataset_hash or str(cache["embedding_version"].item()) != embedding_version or not np.array_equal(ids, cache["sample_ids"]):
                raise ValueError("Embedding cache does not match hashed inputs/encoder")
            embeddings = cache["embeddings"]
        print("Reused hash-verified image embeddings", flush=True)
    else:
        embeddings = np.empty((len(rows), int(backbone.output_shape[-1])), dtype=np.float32)
        reusable = {}
        if args.reuse_model_dir:
            previous_manifest = json.loads((args.reuse_model_dir / "manifest.json").read_text(encoding="utf-8"))
            previous_cache = args.reuse_model_dir / "embeddings.npz"
            if previous_manifest["embedding_version"] != embedding_version or file_sha256(previous_cache) != previous_manifest["artifact_sha256"]["embeddings.npz"]:
                raise ValueError("Reusable embedding cache/encoder checksum mismatch")
            previous_hashes = {r["sample_id"]: r["sha256"] for r in json.loads((args.reuse_model_dir / "input_hashes.json").read_text(encoding="utf-8"))["images"]}
            current_hashes = {r["sample_id"]: r["sha256"] for r in image_hashes}
            with np.load(previous_cache, allow_pickle=False) as cache:
                reusable = {str(key): feature for key, feature in zip(cache["sample_ids"], cache["embeddings"])
                            if previous_hashes.get(str(key)) == current_hashes.get(str(key))}
        missing = []
        for index, row in enumerate(rows):
            if row["sample_id"] in reusable:
                embeddings[index] = reusable[row["sample_id"]]
            else:
                missing.append(index)
        print(f"Reusing {len(rows) - len(missing)} hash-verified image embeddings; extracting {len(missing)} new images", flush=True)
        for start in range(0, len(missing), args.batch_size):
            batch_indices = missing[start:start + args.batch_size]
            pixels = []
            for index in batch_indices:
                with Image.open(data_dir / rows[index]["image_path"]) as image:
                    pixels.append(prepare_image(image))
            embeddings[batch_indices] = np.asarray(backbone(np.stack(pixels), training=False))
            print(f"Embedded {min(start + args.batch_size, len(missing))}/{len(missing)} new actual images", flush=True)
        np.savez_compressed(cache_path, embeddings=embeddings, sample_ids=ids, dataset_sha256=dataset_hash,
                            embedding_version=embedding_version)
    labels = np.asarray([int(r["label"]) for r in rows])
    masks = {name: np.asarray([r["split"] == name for r in rows]) for name in splits}
    scaler = StandardScaler().fit(embeddings[masks["train"]])
    train_x = scaler.transform(embeddings[masks["train"]])
    classifier = LogisticRegression(C=CONFIG["classifier_C"], class_weight="balanced", max_iter=CONFIG["classifier_max_iter"], random_state=CONFIG["seed"])
    classifier.fit(train_x, labels[masks["train"]])
    validation_scores = classifier.decision_function(scaler.transform(embeddings[masks["validation"]]))
    calibrator = LogisticRegression(C=CONFIG["calibration_C"], max_iter=2000, random_state=CONFIG["seed"])
    calibrator.fit(validation_scores[:, None], labels[masks["validation"]])
    parameters = {"format_version": 1, "threshold": CONFIG["threshold"], "scaler_mean": scaler.mean_.tolist(),
                  "scaler_scale": scaler.scale_.tolist(), "classifier_coef": classifier.coef_[0].tolist(),
                  "classifier_intercept": float(classifier.intercept_[0]), "calibrator_coef": float(calibrator.coef_[0, 0]),
                  "calibrator_intercept": float(calibrator.intercept_[0]), "embedding_version": embedding_version}
    write_json(model_dir / "parameters.json", parameters)
    # The first and only test evaluation happens after the full pipeline is fixed.
    probabilities = score_embeddings(embeddings[masks["test"]], parameters)
    test_labels = labels[masks["test"]]
    evaluation = metrics(test_labels, probabilities)
    baseline = metrics(test_labels, np.zeros_like(test_labels, dtype=float))
    prevalence_baseline = metrics(test_labels, np.full(len(test_labels), float(labels[masks["train"]].mean())))
    calibration_prior = metrics(test_labels, np.full(len(test_labels), float(labels[masks["validation"]].mean())))
    model_version = file_sha256(model_dir / "parameters.json")[:16] + "-" + embedding_version[:12]
    limitations = [
        "Publisher binary labels have not been independently adjudicated; anomalies have no verified subtype or boxes.",
        "Only 45 positive test frames from a small number of held-out recordings; nearby frames are correlated.",
        "Frozen 224px ImageNet features can miss small actors, motion-dependent violations, signal state, and other street details.",
        "Calibration is fitted on 225 validation frames and may fail under new cameras, geography, illumination, or synthetic graphics.",
        "This is a binary image classifier, not a vehicle detector, a learned light policy, or evidence that a signal action is safe.",
    ]
    if args.retrospective:
        limitations.append("This test set was previously evaluated; this expanded-training run is retrospective, not an untouched confirmation.")
    manifest = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "model_version": model_version,
        "config": CONFIG, "dataset": f"UIT-ADrone selected {len(rows)} publisher-labelled real aerial frames",
        "evaluation_reused": bool(args.retrospective),
        "dataset_sha256": dataset_hash, "labels_sha256": labels_hash, "embedding_version": embedding_version,
        "split_counts": {name: {"total": len(group), "normal": sum(int(r["label"]) == 0 for r in group),
                      "anomaly": sum(int(r["label"]) == 1 for r in group), "recordings": len(groups[name])} for name, group in splits.items()},
        "split_recordings": groups, "test_metrics": evaluation, "always_normal_baseline": baseline,
        "train_prevalence_baseline": prevalence_baseline, "calibration_prevalence_baseline": calibration_prior, "limitations": limitations,
        "calibration_fit_split": "validation", "test_used_for_training_or_selection": False,
        "validation_result": "inadequate at preset threshold: no held-out anomalies detected" if evaluation["recall"] == 0 else "measured; research prototype only, no deployment acceptance claim",
        "deployment_ready": False,
        "signal_control_validated": False,
        "artifact_sha256": {name: file_sha256(model_dir / name) for name in ("parameters.json", "backbone.weights.h5", "backbone.json", "embeddings.npz", "input_hashes.json")},
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__}}
    write_json(model_dir / "manifest.json", manifest)
    test_rows = [r for r in rows if r["split"] == "test"]
    write_json(model_dir / "test_predictions.json", [{"sample_id": r["sample_id"], "source_video": r["source_video"],
               "label": int(r["label"]), "anomaly_probability": float(p)} for r, p in zip(test_rows, probabilities)])
    metric_names = [("Accuracy", "accuracy"), ("Anomaly precision", "precision"), ("Anomaly recall", "recall"),
                    ("Anomaly F1", "f1"), ("Balanced accuracy", "balanced_accuracy"), ("Average precision", "average_precision"),
                    ("ROC AUC", "roc_auc"), ("Brier score (lower is better)", "brier"), ("10-bin ECE (lower is better)", "ece_10_bins")]
    table = "\n".join(f"| {label} | {evaluation[key]:.4f} | {baseline[key]:.4f} | {prevalence_baseline[key]:.4f} |" for label, key in metric_names)
    bin_table = "\n".join(f"| {b['lower']:.1f}-{b['upper']:.1f} | {b['count']} | {b['mean_probability']:.4f} | {b['anomaly_rate']:.4f} |" if b['count'] else f"| {b['lower']:.1f}-{b['upper']:.1f} | 0 | - | - |" for b in evaluation['reliability_bins'])
    split_table = "\n".join(f"| {name} | {v['total']} | {v['normal']} | {v['anomaly']} | {v['recordings']} |" for name, v in manifest["split_counts"].items())
    artifact_location = model_dir.relative_to(ROOT).as_posix() if model_dir.is_relative_to(ROOT) else str(model_dir)
    dataset_argument = data_dir.relative_to(ROOT).as_posix() if data_dir.is_relative_to(ROOT) else '"<local dataset folder>"'
    reproduce = f"python scripts/train_street_vision.py --data-dir {dataset_argument} --model-dir {artifact_location}"
    if args.reuse_model_dir:
        reuse = args.reuse_model_dir.resolve()
        reproduce += " --reuse-model-dir " + (reuse.relative_to(ROOT).as_posix() if reuse.is_relative_to(ROOT) else str(reuse))
    if args.retrospective:
        reproduce += " --retrospective"
    reproduce += " --report-name " + args.report_name
    report = f"""# Real aerial-frame anomaly experiment

Generated {manifest['created_utc']}. Model `{model_version}`.

Evaluation scope: {"RETROSPECTIVE: the same test images were previously evaluated; a new untouched dataset is required for confirmation." if args.retrospective else "First fixed-configuration evaluation of this project split."}

**Outcome:** {manifest["validation_result"]}. At the fixed threshold this model detected {evaluation["confusion_matrix"][1][1]} of {evaluation["positives"]} held-out anomaly frames. Test probabilities ranged from {float(probabilities.min()):.4f} to {float(probabilities.max()):.4f}. A fitted calibration step must not be presented as proof of meaningful certainty.

This experiment actually trained a binary classifier on real UIT-ADrone aerial images. It does **not** validate object detection or traffic-light decisions. The current screenshot detector and light controller are separate systems.

## Fixed experiment

ImageNet-pretrained MobileNetV3Small was frozen; each full image was resized to 224x224 using bilinear sampling. Its pooled visual embedding was standardized using training data only. L2 logistic regression (C=0.1, balanced training class weights, seed 42) learned the binary publisher anomaly labels on the training split. A second one-variable logistic model (C=1.0, natural class prevalence) fitted sigmoid probability calibration on validation scores only. The review threshold was fixed at 0.5 before test evaluation. No hyperparameter search or threshold selection used test data. No full neural-network fine-tuning was performed.

| Split | Images | Normal | Anomaly | Source recordings |
|---|---:|---:|---:|---:|
{split_table}

Whole source recordings are disjoint across these splits. Exact image hashes were also checked for cross-split duplication. These are new project splits of the selected publisher data, not a claim to reproduce the original paper benchmark. Nearby frames within a recording are correlated, so 225 test frames are not 225 independent street environments.

## Held-out result

| Metric | Calibrated model | Always normal (p=0) | Training prevalence (p={float(labels[masks["train"]].mean()):.4f}) |
|---|---:|---:|---:|
{table}

Confusion matrix, rows=true [normal, anomaly], columns=predicted [normal, anomaly]: `{evaluation['confusion_matrix']}`. Undefined positive precision for an always-normal classifier is reported as zero. AP uses ranked probabilities; it does not use the 0.5 threshold.

A constant predictor using the validation anomaly prevalence has Brier score {calibration_prior["brier"]:.6f}; this is the relevant probability baseline when training prevalence changes.

The model's Brier score is {'lower' if evaluation['brier'] < prevalence_baseline['brier'] else 'higher'} than the constant training-prevalence predictor, and its balanced accuracy is {'higher' if evaluation['balanced_accuracy'] > 0.5 else 'not higher'} than the always-normal baseline. Fitting a calibrator is **not proof of reliable probabilities**. The table and bins are the measured evidence, and no real-road deployment acceptance criterion has been met.

## Test reliability bins

| Estimated probability interval | Frames | Mean predicted probability | Observed anomaly fraction |
|---|---:|---:|---:|
{bin_table}

ECE is the frame-weighted absolute difference between the final two columns across ten equal-width bins. Small bins and correlated frames make this estimate unstable; no independent multi-site calibration claim is made.

## Limitations and scope

""" + "\n".join(f"- {item}" for item in limitations) + f"""

A 'review' prediction asks a person to inspect a frame. It does not authorize a light change. Anomaly probability is not the probability that a selected signal action is correct. Real-footage policy labels and separate group-held-out policy validation are still required to establish decision quality.

## Reproduction and artifacts

Run `{reproduce}` once. The script refuses to overwrite an evaluated experiment; a new experiment requires a new model directory and an explicit evaluation plan. It downloads official Keras ImageNet weights on first run. Local files under `{artifact_location}` include JSON coefficients/scaler/calibrator, frozen HDF5 weights, hashed embedding cache, per-image SHA256 records, test predictions, and the complete manifest. No pickle deserialization is used.

Dataset SHA256: `{dataset_hash}`. Label CSV SHA256: `{labels_hash}`. Embedding version: `{embedding_version}`. Parameters SHA256: `{manifest['artifact_sha256']['parameters.json']}`. Frozen weights SHA256: `{weights_hash}`.

Preprocessing shared by training and inference: `{PREPROCESSING}`. The exact same frozen embedding is available for separately teaching decisions, but anomaly labels never become policy labels automatically.
"""
    report_path = (ROOT / "reports" / args.report_name).resolve()
    if not report_path.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Report must remain within reports directory")
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"model_version": model_version, "embedding_version": embedding_version,
                      "test_metrics": evaluation, "model_dir": str(model_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
