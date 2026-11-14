"""Predeclared recording-group selection for a real aerial anomaly review head.

No test outcome selects model, calibrator, threshold, or activation. Existing
test frames were exposed before this experiment: evaluation is retrospective.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
from street_vision import file_sha256, score_embeddings
from train_street_vision import metrics, write_json

PLAN = {
    "seed": 20261114,
    "encoder": "existing frozen ImageNet MobileNetV3Small, 224x224, average pooling",
    "candidates": [{"kind": "logistic", "C": c} for c in [.001, .01, .1]] + [{"kind": "rbf_svm", "C": c} for c in [.1, 1., 10.]],
    "selection": "Highest mean average precision in three source-recording-disjoint TRAIN folds; first candidate wins exact ties. Scaler is fitted separately inside each training fold.",
    "classifier_class_weight": "balanced",
    "calibration": "Validation split into two disjoint recording-group halves using StratifiedGroupKFold(seed). Fit sigmoid logistic regression C=1 on first half only.",
    "threshold": "Second validation half only: maximize balanced accuracy among thresholds with normal-frame false-positive rate <=0.20; ties favor greater recall then greater threshold.",
    "review_activation_gate": "Threshold-half recall >=0.40, precision >=0.30, FPR <=0.20, and positive calibration slope. This is a development gate, never deployment validation.",
    "test": "Evaluate previously exposed test once after parameters and gate are fixed. Do not change training choices in response. No 90%-reliability claim.",
}


def classifier(candidate):
    if candidate["kind"] == "rbf_svm":
        return SVC(C=candidate["C"], kernel="rbf", gamma="scale", class_weight="balanced", cache_size=256)
    return LogisticRegression(C=candidate["C"], class_weight="balanced", max_iter=3000, random_state=PLAN["seed"])


def choose_review_threshold(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    thresholds = np.r_[np.nextafter(float(probabilities.max()), np.inf), np.unique(probabilities)]
    results = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(((labels == 1) & predicted).sum())
        fp = int(((labels == 0) & predicted).sum())
        recall = tp / int((labels == 1).sum())
        fpr = fp / int((labels == 0).sum())
        if fpr <= .20:
            results.append(((recall + 1 - fpr) / 2, recall, float(threshold)))
    return max(results)[2]


def serialize_model(scaler, model, candidate, calibrator, embedding_version, threshold):
    result = {
        "format_version": 2, "classifier_type": candidate["kind"], "threshold": threshold,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "classifier_intercept": float(model.intercept_[0]),
        "calibrator_coef": float(calibrator.coef_[0, 0]), "calibrator_intercept": float(calibrator.intercept_[0]),
        "embedding_version": embedding_version,
    }
    if candidate["kind"] == "rbf_svm":
        result.update(support_vectors=model.support_vectors_.tolist(), dual_coef=model.dual_coef_[0].tolist(), gamma=float(model._gamma))
    else:
        result["classifier_coef"] = model.coef_[0].tolist()
    return result


def write_public_summary(manifest):
    selection = manifest["selection"]
    summary = {
        "schema_version": 1,
        "training_completed": True,
        "created_utc": manifest["created_utc"],
        "model_version": manifest["model_version"],
        "accepted": manifest["review_candidate_accepted"],
        "active_model_changed": False,
        "outcome": manifest["validation_result"],
        "data_scope": "Publisher-labelled real UIT-ADrone aerial frames, including roundabouts; not validated on signalized FLUID videos.",
        "total_images": sum(split["total"] for split in manifest["split_counts"].values()),
        "split_counts": manifest["split_counts"],
        "calibration": {"frames": selection["calibration_count"], "anomalies": selection["calibration_positive_count"], "recordings": len(selection["calibration_groups"])},
        "threshold_validation": {"frames": selection["threshold_count"], "anomalies": selection["threshold_positive_count"], "recordings": len(selection["threshold_groups"])},
        "selected_classifier": selection["selected"],
        "development_metrics": selection["operating_metrics"],
        "test_metrics": manifest["test_metrics"],
        "evaluation_reused": True,
        "test_used_for_training_or_selection": False,
        "signal_control_validated": False,
        "deployment_ready": False,
    }
    write_json(ROOT / "reports" / "street_anomaly_retraining.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "intersection_anomalies_expanded")
    parser.add_argument("--source-model-dir", type=Path, default=ROOT / ".runtime" / "street_vision_expanded")
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".runtime" / "street_vision_review")
    args = parser.parse_args()
    data, source, target = args.data_dir.resolve(), args.source_model_dir.resolve(), args.model_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if (target / "experiment_plan.json").exists():
        raise SystemExit("An experiment already exists here. Choose a new output directory; do not overwrite an evaluated run.")
    write_json(target / "experiment_plan.json", {"declared_utc": datetime.now(timezone.utc).isoformat(), **PLAN})
    baseline = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    for name, expected in baseline["artifact_sha256"].items():
        if file_sha256(source / name) != expected:
            raise ValueError("Source artifact checksum mismatch: " + name)
    if file_sha256(data / "labels.csv") != baseline["labels_sha256"]:
        raise ValueError("Source labels changed since embedding extraction")
    with (data / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = {r["sample_id"]: r for r in source_rows}
    if len(rows) != len(source_rows) or any(row["label"] not in ("0", "1") for row in source_rows):
        raise ValueError("Labels require unique sample IDs and binary publisher labels")
    hashes = json.loads((source / "input_hashes.json").read_text(encoding="utf8"))
    if len(hashes["images"]) != len(rows) or {entry["sample_id"] for entry in hashes["images"]} != set(rows):
        raise ValueError("Image hash records do not correspond exactly to label rows")
    for entry in hashes["images"]:
        path = (data / rows[entry["sample_id"]]["image_path"]).resolve()
        if not path.is_relative_to(data) or file_sha256(path) != entry["sha256"]:
            raise ValueError("Actual image bytes differ from frozen features")
    with np.load(source / "embeddings.npz", allow_pickle=False) as cache:
        ids = [str(key) for key in cache["sample_ids"]]
        features = np.asarray(cache["embeddings"], dtype=np.float64)
        if str(cache["embedding_version"].item()) != baseline["embedding_version"] or str(cache["dataset_sha256"].item()) != baseline["dataset_sha256"]:
            raise ValueError("Embedding identity differs from source manifest")
    if len(ids) != len(set(ids)) or set(ids) != set(rows) or len(features) != len(ids) or not np.isfinite(features).all():
        raise ValueError("Cached embeddings do not correspond exactly to labelled sample IDs")
    ordered = [rows[key] for key in ids]
    labels = np.asarray([int(row["label"]) for row in ordered])
    groups = np.asarray([row["source_video"] for row in ordered])
    indices = {split: np.asarray([i for i, row in enumerate(ordered) if row["split"] == split]) for split in ("train", "validation", "test")}
    sets = {split: set(groups[index]) for split, index in indices.items()}
    if any(sets[a] & sets[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Source recording leakage")
    train, validation = indices["train"], indices["validation"]
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=PLAN["seed"]).split(features[train], labels[train], groups[train]))
    search = []
    for candidate in PLAN["candidates"]:
        scores = []
        for fit, held in cv:
            scaler = StandardScaler().fit(features[train[fit]])
            model = classifier(candidate).fit(scaler.transform(features[train[fit]]), labels[train[fit]])
            scores.append(float(average_precision_score(labels[train[held]], model.decision_function(scaler.transform(features[train[held]])))))
        result = {**candidate, "fold_average_precision": scores, "mean_average_precision": float(np.mean(scores))}
        search.append(result)
        print(json.dumps(result), flush=True)
    chosen = max(search, key=lambda row: row["mean_average_precision"])
    write_json(target / "train_selection.json", {"plan": PLAN, "results": search, "selected": chosen})
    scaler = StandardScaler().fit(features[train])
    model = classifier(chosen).fit(scaler.transform(features[train]), labels[train])
    first, second = next(StratifiedGroupKFold(2, shuffle=True, random_state=PLAN["seed"]).split(features[validation], labels[validation], groups[validation]))
    calibration, operating = validation[first], validation[second]
    if len(np.unique(labels[calibration])) != 2 or len(np.unique(labels[operating])) != 2:
        raise ValueError("Calibration and threshold groups each need both labels")
    raw_calibration = model.decision_function(scaler.transform(features[calibration]))
    calibrator = LogisticRegression(C=1., max_iter=3000, random_state=PLAN["seed"]).fit(raw_calibration[:, None], labels[calibration])
    raw_operating = model.decision_function(scaler.transform(features[operating]))
    operating_probabilities = calibrator.predict_proba(raw_operating[:, None])[:, 1]
    threshold = choose_review_threshold(labels[operating], operating_probabilities)
    parameters = serialize_model(scaler, model, chosen, calibrator, baseline["embedding_version"], threshold)
    # Inference serialization parity is checked on validation, before opening test.
    np.testing.assert_allclose(score_embeddings(features[operating], parameters), operating_probabilities, rtol=1e-8, atol=1e-9)
    operating_metrics = metrics(labels[operating], operating_probabilities, threshold)
    tn, fp = operating_metrics["confusion_matrix"][0]
    accepted = bool(operating_metrics["recall"] >= .4 and operating_metrics["precision"] >= .3 and fp / (tn + fp) <= .2 and parameters["calibrator_coef"] > 0)
    write_json(target / "parameters.json", parameters)
    decision = {"selected": chosen, "calibration_groups": sorted(set(groups[calibration])), "threshold_groups": sorted(set(groups[operating])),
        "calibration_count": len(calibration), "threshold_count": len(operating), "calibration_positive_count": int(labels[calibration].sum()),
        "threshold_positive_count": int(labels[operating].sum()), "operating_metrics": operating_metrics,
        "review_candidate_accepted": accepted, "calibration_slope": parameters["calibrator_coef"], "threshold": threshold,
        "parameters_sha256_before_test": file_sha256(target / "parameters.json"), "selection_used_test": False}
    write_json(target / "development_decision.json", decision)
    # Test is first scored only after all model/threshold/activation choices are saved.
    test = indices["test"]
    test_probabilities = score_embeddings(features[test], parameters)
    evaluation = metrics(labels[test], test_probabilities, threshold)
    limitations = [
        "Training labels are publisher binary UIT-ADrone frame masks, not independently adjudicated anomalies or verified violation subtypes.",
        "UIT-ADrone includes roundabout and other aerial scenes; these outcomes do not validate FLUID traffic-light footage or signal decisions.",
        "The 225-frame test set was already exposed by earlier experiments. These results are retrospective, not an untouched confirmation.",
        "Only 45 positive test frames and eight test recordings; nearby frames are correlated and apparent frame-level rates are uncertain.",
        "Calibration and operating-threshold subsets each contain only four validation recordings. The development gate is not independent reliability evidence.",
        "Frozen 224px image features cannot reliably establish temporal events, right-of-way, or red-light violations from a single frame.",
        "No pretrained detector weights or traffic-light decision policy were trained in this experiment.",
    ]
    for name in ["backbone.json", "backbone.weights.h5", "input_hashes.json", "embeddings.npz"]:
        try:
            os.link(source / name, target / name)
        except OSError:
            shutil.copy2(source / name, target / name)
    manifest = {"format_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(), "model_version": file_sha256(target / "parameters.json")[:16] + "-review",
        "architecture": "ImageNet MobileNetV3Small frozen 224px features + " + chosen["kind"] + " + separate-group validation sigmoid calibration",
        "dataset": baseline["dataset"], "dataset_sha256": baseline["dataset_sha256"], "labels_sha256": baseline["labels_sha256"],
        "embedding_version": baseline["embedding_version"], "split_counts": baseline["split_counts"], "split_recordings": baseline["split_recordings"],
        "config": PLAN, "selection": decision, "review_candidate_accepted": accepted,
        "test_metrics": evaluation, "previous_expanded_test_metrics": baseline["test_metrics"], "limitations": limitations,
        "calibration_fit_split": "four validation recording groups", "calibration_method": "Sigmoid fit on a calibration-only half of validation recordings; review threshold selected on the other half",
        "test_used_for_training_or_selection": False, "evaluation_reused": True, "deployment_ready": False, "signal_control_validated": False,
        "validation_result": "development review gate passed; unvalidated research candidate" if accepted else "inadequate: development review gate failed; do not activate as a reliable anomaly recognizer",
        "artifact_sha256": {name: file_sha256(target / name) for name in ["parameters.json", "backbone.weights.h5", "backbone.json", "embeddings.npz", "input_hashes.json"]},
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__}}
    write_json(target / "manifest.json", manifest)
    write_public_summary(manifest)
    write_json(target / "test_predictions.json", [{"sample_id": ids[i], "source_video": str(groups[i]), "label": int(labels[i]), "anomaly_probability": float(probability)} for i, probability in zip(test, test_probabilities)])
    report = f"""# Real aerial anomaly retraining

Created {manifest['created_utc']}. Model `{manifest['model_version']}`.

**Result: {manifest['validation_result']}.** This run actually trained a new binary anomaly classifier. It does not train the object detector, create signalized-intersection anomaly ground truth, or learn safe signal decisions.

The pretrained MobileNetV3Small encoder stayed frozen and unchanged (224×224 average-pooled features). The script checked all 3,000 actual image hashes, labels, cached features, and encoder identity before training. The final classifier fitted **2,550 real publisher-labelled frames from 35 recordings**: 1,412 normal and 1,138 anomalous. Validation and test each retain 225 frames from eight disjoint recordings.

## Predeclared selection

The experiment plan was saved before fitting in `.runtime/street_vision_review/experiment_plan.json`. Six balanced classifiers competed only on three training-recording folds. Standardization was fitted inside each training fold. Mean average precision selected `{chosen['kind']}`, C={chosen['C']}, with mean training-CV AP {chosen['mean_average_precision']:.4f}. This CV result is a development selection score, not external validation.

The 225 validation frames were divided by recording into **{len(calibration)} calibration frames** ({int(labels[calibration].sum())} anomalous) and **{len(operating)} threshold-selection frames** ({int(labels[operating].sum())} anomalous). The sigmoid only fitted the first subset. The threshold only used the second subset, maximizing balanced accuracy with a 20% normal-frame false-alert ceiling. Its value is {threshold:.8f}; this is a review operating point, not a reliability percentage.

Development activation required recall >=40%, precision >=30%, false-alert rate <=20%, and a positive sigmoid slope. Accepted: **{accepted}**. Calibration slope: {parameters['calibrator_coef']:.6f}. Threshold-subset precision {operating_metrics['precision']:.1%}, recall {operating_metrics['recall']:.1%}, false alerts {fp}/{tn + fp}.

## Retrospective test result

The test set was previously viewed in older experiments. It was not used for this run's fitting, model choice, calibration, threshold choice, or activation gate. Parameters and the development decision were saved before this test evaluation. No choice was changed in response to its outcome.

| Metric | New candidate | Previous expanded classifier |
|---|---:|---:|
| Anomaly precision | {evaluation['precision']:.2%} | {baseline['test_metrics']['precision']:.2%} |
| Anomaly recall | {evaluation['recall']:.2%} | {baseline['test_metrics']['recall']:.2%} |
| F1 | {evaluation['f1']:.4f} | {baseline['test_metrics']['f1']:.4f} |
| Balanced accuracy | {evaluation['balanced_accuracy']:.2%} | {baseline['test_metrics']['balanced_accuracy']:.2%} |
| Average precision | {evaluation['average_precision']:.4f} | {baseline['test_metrics']['average_precision']:.4f} |
| Overall accuracy | {evaluation['accuracy']:.2%} | {baseline['test_metrics']['accuracy']:.2%} |

Confusion matrix rows=true normal/anomaly, columns=predicted normal/review: `{evaluation['confusion_matrix']}`. The model found {evaluation['confusion_matrix'][1][1]} of 45 anomalies, missed {evaluation['confusion_matrix'][1][0]}, and raised {evaluation['confusion_matrix'][0][1]} false alerts among 180 normal frames. Always predicting normal gives 80% accuracy but 0% anomaly recall; therefore accuracy alone is not success.

## Limits and next evidence

""" + "\n".join("- " + line for line in limitations) + "\n\nNo 90% reliability or real-road safety claim is supported. Obtain independently reviewed anomaly sequences from multiple actual signalized intersections and evaluate on new recording groups before such claims.\n\nReproduce with `python scripts/train_street_anomaly_review.py --model-dir .runtime/street_vision_review_new` after installing the original expanded dataset and frozen feature artifacts. Artifacts use safe JSON/NumPy formats, without pickle execution.\n"
    (ROOT / "reports" / "STREET_ANOMALY_RETRAINING.md").write_text(report, encoding="utf8")
    print(json.dumps({"selected": chosen, "development": decision, "test_metrics": evaluation, "model_dir": str(target)}), flush=True)


if __name__ == "__main__":
    main()
