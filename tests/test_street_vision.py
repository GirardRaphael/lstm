from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import unittest
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
sys.path.insert(0, str(ROOT / "scripts"))
from street_vision import StreetVision, file_sha256, pool_features, prepare_image, score_embeddings
from train_street_vision import metrics, reliability_bins
from setup_street_encoder import setup as setup_encoder


def parameters():
    return {"scaler_mean": [1.0, 2.0], "scaler_scale": [2.0, 4.0],
            "classifier_coef": [0.5, -1.0], "classifier_intercept": 0.2,
            "calibrator_coef": 0.8, "calibrator_intercept": -0.1, "threshold": 0.5}


class StreetVisionTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)

    def test_image_preprocessing_is_pixel_only_rgb_and_fixed_range(self):
        image = Image.new("RGBA", (500, 300), (255, 20, 0, 100))
        prepared = prepare_image(image)
        assert prepared.shape == (224, 224, 3)
        assert prepared.dtype == np.float32
        np.testing.assert_array_equal(prepared[0, 0], [255.0, 20.0, 0.0])


    def test_json_inference_matches_declared_standardize_linear_sigmoid_pipeline(self):
        sample = np.asarray([[3.0, 6.0], [1.0, 2.0]])
        expected = 1 / (1 + np.exp(-np.asarray([-0.34, 0.06])))
        np.testing.assert_allclose(score_embeddings(sample, parameters()), expected)


    def test_extreme_values_are_finite_probabilities(self):
        result = score_embeddings(np.asarray([[1e300, 1.0], [-1e300, 1.0]]), parameters())
        assert np.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()


    def test_encoder_only_setup_preserves_existing_contract_without_training_head(self):
        source = ROOT / ".runtime" / "street_vision"
        if not (source / "backbone.weights.h5").exists():
            self.skipTest("Local pretrained backbone is not available")
        directory = Path(self.scratch.name) / "encoder_only"
        directory.mkdir()
        for filename in ("backbone.weights.h5", "backbone.json"):
            shutil.copy2(source / filename, directory / filename)
        before = {filename: file_sha256(directory / filename) for filename in ("backbone.weights.h5", "backbone.json")}
        result = setup_encoder(directory)
        assert result["already_present"] is True
        assert result["anomaly_head_trained"] is False
        assert not (directory / "manifest.json").exists()
        assert not (directory / "parameters.json").exists()
        assert before == {filename: file_sha256(directory / filename) for filename in before}
        status = StreetVision(directory, Path(self.scratch.name) / "no_dataset").status()
        assert status["embedding_available"] is True
        assert status["available"] is False
        assert status["calibrated"] is False

    def test_missing_model_is_explicit_and_does_not_create_predictions(self):
        tmp_path = Path(self.scratch.name)
        model = StreetVision(tmp_path / "model", tmp_path / "data")
        assert model.status()["available"] is False
        with self.assertRaisesRegex(RuntimeError, "backbone"):
            model.embed(Image.new("RGB", (224, 224)))


    def test_spatial_embedding_uses_pixel_feature_mean_max_and_std(self):
        output = np.asarray([[[[1.0], [2.0]], [[3.0], [4.0]]]])
        result = pool_features(output, "spatial_mean_max_std")
        np.testing.assert_allclose(result, [[2.5, 4.0, np.sqrt(1.25)]])
        assert result.shape == (1, 3)

    def test_metric_threshold_is_the_trained_review_operating_point(self):
        result = metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.3, 0.25, 0.4]), threshold=0.2)
        assert result["threshold"] == 0.2
        assert result["recall"] == 1.0
        assert result["confusion_matrix"] == [[1, 1], [0, 2]]

    def test_metrics_do_not_confuse_80_percent_accuracy_with_anomaly_detection(self):
        labels = np.asarray([0] * 8 + [1] * 2)
        result = metrics(labels, np.zeros(10))
        assert result["accuracy"] == 0.8
        assert result["balanced_accuracy"] == 0.5
        assert result["recall"] == 0
        assert result["f1"] == 0
        assert result["average_precision"] == 0.2
        assert result["brier"] == 0.2
        assert result["ece_10_bins"] == 0.2
        assert result["confusion_matrix"] == [[8, 0], [2, 0]]


    def test_reliability_bins_include_probability_one_once(self):
        rows = reliability_bins(np.asarray([0, 1, 1]), np.asarray([0.0, 0.5, 1.0]))
        assert sum(row["count"] for row in rows) == 3
        assert rows[-1]["count"] == 1
        assert rows[-1]["mean_probability"] == 1


    def test_sample_access_rejects_unregistered_or_escaping_paths(self):
        tmp_path = Path(self.scratch.name)
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        with (dataset / "labels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path"])
            writer.writeheader()
            writer.writerow({"sample_id": "bad", "image_path": "../elsewhere.png"})
        model = StreetVision(tmp_path / "model", dataset)
        with self.assertRaisesRegex(ValueError, "Unknown sample"):
            model.sample_path("../../something")
        with self.assertRaisesRegex(ValueError, "within"):
            model.sample_path("bad")


    def test_expanded_model_keeps_original_encoder_and_declares_test_reuse(self):
        directory = ROOT / ".runtime" / "street_vision_expanded"
        if not (directory / "manifest.json").exists():
            self.skipTest("Expanded training experiment is not complete")
        baseline = json.loads((ROOT / ".runtime" / "street_vision" / "manifest.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["evaluation_reused"] is True
        assert manifest["embedding_version"] == baseline["embedding_version"]
        assert manifest["split_counts"]["train"]["total"] == 2550
        assert manifest["split_recordings"] == baseline["split_recordings"]
        assert manifest["test_metrics"]["n"] == 225
        assert manifest["test_metrics"]["threshold"] == 0.5
        assert manifest["config"]["classifier_C"] == 0.1
        assert manifest["test_used_for_training_or_selection"] is False
        for filename, expected in manifest["artifact_sha256"].items():
            assert file_sha256(directory / filename) == expected

    def test_expansion_only_adds_training_rows_and_keeps_holdout_examples(self):
        expanded = ROOT / "data" / "intersection_anomalies_expanded"
        source = ROOT.parents[1] / "data" / "intersection_anomalies"
        if not (expanded / "labels.csv").exists() or not (source / "labels.csv").exists():
            self.skipTest("Local expanded dataset is not available")
        with (source / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
            old = {row["sample_id"]: row for row in csv.DictReader(handle)}
        with (expanded / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
            new = {row["sample_id"]: row for row in csv.DictReader(handle)}
        assert len(new) == 3000
        assert all(new[key] == row for key, row in old.items())
        training_groups = {row["source_video"] for row in old.values() if row["split"] == "train"}
        for key in set(new) - set(old):
            assert new[key]["split"] == "train"
            assert new[key]["source_video"] in training_groups
        assert sum(row["split"] == "test" for row in new.values()) == 225
        assert sum(row["split"] == "validation" for row in new.values()) == 225

    def test_candidate_declares_retrospective_scope_and_training_only_selection(self):
        directory = ROOT / ".runtime" / "street_vision_candidate"
        if not (directory / "manifest.json").exists():
            self.skipTest("Optional spatial-feature follow-up has not been trained")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["evaluation_reused"] is True
        assert manifest["deployment_ready"] is False
        assert manifest["selection"]["selection_used_test"] is False
        assert manifest["selection"]["calibration_split"] == "validation"
        assert manifest["selection"]["threshold_train_oof_recall"] >= 0.8
        assert manifest["selection"]["selected_C"] in [0.001, 0.01, 0.1]
        parameters = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        assert manifest["test_metrics"]["threshold"] == parameters["threshold"]
        for filename, expected in manifest["artifact_sha256"].items():
            assert file_sha256(directory / filename) == expected

    def test_trained_artifacts_preserve_split_and_calibration_contract(self):
        directory = ROOT / ".runtime" / "street_vision"
        if not (directory / "manifest.json").exists():
            self.skipTest("Local real-image model has not been trained")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["calibration_fit_split"] == "validation"
        assert manifest["test_used_for_training_or_selection"] is False
        assert manifest["signal_control_validated"] is False
        groups = {key: set(value) for key, value in manifest["split_recordings"].items()}
        assert not (groups["train"] & groups["validation"])
        assert not (groups["train"] & groups["test"])
        assert not (groups["validation"] & groups["test"])
        assert manifest["split_counts"]["test"]["total"] == 225
        for filename, expected in manifest["artifact_sha256"].items():
            assert file_sha256(directory / filename) == expected

if __name__ == "__main__":
    unittest.main()
