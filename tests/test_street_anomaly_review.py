from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
sys.path.insert(0, str(ROOT / "scripts"))
from street_vision import StreetVision, file_sha256, score_embeddings
from train_street_anomaly_review import choose_review_threshold, serialize_model


class AnomalyReviewTests(unittest.TestCase):
    def test_rbf_json_inference_matches_sklearn_on_new_examples(self):
        features, labels = make_moons(120, noise=.12, random_state=7)
        scaler = StandardScaler().fit(features[:80])
        model = SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced").fit(scaler.transform(features[:80]), labels[:80])
        calibration_scores = model.decision_function(scaler.transform(features[80:100]))
        calibrator = LogisticRegression().fit(calibration_scores[:, None], labels[80:100])
        parameters = serialize_model(scaler, model, {"kind": "rbf_svm"}, calibrator, "test-version", .4)
        # A JSON round trip exercises precisely the non-pickle inference format.
        parameters = json.loads(json.dumps(parameters))
        expected = calibrator.predict_proba(model.decision_function(scaler.transform(features[100:]))[:, None])[:, 1]
        np.testing.assert_allclose(score_embeddings(features[100:], parameters), expected, rtol=1e-9)

    def test_threshold_constrains_false_alerts_and_detects_separable_anomalies(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        probabilities = np.array([.1, .2, .25, .3, .7, .6, .8, .9])
        threshold = choose_review_threshold(labels, probabilities)
        predictions = probabilities >= threshold
        self.assertEqual(int(predictions[labels == 1].sum()), 3)
        self.assertEqual(int(predictions[labels == 0].sum()), 1)

    def test_identical_scores_do_not_trigger_unbounded_false_alerts(self):
        labels = np.array([0, 0, 0, 0, 1])
        probabilities = np.full(5, .2)
        threshold = choose_review_threshold(labels, probabilities)
        self.assertGreater(threshold, .2)

    def test_actual_run_preserves_group_separation_and_pretest_model(self):
        directory = ROOT / ".runtime" / "street_vision_review"
        if not (directory / "manifest.json").exists():
            self.skipTest("Optional local anomaly review experiment is not installed")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        decision = manifest["selection"]
        self.assertFalse(manifest["test_used_for_training_or_selection"])
        self.assertFalse(decision["selection_used_test"])
        self.assertTrue(manifest["evaluation_reused"])
        self.assertFalse(manifest["deployment_ready"])
        self.assertFalse(set(decision["calibration_groups"]) & set(decision["threshold_groups"]))
        self.assertEqual(set(decision["calibration_groups"]) | set(decision["threshold_groups"]), set(manifest["split_recordings"]["validation"]))
        self.assertEqual(file_sha256(directory / "parameters.json"), decision["parameters_sha256_before_test"])
        self.assertEqual(manifest["split_counts"]["train"]["total"], 2550)
        for name, expected in manifest["artifact_sha256"].items():
            self.assertEqual(file_sha256(directory / name), expected)
        model = StreetVision(directory, ROOT / "data" / "intersection_anomalies_expanded")
        status = model.status()
        self.assertEqual(status["review_candidate_accepted"], manifest["review_candidate_accepted"])
        with np.load(directory / "embeddings.npz", allow_pickle=False) as cache:
            feature = cache["embeddings"][0]
        result = model.predict_embedding(feature, embedding_version=manifest["embedding_version"])
        self.assertTrue(np.isfinite(result["anomaly_probability"]))
        self.assertFalse(result["deployment_ready"])


if __name__ == "__main__":
    unittest.main()
