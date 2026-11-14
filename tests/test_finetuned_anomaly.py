from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation/server"))
sys.path.insert(0, str(ROOT / "scripts"))
from finetune_street_anomaly import augmented_pixels
from finetuned_anomaly import FineTunedAnomaly
from street_vision import file_sha256


class FineTunedAnomalyTests(unittest.TestCase):
    def test_failed_weight_load_cannot_leave_random_model_for_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            weights = directory / "anomaly.weights.h5"
            weights.write_bytes(b"test-only-weights")
            (directory / "manifest.json").write_text(json.dumps({"artifact_sha256": {weights.name: file_sha256(weights)}, "model_version": "test", "review_candidate_accepted": False}), encoding="utf8")
            (directory / "calibration.json").write_text(json.dumps({"coef": 1., "intercept": 0., "threshold": .5}), encoding="utf8")
            failed = Mock()
            failed.load_weights.side_effect = RuntimeError("Failed to load all weights")
            successful = Mock(return_value=np.asarray([[.4]], dtype=np.float32))
            model = FineTunedAnomaly(directory)
            with patch.dict(sys.modules, {"tensorflow": types.ModuleType("tensorflow")}), patch("finetuned_anomaly.build_models", side_effect=[(None, None, None, failed), (None, None, None, successful)]) as builder:
                with self.assertRaisesRegex(RuntimeError, "Failed to load"):
                    model.predict(Image.new("RGB", (32, 32)))
                self.assertIsNone(model._model)
                failed.assert_not_called()
                prediction = model.predict(Image.new("RGB", (32, 32)))
                self.assertIs(model._model, successful)
                self.assertEqual(builder.call_count, 2)
                successful.load_weights.assert_called_once_with(str(weights))
                self.assertAlmostEqual(prediction["anomaly_probability"], .59868766, places=6)

    def test_training_augmentation_is_deterministic_photometric_only(self):
        source = np.linspace(30, 200, 60, dtype=np.float32).reshape(4, 5, 3)
        first = np.clip(augmented_pixels(source, "sample-one"), 0, 255)
        second = np.clip(augmented_pixels(source, "sample-one"), 0, 255)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, source.shape)
        self.assertGreaterEqual(first.min(), 0)
        self.assertLessEqual(first.max(), 255)
        # Positive brightness/contrast preserves spatial order within every channel.
        for channel in range(3):
            np.testing.assert_array_equal(np.argsort(first[:, :, channel].ravel()), np.argsort(source[:, :, channel].ravel()))

    def test_actual_finetune_has_disjoint_groups_and_frozen_pretest_decision(self):
        directory = ROOT / ".runtime/street_anomaly_finetuned"
        if not (directory / "manifest.json").exists():
            self.skipTest("Optional separate fine-tuned anomaly experiment is not installed")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        parts = manifest["partitions"]
        names = list(parts)
        for index, name in enumerate(names):
            for other in names[index + 1:]:
                self.assertFalse(set(parts[name]["recordings"]) & set(parts[other]["recordings"]))
        self.assertEqual(sum(part["frames"] for part in parts.values()), 3000)
        self.assertEqual(parts["test"]["frames"], 225)
        decision = manifest["development_decision"]
        self.assertTrue(decision["visual_weights_changed"])
        self.assertTrue(decision["action_encoder_unchanged"])
        self.assertEqual(file_sha256(directory / "anomaly.weights.h5"), decision["weights_sha256_before_test"])
        self.assertEqual(file_sha256(directory / "calibration.json"), decision["calibration_sha256_before_test"])
        self.assertEqual(file_sha256(ROOT / ".runtime/street_vision_expanded/backbone.weights.h5"), manifest["source_encoder_sha256"])
        self.assertTrue(manifest["evaluation_reused"])
        self.assertFalse(manifest["test_used_for_training_or_selection"])
        self.assertFalse(manifest["deployment_ready"])
        self.assertTrue(manifest["separate_from_action_encoder"])
        model = FineTunedAnomaly(directory)
        self.assertEqual(model.status()["review_candidate_accepted"], manifest["review_candidate_accepted"])

    def test_fresh_pixel_inference_reproduces_saved_probability(self):
        directory = ROOT / ".runtime/street_anomaly_finetuned"
        predictions_path = directory / "test_predictions.json"
        data = ROOT / "data/intersection_anomalies_expanded"
        if not predictions_path.is_file() or not (data / "labels.csv").is_file():
            self.skipTest("Optional completed experiment and source images are not installed")
        predictions = json.loads(predictions_path.read_text(encoding="utf8"))
        with (data / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
        model = FineTunedAnomaly(directory)
        # Fixed indices check saved inference parity, never select or fit a model.
        for saved in [predictions[0], predictions[-1]]:
            row = rows[saved["sample_id"]]
            self.assertEqual(row["split"], "test")
            with Image.open(data / row["image_path"]) as image:
                actual = model.predict(image)
            self.assertAlmostEqual(actual["anomaly_probability"], saved["anomaly_probability"], delta=1e-4)


if __name__ == "__main__":
    unittest.main()
