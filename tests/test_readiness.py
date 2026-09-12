"""Fault injection and inference tests for the read-only street prototype."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_lstm.config import TrainingConfig
from traffic_lstm.evaluate import naive_seasonal, traffic_level
from traffic_lstm.predict import TrafficForecaster
from traffic_lstm.street_data import ObservationContract, demo_observations, inspect_observations


class StreetDataTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-09-12T12:00:00Z")
        self.frame = demo_observations(self.now)
        self.contract = ObservationContract("DEMO-NOT-A-REAL-STREET")

    def inspect(self, frame=None, **kwargs):
        return inspect_observations(self.frame if frame is None else frame,
                                    self.contract, now=kwargs.get("now", self.now))

    def assertBlocked(self, frame, code, **kwargs):
        report = self.inspect(frame, **kwargs)
        self.assertFalse(report["valid_for_analysis"])
        self.assertIn(code, [issue["code"] for issue in report["issues"]])
        json.dumps(report, allow_nan=False)

    def test_valid_does_not_approve_deployment(self):
        report = self.inspect()
        self.assertTrue(report["valid_for_analysis"])
        self.assertFalse(report["street_deployment_approved"])
        self.assertEqual(report["age_seconds"], 0)

    def test_stale(self):
        self.assertBlocked(self.frame, "stale_observations", now=self.now + pd.Timedelta(minutes=11))

    def test_future(self):
        self.assertBlocked(self.frame, "future_observation", now=self.now - pd.Timedelta(seconds=1))

    def test_naive_missing_and_numeric_timestamps(self):
        for value in ("2026-09-12 12:00:00", None, 123, "invalid"):
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame["timestamp"] = frame["timestamp"].astype(object)
                frame.loc[0, "timestamp"] = value
                self.assertBlocked(frame, "invalid_timestamps")

    def test_duplicate(self):
        frame = self.frame.copy()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        self.assertBlocked(frame, "duplicate_timestamps")

    def test_gap(self):
        self.assertBlocked(self.frame.drop(index=10), "non_contiguous")

    def test_order(self):
        self.assertBlocked(self.frame.iloc[::-1], "out_of_order")

    def test_empty_short_and_missing_columns(self):
        self.assertBlocked(self.frame.iloc[:0], "insufficient_history")
        self.assertBlocked(self.frame.iloc[:3], "insufficient_history")
        self.assertBlocked(self.frame.drop(columns="vehicle_count"), "missing_columns")

    def test_stream_identity(self):
        for value in (None, "another-intersection", ""):
            frame = self.frame.copy()
            frame.loc[0, "stream_id"] = value
            self.assertBlocked(frame, "stream_mismatch")

    def test_interval_units(self):
        self.assertBlocked(self.frame.assign(interval_seconds=3600), "interval_mismatch")

    def test_invalid_counts(self):
        for value in (np.nan, np.inf, -1, 0.5):
            with self.subTest(value=value):
                self.assertBlocked(self.frame.assign(vehicle_count=value), "invalid_counts")

    def test_duplicate_columns(self):
        frame = pd.concat([self.frame, self.frame[["vehicle_count"]]], axis=1)
        self.assertBlocked(frame, "duplicate_columns")

    def test_explicit_offsets_across_dst_are_unambiguous(self):
        frame = self.frame.iloc[:2].copy()
        frame["timestamp"] = ["2026-11-01T01:55:00-04:00", "2026-11-01T01:00:00-05:00"]
        contract = ObservationContract("DEMO-NOT-A-REAL-STREET", minimum_samples=2)
        report = inspect_observations(frame, contract, now="2026-11-01T06:00:00Z")
        self.assertTrue(report["valid_for_analysis"])

    def test_invalid_contract(self):
        for kwargs in ({"interval_seconds": 0}, {"minimum_samples": 0},
                       {"max_age_seconds": -1}, {"interval_seconds": True}):
            with self.assertRaises(ValueError):
                ObservationContract("site", **kwargs)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.cfg = TrainingConfig(sequence_length=3)
        self.scaler = MinMaxScaler().fit(np.array([[0.], [100.]]))
        self.model = Mock(input_shape=(None, 3, 1))
        self.model.predict.return_value = np.array([[0.5]])
        self.forecaster = TrafficForecaster(self.model, self.scaler, self.cfg)

    def test_valid_forecast(self):
        result = self.forecaster.predict([10, 20, 30])
        self.assertEqual(result["forecasts"][0]["volume"], 50)

    def test_bad_input_never_reaches_model(self):
        for values in ([1, 2, np.nan], [1, 2, np.inf], [[1, 2, 3]], [1, 2]):
            with self.assertRaises(ValueError):
                self.forecaster.predict(values)
        self.model.predict.assert_not_called()

    def test_bad_model_output(self):
        for prediction in (np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1., 2.]])):
            self.model.predict.return_value = prediction
            with self.assertRaises(ValueError):
                self.forecaster.predict([1, 2, 3])

    def test_multivariate_requires_full_contract(self):
        self.model.input_shape = (None, 3, 11)
        with self.assertRaisesRegex(ValueError, "feature scaler"):
            self.forecaster.predict([1, 2, 3])

    def test_volume_is_not_congestion(self):
        for volume in (100, 3000, 6000):
            self.assertIn("not assessed", traffic_level(volume)[1])
        with self.assertRaises(ValueError):
            traffic_level(np.nan)

    def test_long_horizon_baseline_never_reads_future(self):
        series = np.arange(200, dtype=float)
        for horizon in (1, 24, 25, 48, 49, 100):
            ends = np.array([24, 25, 30])
            result = naive_seasonal(series, 24, horizon, ends)
            self.assertTrue((result < ends).all(), (horizon, result))


class TrainingIntegrationTests(unittest.TestCase):
    def test_callback_and_scaler_saved_without_touching_existing_models(self):
        import traffic_lstm.config as config
        import traffic_lstm.train as training
        from tensorflow.keras.callbacks import Callback
        from traffic_lstm.data import build_datasets

        observed = []

        class Observer(Callback):
            def on_epoch_end(self, epoch, logs=None):
                observed.append(epoch)

        cfg = TrainingConfig(sequence_length=6, lstm_units=(4,), dense_units=2,
                             epochs=1, batch_size=16, run_name="unit-test-only")
        frame = pd.DataFrame({"date_time": pd.date_range("2020-01-01", periods=160, freq="h"),
                              "traffic_volume": 30 + 10 * np.sin(np.arange(160) / 4)})
        bundle = build_datasets(cfg, df=frame)
        with tempfile.TemporaryDirectory(prefix="lstm-test-") as folder:
            target = Path(folder)
            with patch.object(config, "MODEL_DIR", target), patch.object(training, "MODEL_DIR", target), \
                    patch.object(training, "REPORT_DIR", target):
                result = training.train(cfg, bundle, verbose=0, extra_callbacks=[Observer()])
                self.assertEqual(observed, [0])
                self.assertTrue(cfg.model_path.exists())
                artifact = json.loads(cfg.artifact_path.read_text())
                restored = training.scaler_from_dict(artifact["feature_scaler"])
                np.testing.assert_allclose(restored.transform([[42.]]),
                                           bundle.feature_scaler.transform([[42.]]))
                self.assertTrue(np.isfinite(result["predicted"]).all())


if __name__ == "__main__":
    unittest.main()
