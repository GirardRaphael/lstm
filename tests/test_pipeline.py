"""Checks for the three claims this project makes about itself.

Runs without pytest:

    python tests/test_pipeline.py

1. The split is chronological and the scaler never sees the test set.
2. Each (X, y) pair is aligned with the real timestamps it claims.
3. The NumPy replay of the LSTM cell equals what Keras computes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from traffic_lstm.config import DEFAULT_DATASET, TrainingConfig
from traffic_lstm.data import build_datasets, clean_series, create_sequences
from traffic_lstm.evaluate import naive_persistence, naive_seasonal, regression_metrics

PASSED = []
FAILED = []


def check(name: str):
    def wrapper(fn):
        try:
            fn()
            PASSED.append(name)
            print("  PASS  {}".format(name))
        except AssertionError as exc:
            FAILED.append((name, str(exc)))
            print("  FAIL  {} -> {}".format(name, exc))
        return fn
    return wrapper


def synthetic_frame(hours: int = 2000) -> pd.DataFrame:
    """A clean daily cycle plus noise - predictable enough to assert on."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=hours, freq="h")
    cycle = 3000 + 2000 * np.sin(2 * np.pi * index.hour / 24)
    return pd.DataFrame({"date_time": index,
                         "traffic_volume": cycle + rng.normal(0, 50, hours)})


CFG = TrainingConfig(sequence_length=24, horizons=(1,), train_ratio=0.8)
FRAME = synthetic_frame()
BUNDLE = build_datasets(CFG, df=FRAME)


# --------------------------------------------------------------------------
@check("config: saved Windows dataset paths remain portable")
def _():
    old_path = r"C:\Users\student\Desktop\Traffic_LSTM_Project\data\raw\Metro_Interstate_Traffic_Volume.csv"
    cfg = TrainingConfig(data_path=old_path)
    assert cfg.data_path == DEFAULT_DATASET.resolve(), cfg.data_path
    assert cfg.to_dict()["data_path"] == "data/raw/Metro_Interstate_Traffic_Volume.csv"


@check("config: invalid hyperparameters fail early")
def _():
    invalid = (
        {"sequence_length": 0}, {"lstm_units": ()}, {"lstm_units": (8, 0)},
        {"dense_units": 0}, {"dropout": 1.0}, {"validation_split": 0.0},
    )
    for kwargs in invalid:
        try:
            TrainingConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError("accepted invalid config: {}".format(kwargs))


@check("sequences: X is the 24 hours immediately before y")
def _():
    data = np.arange(200, dtype="float64").reshape(-1, 1)
    X, y, idx = create_sequences(data, 24, (1,))
    assert X.shape == (176, 24, 1), X.shape
    # First window is rows 0..23, predicting row 24.
    assert np.array_equal(X[0, :, 0], np.arange(24)), X[0, :, 0]
    assert y[0, 0] == 24, y[0]
    # Last window must not run past the end of the array.
    assert idx[-1] + 1 - 1 == len(data) - 1, idx[-1]


@check("sequences: multi-horizon targets land on the right steps")
def _():
    data = np.arange(200, dtype="float64").reshape(-1, 1)
    X, y, _ = create_sequences(data, 24, (1, 3, 6))
    assert y.shape[1] == 3, y.shape
    assert list(y[0]) == [24.0, 26.0, 29.0], y[0]


@check("split: training data is strictly older than test data")
def _():
    series = BUNDLE.series
    boundary = series["date_time"].iloc[BUNDLE.train_size]
    assert series["date_time"].iloc[BUNDLE.train_size - 1] < boundary
    assert BUNDLE.test_timestamps.min() >= boundary, "a test window ends before the split"


@check("no leakage: the scaler was fitted on training rows only")
def _():
    values = BUNDLE.series["traffic_volume"].to_numpy()
    train_values = values[: BUNDLE.train_size]
    assert np.isclose(BUNDLE.scaler.data_min_[0], train_values.min())
    assert np.isclose(BUNDLE.scaler.data_max_[0], train_values.max())
    # And the guard only means something if the test set really is different.
    test_values = values[BUNDLE.train_size:]
    assert not np.isclose(train_values.max(), test_values.max()) or \
        not np.isclose(train_values.min(), test_values.min()), \
        "synthetic data too uniform for this check to be meaningful"


@check("no leakage: training windows never contain a test-set hour")
def _():
    n_train_windows = len(BUNDLE.X_train)
    last_train_index = CFG.sequence_length + n_train_windows - 1
    assert last_train_index < BUNDLE.train_size, (last_train_index, BUNDLE.train_size)


@check("test set: timestamps line up with the values being predicted")
def _():
    series = BUNDLE.series
    values = series["traffic_volume"].to_numpy(dtype="float64")
    actual = BUNDLE.scaler.inverse_transform(BUNDLE.y_test).ravel()
    for k in (0, 1, 17, len(actual) - 1):
        stamp = BUNDLE.test_timestamps.iloc[k]
        row = series.index[series["date_time"] == stamp][0]
        assert np.isclose(actual[k], values[row], atol=1e-6), (k, actual[k], values[row])


@check("cleaning: duplicate timestamps are collapsed, not duplicated")
def _():
    doubled = pd.concat([FRAME, FRAME.head(50)], ignore_index=True)
    cleaned = clean_series(doubled, "date_time", "traffic_volume")
    assert len(cleaned) == len(FRAME), (len(cleaned), len(FRAME))
    assert cleaned["date_time"].is_monotonic_increasing
    assert not cleaned["date_time"].duplicated().any()


@check("baselines: persistence and seasonal produce aligned predictions")
def _():
    values = np.arange(500, dtype="float64")
    p = naive_persistence(values, 24, 1)
    s = naive_seasonal(values, 24, 1)
    assert len(p) == len(s) == 500 - 24, (len(p), len(s))
    assert p[0] == 23.0, p[0]        # last hour seen before the target
    assert s[0] == 0.0, s[0]         # same hour, 24 steps earlier


@check("gaps: windows spanning a break are kept by default, dropped on request")
def _():
    frame = synthetic_frame(600)
    # Punch a 10-day hole in the middle of the series.
    hole = (frame["date_time"] >= "2020-01-10") & (frame["date_time"] < "2020-01-20")
    gapped = frame[~hole].reset_index(drop=True)

    loose = build_datasets(TrainingConfig(sequence_length=24, train_ratio=0.8), df=gapped)
    strict = build_datasets(
        TrainingConfig(sequence_length=24, train_ratio=0.8, drop_gapped_windows=True),
        df=gapped)
    assert len(strict.X_train) < len(loose.X_train), \
        "the guard should remove the windows that straddle the hole"
    assert len(loose.X_train) - len(strict.X_train) == 24, \
        (len(loose.X_train), len(strict.X_train))


@check("gaps: baselines stay aligned when windows are dropped")
def _():
    frame = synthetic_frame(600)
    hole = (frame["date_time"] >= "2020-01-10") & (frame["date_time"] < "2020-01-20")
    gapped = frame[~hole].reset_index(drop=True)
    cfg = TrainingConfig(sequence_length=24, train_ratio=0.8, drop_gapped_windows=True)
    bundle = build_datasets(cfg, df=gapped)

    values = bundle.series["traffic_volume"].to_numpy(dtype="float64")
    bridged = values[bundle.train_size - cfg.sequence_length:]
    from traffic_lstm.evaluate import naive_persistence
    preds = naive_persistence(bridged, cfg.sequence_length, 1, bundle.test_indices)
    assert len(preds) == len(bundle.y_test), (len(preds), len(bundle.y_test))
    # The persistence prediction must be the value immediately before the target.
    actual = bundle.scaler.inverse_transform(bundle.y_test).ravel()
    for k in (0, len(preds) // 2, len(preds) - 1):
        end = bundle.test_indices[k]
        assert np.isclose(preds[k], bridged[end - 1])
        assert np.isclose(actual[k], bridged[end], atol=1e-6)


@check("baselines: the index-aware form matches the original loop exactly")
def _():
    from traffic_lstm.evaluate import naive_persistence, naive_seasonal
    series = np.sin(np.arange(400) / 5.0) * 100 + 500
    expected_p = np.array([series[i - 1] for i in range(24, len(series))])
    expected_s = np.array([series[i - 24] if i - 24 >= 0 else series[i - 1]
                           for i in range(24, len(series))])
    assert np.allclose(naive_persistence(series, 24, 1), expected_p)
    assert np.allclose(naive_seasonal(series, 24, 1), expected_s)


@check("metrics: a perfect prediction scores zero")
def _():
    truth = np.array([100.0, 200.0, 300.0])
    m = regression_metrics(truth, truth)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["mape"] == 0.0, m


@check("diagnosis: tells overfitting, capping and convergence apart")
def _():
    from traffic_lstm.evaluate import training_diagnosis

    # Bottomed out at epoch 3 of 12, then climbed steadily.
    overfit = {"val_loss": [0.9, 0.5, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45,
                            0.48, 0.51, 0.54, 0.57]}
    assert training_diagnosis(overfit)["verdict"] == "overfit", training_diagnosis(overfit)

    # Still falling on the final epoch.
    capped = {"val_loss": [0.9, 0.7, 0.55, 0.44, 0.36, 0.30, 0.26, 0.23]}
    assert training_diagnosis(capped)["verdict"] == "capped", training_diagnosis(capped)

    # Best in the middle, flat afterwards.
    flat = {"val_loss": [0.9, 0.5, 0.34, 0.31, 0.302, 0.300, 0.301, 0.303,
                         0.302, 0.304]}
    assert training_diagnosis(flat)["verdict"] == "plateaued", training_diagnosis(flat)

    assert training_diagnosis({"val_loss": [0.4]})["verdict"] == "unknown"


@check("diagnosis: matches the real runs already on disk")
def _():
    import json
    from traffic_lstm.config import MODEL_DIR
    from traffic_lstm.evaluate import training_diagnosis

    path = MODEL_DIR / "baseline_univariate_artifacts.json"
    if not path.exists():
        return  # nothing trained yet; the synthetic cases above still cover it
    artifacts = json.loads(path.read_text(encoding="utf-8"))
    result = training_diagnosis(artifacts["history"])
    assert result["verdict"] in {"plateaued", "overfit", "capped"}, result
    assert result["best_epoch"] <= result["epochs_run"], result


@check("introspection: the NumPy replay matches Keras to floating-point noise")
def _():
    from traffic_lstm.introspect import trace_network
    from traffic_lstm.model import build_model, set_seeds

    set_seeds(1)
    cfg = TrainingConfig(sequence_length=24, lstm_units=(16, 8), dense_units=4)
    model = build_model(cfg)
    # Untrained weights are fine: the replay must match whatever the weights are.
    sample = np.random.default_rng(3).normal(size=(24, 1))
    trace = trace_network(model, sample, verify=True)
    assert trace.max_abs_error < 1e-5, trace.max_abs_error
    layer = trace.layers[0]
    assert layer.h.shape == (24, 16), layer.h.shape
    # Gate values must stay inside their mathematical bounds.
    for key in ("i", "f", "o"):
        gate = layer.gate(key)
        assert gate.min() >= 0.0 and gate.max() <= 1.0, (key, gate.min(), gate.max())
    assert np.abs(layer.g).max() <= 1.0, "candidate memory must be a tanh"


@check("introspection: cell state follows c = f*c_prev + i*g exactly")
def _():
    from traffic_lstm.introspect import trace_network
    from traffic_lstm.model import build_model, set_seeds

    set_seeds(2)
    model = build_model(TrainingConfig(sequence_length=12, lstm_units=(8, 4), dense_units=4))
    trace = trace_network(model, np.random.default_rng(5).normal(size=(12, 1)), verify=False)
    layer = trace.layers[0]
    previous = np.zeros(layer.units)
    for t in range(layer.timesteps):
        expected = layer.f[t] * previous + layer.i[t] * layer.g[t]
        assert np.allclose(layer.c[t], expected, atol=1e-12), t
        assert np.allclose(layer.h[t], layer.o[t] * np.tanh(layer.c[t]), atol=1e-12), t
        previous = layer.c[t]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n{} passed, {} failed\n".format(len(PASSED), len(FAILED)))
    for name, reason in FAILED:
        print("  {}: {}".format(name, reason))
    sys.exit(1 if FAILED else 0)
