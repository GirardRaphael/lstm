import numpy as np
import pandas as pd
import pytest

from traffic_lstm.evaluation_v2 import weekly_mean, direct_features, block_interval
from traffic_lstm.pipeline_v2 import seasonal_baseline, regression_metrics_v2, V2Config, prepare_v2, train_v2, load_package


def frame():
    return pd.DataFrame({"date_time": pd.date_range("2020-01-01", periods=1000, freq="h"),
                         "traffic_volume": np.arange(1000, dtype=float)})


def test_climatology_fits_only_supplied_training_rows():
    df = frame()
    target = df.date_time.iloc[800:]
    fit = df.iloc[:600]
    expected = weekly_mean(fit, "date_time", "traffic_volume", target)
    df.loc[600:, "traffic_volume"] = 1e12
    np.testing.assert_array_equal(expected, weekly_mean(df.iloc[:600], "date_time", "traffic_volume", target))
    assert np.isfinite(weekly_mean(fit.head(1), "date_time", "traffic_volume", target)).all()


def test_long_horizon_seasonal_never_reads_after_origin():
    df = frame()
    origin = df.date_time.iloc[200]
    target = origin + pd.Timedelta(hours=49)
    prediction = seasonal_baseline(df, "date_time", "traffic_volume", [target], [origin])
    df.loc[201:, "traffic_volume"] = 1e12
    np.testing.assert_array_equal(prediction, seasonal_baseline(df, "date_time", "traffic_volume", [target], [origin]))


def test_direct_lags_use_timestamps_and_refuse_future_observations():
    df = frame().drop(index=177)
    target = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=201)
    x = direct_features(
        df, "date_time", "traffic_volume", [target],
        [target - pd.Timedelta(hours=1)], "seasonal_v1")
    assert x[0, 0] == 200 and np.isnan(x[0, 1]) and x[0, 2] == 33
    future = direct_features(
        df, "date_time", "traffic_volume", [target],
        [target - pd.Timedelta(hours=25)], "seasonal_v1")
    assert np.isnan(future[0, :2]).all()


def test_direct_v2_features_are_causal_and_do_not_fill_outages():
    df = frame()
    target = pd.Timestamp("2020-01-10 00:00")
    origin = target - pd.Timedelta(hours=1)
    got = direct_features(
        df, "date_time", "traffic_volume", [target], [origin],
        "seasonal_v2")
    assert got.shape == (1, 14)
    assert got[0, :6].tolist() == [215, 214, 213, 192, 191, 48]
    assert got[0, 6] == pytest.approx(np.mean([213, 214, 215]))
    assert got[0, 7] == pytest.approx(np.mean(np.arange(192, 216)))
    assert got[0, 8] == pytest.approx(np.std(np.arange(192, 216)))

    missing = df[df.date_time != origin - pd.Timedelta(hours=5)]
    with_gap = direct_features(
        missing, "date_time", "traffic_volume", [target], [origin],
        "seasonal_v2")
    assert np.isnan(with_gap[0, 7:9]).all()


def test_intervals_preserve_calendar_blocks_and_constant_difference():
    stamps = pd.to_datetime(["2020-01-01", "2020-01-01 01:00", "2020-03-01"], format="mixed")
    ci = block_interval(np.ones(3) * -5, stamps)
    assert ci["blocks"] == 2 and ci["low"] == ci["high"] == -5
    assert block_interval([1], stamps[:1]) is None


def test_metrics_reject_broadcasting_and_nonfinite():
    with pytest.raises(ValueError):
        regression_metrics_v2([1, 2], [1])
    with pytest.raises(ValueError):
        regression_metrics_v2([1], [np.inf])
    metrics = regression_metrics_v2([0, 10, 20], [0, 12, 10])
    assert metrics["median_ae"] == 2
    assert metrics["p90_ae"] == pytest.approx(8.4)
    assert metrics["wape"] == pytest.approx(40)
    assert metrics["smape"] is not None


def test_future_mutation_cannot_change_fit_or_validation_inputs():
    cfg = V2Config(units={"traffic_volume": "vehicles/hour"}, site_scope="test", calendar_features=True)
    df = frame()
    original = prepare_v2(cfg, df)
    df.loc[850:, "traffic_volume"] = 1e12
    changed = prepare_v2(cfg, df)
    for name in ("fit", "validation"):
        np.testing.assert_array_equal(original.partitions[name].X, changed.partitions[name].X)
        np.testing.assert_array_equal(original.partitions[name].y, changed.partitions[name].y)


def test_direct_tree_package_roundtrip_and_baseline_alignment(tmp_path):
    cfg = V2Config(units={"traffic_volume": "vehicles/hour"}, site_scope="test",
                   output_dir=tmp_path, run_name="direct", xgb_layout="direct_lags",
                   epochs=1, lstm_units=(4,), xgb_n_estimators=5, batch_size=128)
    df = frame()
    result = train_v2(cfg, df, verbose=0)
    package = load_package(result["package_dir"])
    part = result["data"].partitions["test"]
    end = pd.Timestamp(part.context_end_stamps[-1])
    history = df[df.date_time <= end]
    features = direct_features(df, "date_time", "traffic_volume", part.target_stamps[-1:, 0], [end])
    scaled = result["xgb_models"][1].predict(features)
    expected = result["data"].target_scaler.inverse_transform(scaled.reshape(-1, 1))[0, 0]
    assert package.forecast(history, model="xgboost")["forecasts"][0]["value"] == pytest.approx(expected)
    metrics = result["evaluation"]["test"]["h1"]
    assert len({metrics[name]["n"] for name in ("lstm", "xgboost", "hour_of_week_mean", "naive_seasonal", "naive_last_week")}) == 1
    assert result["manifest"]["selected_on_validation"] is not None
    assert result["manifest"]["baselines"]["hour_of_week_mean"]["slots"]
    selected = package.forecast(history)
    assert selected["model"] == result["manifest"]["selected_on_validation"]
    short_history = history.tail(cfg.sequence_length)
    degraded = package.forecast(short_history, model="xgboost")
    assert degraded["input_warnings"]
    assert "target_t-168h" in degraded["input_warnings"][0]
    cfg_default = V2Config(units={"traffic_volume": "vehicles/hour"}, site_scope="test")
    assert cfg_default.xgb_layout == "direct_lags"
