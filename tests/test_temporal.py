"""Checks for the causal v2 pipeline (src/traffic_lstm/pipeline_v2.py).

Runs without pytest:

    python tests/test_temporal.py

Fixtures are small deterministic synthetic series. The Keras checks use a
4-unit LSTM trained for 1 epoch on a few hundred rows, so the whole file
runs in about a minute on CPU. Save/load parity tolerances are tight
(rtol=1e-5, atol=1e-6): same-process CPU inference after a .keras round-trip
should be identical up to floating-point noise, but oneDNN may reorder
reductions on other hardware, so exact bitwise equality is not asserted.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from traffic_lstm import pipeline_v2 as v2

PASSED = []
FAILED = []


def check(name: str):
    def wrapper(fn):
        try:
            fn()
            PASSED.append(name)
            print("  PASS  {}".format(name))
        except AssertionError as exc:
            frame = next((f for f in reversed(traceback.extract_tb(*sys.exc_info()[2:]))
                          if f.filename.endswith("test_temporal.py")), None)
            where = "line {}".format(frame.lineno) if frame else "?"
            FAILED.append((name, "{} ({})".format(exc, where)))
            print("  FAIL  {} -> {} ({})".format(name, exc, where))
        return fn
    return wrapper


class expect_raises:
    """Assert that the body raises (a subclass of) the given exception."""

    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is not None, "expected {} but nothing was raised".format(
            self.exc_type.__name__)
        assert issubclass(exc_type, self.exc_type), \
            "expected {} but got {}: {}".format(
                self.exc_type.__name__, exc_type.__name__, exc)
        return True


TMP = Path(tempfile.mkdtemp(prefix="traffic_v2_tests_"))


# ---------------------------------------------------------------- fixtures --
def make_frame(hours=600, start="2021-01-01", seed=0, trend=0.0,
               exogenous=True) -> pd.DataFrame:
    """A clean daily cycle plus noise; `temp` is deliberately CELSIUS."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    cycle = 3000 + 1200 * np.sin(2 * np.pi * index.hour / 24) + trend * t
    frame = pd.DataFrame({
        "date_time": index,
        "traffic_volume": cycle + rng.normal(0, 30, hours),
    })
    if exogenous:
        frame["temp"] = (10 + 8 * np.sin(2 * np.pi * (index.hour - 14) / 24)
                         + rng.normal(0, 0.4, hours))
        frame["rain_1h"] = np.clip(rng.gamma(0.4, 1.0, hours) - 0.3, 0, None)
    return frame


def _cfg(run_name, out_dir, **params) -> v2.V2Config:
    defaults = dict(
        data_path=TMP / "unused.csv",      # every check passes df= explicitly
        site_scope="test-corridor",
        sequence_length=8,
        horizons=(1,),
        lstm_units=(4,),
        dense_units=4,
        epochs=1,
        batch_size=64,
        patience=2,
        xgb_n_estimators=30,
        xgb_early_stopping=5,
        run_name=run_name,
        output_dir=out_dir,
    )
    defaults.update(params)
    return v2.V2Config(**defaults)


def uv_cfg(run_name, out_dir, **params) -> v2.V2Config:
    params.setdefault("units", {"traffic_volume": "vehicles/hour"})
    return _cfg(run_name, out_dir, **params)


def mv_cfg(run_name, out_dir, **params) -> v2.V2Config:
    params.setdefault("exogenous_columns", ("temp", "rain_1h"))
    params.setdefault("units", {"traffic_volume": "vehicles/hour",
                                "temp": "celsius", "rain_1h": "millimetre"})
    return _cfg(run_name, out_dir, **params)


_TRAINED = {}


def trained(kind):
    """Train (once) and cache a tiny package: "uv" univariate, "mv" multivariate."""
    if kind not in _TRAINED:
        if kind == "uv":
            cfg = uv_cfg("uv_pack", TMP / "uv")
            frame = make_frame(600, seed=1, exogenous=False)
        else:
            cfg = mv_cfg("mv_pack", TMP / "mv", horizons=(1, 2),
                         calendar_features=True)
            frame = make_frame(600, seed=2)
        _TRAINED[kind] = (v2.train_v2(cfg, df=frame, verbose=0), cfg, frame)
    return _TRAINED[kind]


# ------------------------------------------------------------- chronology ---
@check("chronology: fit/validation/test partitions are strictly ordered")
def _():
    cfg = uv_cfg("chrono", TMP / "x1")
    data = v2.prepare_v2(cfg, df=make_frame(400, seed=3, exogenous=False))
    b = data.boundaries
    assert (b["fit"]["rows"], b["validation"]["rows"], b["test"]["rows"]) == (280, 60, 60), b
    val_start = pd.Timestamp(b["validation"]["start"])
    test_start = pd.Timestamp(b["test"]["start"])
    assert pd.Timestamp(b["fit"]["end"]) < val_start
    assert pd.Timestamp(b["validation"]["end"]) < test_start
    fit, val, test = (data.partitions[n] for n in ("fit", "validation", "test"))
    assert pd.Timestamp(fit.target_stamps.max()) < val_start
    assert val.target_stamps.size and pd.Timestamp(val.target_stamps.min()) >= val_start
    assert pd.Timestamp(val.target_stamps.max()) < test_start
    assert pd.Timestamp(test.target_stamps.min()) >= test_start
    # Bridging is allowed: the first validation example's input context ends
    # in the fit partition - past rows as context are history, not leakage.
    assert pd.Timestamp(val.context_end_stamps[0]) < val_start


@check("chronology: scalers are fitted on fit rows only, never validation/test")
def _():
    # A strong trend guarantees validation/test rows exceed the fit range, so
    # a scaler that had seen them would have different parameters.
    frame = make_frame(600, seed=4, trend=3.0, exogenous=True)
    cfg = mv_cfg("chrono2", TMP / "x2", calendar_features=False)
    data = v2.prepare_v2(cfg, df=frame)
    fit_slice = data.series.iloc[: int(600 * 0.7)]
    assert np.isclose(data.target_scaler.data_min_[0],
                      fit_slice["traffic_volume"].min())
    assert np.isclose(data.target_scaler.data_max_[0],
                      fit_slice["traffic_volume"].max())
    assert data.target_scaler.data_max_[0] < data.series["traffic_volume"].max(), \
        "the test-set maximum should be outside the fitted range"
    temp_idx = data.feature_names.index("temp")
    assert np.isclose(data.feature_scaler.data_min_[temp_idx],
                      fit_slice["temp"].min())
    assert np.isclose(data.feature_scaler.data_max_[temp_idx],
                      fit_slice["temp"].max())


# -------------------------------------------------------- future mutation ---
@check("future mutation: rows after the fit boundary cannot change fitted transforms")
def _():
    frame = make_frame(500, seed=5, exogenous=True)
    cfg = mv_cfg("mutate", TMP / "x3", calendar_features=False)
    before = v2.prepare_v2(cfg, df=frame)

    mutated = frame.copy()
    boundary = pd.Timestamp(before.boundaries["validation"]["start"])
    mask = mutated["date_time"] >= boundary
    mutated.loc[mask, "traffic_volume"] *= 5
    mutated.loc[mask, "temp"] += 40
    after = v2.prepare_v2(cfg, df=mutated)

    assert before.series_sha256 != after.series_sha256, "mutation had no effect?"
    for scaler in ("target_scaler", "feature_scaler"):
        assert np.array_equal(getattr(before, scaler).data_min_,
                              getattr(after, scaler).data_min_), scaler
        assert np.array_equal(getattr(before, scaler).data_max_,
                              getattr(after, scaler).data_max_), scaler
    assert np.array_equal(before.partitions["fit"].X, after.partitions["fit"].X)
    assert np.array_equal(before.partitions["fit"].y, after.partitions["fit"].y)
    assert before.coverage["eligible"]["fit"] == after.coverage["eligible"]["fit"]
    assert before.coverage["excluded"] == after.coverage["excluded"]


@check("future mutation: editing test rows leaves validation windows untouched")
def _():
    frame = make_frame(500, seed=6, exogenous=True)
    cfg = mv_cfg("mutate2", TMP / "x4", calendar_features=False)
    before = v2.prepare_v2(cfg, df=frame)

    mutated = frame.copy()
    boundary = pd.Timestamp(before.boundaries["test"]["start"])
    mask = mutated["date_time"] >= boundary
    mutated.loc[mask, "traffic_volume"] *= 7
    after = v2.prepare_v2(cfg, df=mutated)

    for name in ("fit", "validation"):
        assert np.array_equal(before.partitions[name].X, after.partitions[name].X), name
        assert np.array_equal(before.partitions[name].y, after.partitions[name].y), name


# -------------------------------------------------------------- missing -----
@check("missing: non-hourly steps exclude windows, counted by reason")
def _():
    # 60 hourly rows, row 30 dropped -> one 2-hour step at new index 30.
    # n=59, fit=41 rows, val=8, test=10; seq_len=4, horizons=(1,).
    # Candidates: i in [4, 58] -> 55. A window spans rows [i-4, i]; it crosses
    # the broken step 30 iff i-3 <= 30 <= i, i.e. i in [30, 33] -> 4 excluded.
    frame = make_frame(60, seed=7, exogenous=False)
    frame = frame.drop(index=30).reset_index(drop=True)
    cfg = uv_cfg("gap", TMP / "x5", sequence_length=4)
    data = v2.prepare_v2(cfg, df=frame)
    cov = data.coverage
    assert cov["candidate_windows"] == 55, cov
    assert cov["excluded"] == {"non_cadence_step": 4, "missing_input": 0,
                               "missing_target": 0, "boundary_purge": 0}, cov
    # fit candidates i in [4,40] (37) minus the 4 gapped; val [41,48]; test [49,58].
    assert cov["eligible"] == {"fit": 33, "validation": 8, "test": 10}, cov
    assert sum(cov["eligible"].values()) + sum(cov["excluded"].values()) == 55


@check("missing: NaN inputs/targets are excluded and never scored as zero")
def _():
    # 60 clean rows; seq_len=4, horizons=(1,); fit=42, val=9, test=9 rows.
    # Candidates: i in [4, 59] -> 56.
    #   NaN target at row 20: i=20 -> missing_target; i in [21,24] -> missing_input.
    #   NaN temp at row 45: i=45 stays ELIGIBLE (future weather is not an
    #     input; its traffic was observed); i in [46,49] -> missing_input.
    #   NaN target at row 55: i=55 -> missing_target; i in [56,59] -> missing_input.
    frame = make_frame(60, seed=8, exogenous=True)
    frame.loc[20, "traffic_volume"] = np.nan
    frame.loc[45, "temp"] = np.nan
    frame.loc[55, "traffic_volume"] = np.nan
    cfg = mv_cfg("nan", TMP / "x6", exogenous_columns=("temp",),
                 units={"traffic_volume": "vehicles/hour", "temp": "celsius"},
                 sequence_length=4, calendar_features=False)
    data = v2.prepare_v2(cfg, df=frame)
    cov = data.coverage
    assert cov["candidate_windows"] == 56, cov
    assert cov["excluded"] == {"non_cadence_step": 0, "missing_input": 12,
                               "missing_target": 2, "boundary_purge": 0}, cov
    assert cov["eligible"] == {"fit": 33, "validation": 5, "test": 4}, cov

    # Nothing was imputed: the cleaned series still carries the NaNs...
    assert int(data.series["traffic_volume"].isna().sum()) == 2
    assert int(data.series["temp"].isna().sum()) == 1
    # ...the missing hours are nobody's target...
    stamps = data.series["date_time"]
    missing_hours = {stamps.iloc[20], stamps.iloc[55]}
    for name in ("fit", "validation", "test"):
        part = data.partitions[name]
        assert not missing_hours.intersection(pd.Timestamp(t) for t in part.target_stamps.ravel())
        assert np.isfinite(part.y).all()
        # ...and no zero was invented where data was missing (traffic here is
        # in the thousands; a fabricated zero would stand out in real units).
        actual = data.target_scaler.inverse_transform(part.y.reshape(-1, 1))
        assert actual.size == 0 or actual.min() > 100, (name, actual.min())
    # ...while the hour with missing WEATHER but observed traffic is scored.
    val_targets = set(pd.Timestamp(t) for t in data.partitions["validation"].target_stamps.ravel())
    assert stamps.iloc[45] in val_targets


# ------------------------------------------------------------------ units ---
@check("units: undeclared units are rejected")
def _():
    with expect_raises(v2.UnitsDeclarationError):
        v2.V2Config(site_scope="s", run_name="r", output_dir=TMP,
                    exogenous_columns=("temp",),
                    units={"traffic_volume": "vehicles/hour"})   # temp undeclared
    with expect_raises(v2.UnitsDeclarationError):
        v2.V2Config(site_scope="s", run_name="r", output_dir=TMP,
                    units={})                                    # target undeclared
    with expect_raises(v2.UnitsDeclarationError):
        v2.V2Config(site_scope="s", run_name="r", output_dir=TMP,
                    exogenous_columns=("temp",),
                    units={"traffic_volume": "vehicles/hour", "temp": "  "})
    with expect_raises(v2.UnitsDeclarationError):
        v2.V2Config(site_scope="s", run_name="r", output_dir=TMP,
                    units={"traffic_volume": "vehicles/hour",
                           "pressure": "hpa"})                   # unknown column


@check("units: a column named temp is not assumed to be kelvin")
def _():
    frame = make_frame(300, seed=9, exogenous=True)   # temp ~ 0..20 celsius
    cfg = mv_cfg("units_ok", TMP / "x7", exogenous_columns=("temp",),
                 units={"traffic_volume": "vehicles/hour", "temp": "celsius"},
                 calendar_features=False)
    data = v2.prepare_v2(cfg, df=frame)
    temp_idx = data.feature_names.index("temp")
    fit_slice = data.series.iloc[: int(300 * 0.7)]
    # Values pass through exactly as declared - no kelvin conversion anywhere.
    assert np.isclose(data.feature_scaler.data_min_[temp_idx], fit_slice["temp"].min())
    assert data.feature_scaler.data_max_[temp_idx] < 100, \
        "a kelvin assumption would show ~250+ here"
    assert data.cfg.units["temp"] == "celsius"


# ---------------------------------------------------------- boundary purge --
@check("boundary purge: multi-horizon targets never cross a partition")
def _():
    # 300 clean rows; seq_len=8, horizons=(1,3,6) -> max_h=6.
    # fit=210, val=45, test=45 rows. Candidates: i in [8, 294] -> 287.
    # A window's targets are rows {i, i+2, i+5}; it straddles boundary B iff
    # i < B <= i+5, i.e. 5 windows per boundary (B=210 and B=255) -> 10 purged.
    cfg = uv_cfg("purge", TMP / "x8", sequence_length=8, horizons=(1, 3, 6))
    data = v2.prepare_v2(cfg, df=make_frame(300, seed=10, exogenous=False))
    cov = data.coverage
    assert cov["candidate_windows"] == 287, cov
    assert cov["excluded"]["boundary_purge"] == 10, cov
    assert cov["eligible"] == {"fit": 197, "validation": 40, "test": 40}, cov

    val_start = pd.Timestamp(data.boundaries["validation"]["start"])
    test_start = pd.Timestamp(data.boundaries["test"]["start"])
    fit, val, test = (data.partitions[n] for n in ("fit", "validation", "test"))
    assert pd.Timestamp(fit.target_stamps.max()) < val_start
    assert pd.Timestamp(val.target_stamps.min()) >= val_start
    assert pd.Timestamp(val.target_stamps.max()) < test_start
    assert pd.Timestamp(test.target_stamps.min()) >= test_start
    # The purged slots are really gone: no example ends within max_h-1 of a boundary.
    boundary_rows = {210, 255}
    for part in (fit, val, test):
        for i in part.end_index:
            assert not any(i < B <= i + 5 for B in boundary_rows), i


# --------------------------------------------------------------- baselines --
@check("seasonal baseline: wall-clock timestamps, persistence fallback")
def _():
    index = pd.date_range("2021-03-01", periods=100, freq="h")
    frame = pd.DataFrame({"date_time": index,
                          "traffic_volume": 1000 + np.arange(100, dtype=float)})
    # Drop 2021-03-01 10:00 (offset 10) so "24h ago" is absent for one target.
    frame = frame[frame["date_time"] != pd.Timestamp("2021-03-01 10:00")]
    frame = frame.reset_index(drop=True)

    # Target 2021-03-02 10:00: 24h earlier is the dropped hour -> fallback to
    # persistence (context end 09:00 -> 1033). A row-offset baseline would
    # wrongly read row i-24 = offset 9 -> 1009.
    got = v2.seasonal_baseline(frame, "date_time", "traffic_volume",
                               [pd.Timestamp("2021-03-02 10:00")],
                               [pd.Timestamp("2021-03-02 09:00")])
    assert got[0] == 1033.0, got
    assert got[0] != 1009.0
    # Target 2021-03-02 11:00: 24h earlier exists (offset 11) -> 1011.
    got = v2.seasonal_baseline(frame, "date_time", "traffic_volume",
                               [pd.Timestamp("2021-03-02 11:00")],
                               [pd.Timestamp("2021-03-02 10:00")])
    assert got[0] == 1011.0, got
    # Previous-week period: target 2021-03-08 08:00 is offset 176 from the
    # series start (2021-03-01 00:00); 176-168 = offset 8 -> value 1008.
    got = v2.seasonal_baseline(frame, "date_time", "traffic_volume",
                               [pd.Timestamp("2021-03-08 08:00")],
                               [pd.Timestamp("2021-03-08 07:00")],
                               period="168h")
    assert got[0] == 1008.0, got
    # Persistence baseline reads the context end by timestamp.
    got = v2.persistence_baseline(frame, "date_time", "traffic_volume",
                                  [pd.Timestamp("2021-03-02 09:00")])
    assert got[0] == 1033.0, got


# --------------------------------------------------------------- training ---
@check("training: tiny LSTM + XGBoost produce a complete versioned package")
def _():
    out, cfg, _ = trained("uv")
    man = out["manifest"]
    assert man["pipeline_version"] == "v2"
    assert man["preprocessing_version"] == v2.PREPROCESSING_VERSION
    assert man["run_name"] == "uv_pack"
    assert man["site_scope"] == "test-corridor"
    assert man["units"] == {"traffic_volume": "vehicles/hour"}
    assert man["schema"]["feature_names"] == ["traffic_volume"]
    assert man["sequence_length"] == 8 and man["horizons"] == [1]
    assert man["seed"] == cfg.seed
    assert len(man["series_sha256"]) == 64
    assert len(man["models"]["lstm"]["sha256"]) == 64
    assert len(man["models"]["xgboost"]["h1"]["sha256"]) == 64
    assert man["weights_path"] == "model.keras"
    assert man["scalers"]["target"]["data_min"] and man["scalers"]["features"]["data_max"]
    assert man["splits"]["fit"]["end"] < man["splits"]["validation"]["start"]
    assert man["splits"]["validation"]["end"] < man["splits"]["test"]["start"]
    assert man["coverage"]["eligible"]["test"] > 0
    assert man["dependencies"]["numpy"] and man["dependencies"]["tensorflow"]
    assert "null" in man["evaluation"]["protocol"]["undefined_policy"]
    for model_name in ("lstm", "xgboost", "naive_persistence", "naive_seasonal"):
        m = man["evaluation"]["test"]["h1"][model_name]
        assert m["mae"] is not None and m["n"] > 0, (model_name, m)
    package_dir = out["package_dir"]
    assert (package_dir / "model.keras").exists()
    assert (package_dir / "xgb_h1.json").exists()
    assert (package_dir / "manifest.json").exists()


@check("parity: save/reload reproduces predictions (univariate)")
def _():
    out, cfg, _ = trained("uv")
    pack = v2.load_package(out["package_dir"])
    test = out["data"].partitions["test"]
    before = out["model"].predict(test.X, verbose=0)
    after = pack.load_lstm().predict(test.X, verbose=0)
    # Same process, same weights, CPU inference: identical up to float noise.
    assert np.allclose(before, after, rtol=1e-5, atol=1e-6), \
        float(np.max(np.abs(before - after)))
    flat = test.X.reshape(len(test.X), -1)
    before_x = v2._xgb_predict(out["xgb_models"][1], flat)
    after_x = v2._xgb_predict(pack.load_xgboost(1), flat)
    assert np.allclose(before_x, after_x, rtol=1e-5, atol=1e-6), \
        float(np.max(np.abs(before_x - after_x)))


@check("parity: save/reload reproduces predictions (multivariate)")
def _():
    out, cfg, _ = trained("mv")
    pack = v2.load_package(out["package_dir"])
    assert pack.feature_names == ["traffic_volume", "temp", "rain_1h",
                                  "hour_sin", "hour_cos",
                                  "weekday_sin", "weekday_cos"], pack.feature_names
    test = out["data"].partitions["test"]
    before = out["model"].predict(test.X, verbose=0)
    after = pack.load_lstm().predict(test.X, verbose=0)
    assert before.shape == after.shape == (len(test.X), 2)
    assert np.allclose(before, after, rtol=1e-5, atol=1e-6), \
        float(np.max(np.abs(before - after)))
    flat = test.X.reshape(len(test.X), -1)
    for horizon in (1, 2):
        before_x = v2._xgb_predict(out["xgb_models"][horizon], flat)
        after_x = v2._xgb_predict(pack.load_xgboost(horizon), flat)
        assert np.allclose(before_x, after_x, rtol=1e-5, atol=1e-6), horizon


@check("forecast: multivariate inference is correctly timestamped and matches the model")
def _():
    out, cfg, frame = trained("mv")
    pack = v2.load_package(out["package_dir"])
    history = frame.tail(cfg.sequence_length).reset_index(drop=True)
    result = pack.forecast(history)
    end = pd.Timestamp(frame["date_time"].iloc[-1])
    assert [f["horizon"] for f in result["forecasts"]] == [1, 2]
    assert result["forecasts"][0]["timestamp"] == (end + pd.Timedelta(hours=1)).isoformat()
    assert result["forecasts"][1]["timestamp"] == (end + pd.Timedelta(hours=2)).isoformat()
    assert result["forecasts"][0]["units"] == "vehicles/hour"
    assert result["window_end"] == end.isoformat()
    # Values equal the model's own output on the same window, inverse-scaled.
    window = pack.build_input_window(history)
    raw = pack.load_lstm().predict(window, verbose=0)[0]
    expected = pack.target_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()
    assert np.allclose([f["value"] for f in result["forecasts"]], expected,
                       rtol=1e-5, atol=1e-6)
    # The XGBoost path agrees with the reloaded boosters, per horizon.
    xres = pack.forecast(history, model="xgboost")
    flat = window.reshape(1, -1)
    for f, horizon in zip(xres["forecasts"], (1, 2)):
        raw = v2._xgb_predict(pack.load_xgboost(horizon), flat)[0]
        exp = pack.target_scaler.inverse_transform([[raw]])[0, 0]
        assert np.isclose(f["value"], exp, rtol=1e-5, atol=1e-6), horizon
    # The univariate package forecasts too (the old CLI could not do this).
    out_uv, cfg_uv, frame_uv = trained("uv")
    res_uv = v2.load_package(out_uv["package_dir"]).forecast(
        frame_uv.tail(cfg_uv.sequence_length))
    assert len(res_uv["forecasts"]) == 1


# -------------------------------------------------------------- corruption --
@check("corruption: tampered or mismatched artifacts are rejected by hash")
def _():
    out, _, _ = trained("uv")
    src = out["package_dir"]

    dst = TMP / "corrupted_weights"
    shutil.copytree(src, dst)
    path = dst / "model.keras"
    blob = bytearray(path.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    path.write_bytes(bytes(blob))
    with expect_raises(v2.ArtifactCorruptionError):
        v2.load_package(dst)

    dst = TMP / "corrupted_xgb"
    shutil.copytree(src, dst)
    path = dst / "xgb_h1.json"
    blob = bytearray(path.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    path.write_bytes(bytes(blob))
    with expect_raises(v2.ArtifactCorruptionError):
        v2.load_package(dst)

    dst = TMP / "corrupted_manifest"
    shutil.copytree(src, dst)
    manifest_path = dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"]["lstm"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with expect_raises(v2.ArtifactCorruptionError):
        v2.load_package(dst)

    v2.load_package(src)   # the untouched original still verifies and loads


# ----------------------------------------------------------------- metrics --
@check("metrics: undefined percentage metrics serialise as JSON null")
def _():
    m = v2.regression_metrics_v2(np.array([0.2, 0.5, 0.9]), np.array([0.2, 0.4, 0.8]))
    assert m["mape"] is None and m["mae"] is not None, m
    assert json.loads(json.dumps(m))["mape"] is None
    empty = v2.regression_metrics_v2(np.array([]), np.array([]))
    assert empty == {"mae": None, "rmse": None, "mape": None, "n": 0}, empty
    assert v2.improvement_pct(0.0, 0.0) is None      # zero-error baseline
    assert v2.improvement_pct(2.0, 0.0) is None
    assert v2.improvement_pct(3.0, 6.0) == 50.0

    # End to end: a near-zero series yields JSON null in the written manifest.
    rng = np.random.default_rng(11)
    index = pd.date_range("2021-06-01", periods=300, freq="h")
    frame = pd.DataFrame({"date_time": index,
                          "traffic_volume": 0.5 + rng.normal(0, 0.01, 300)})
    cfg = uv_cfg("near_zero", TMP / "nz", sequence_length=6)
    out = v2.train_v2(cfg, df=frame, verbose=0)
    block = out["manifest"]["evaluation"]["test"]["h1"]
    assert block["lstm"]["mape"] is None, block
    assert block["naive_seasonal"]["mape"] is None, block
    assert block["lstm"]["mae"] is not None
    text = (out["package_dir"] / "manifest.json").read_text(encoding="utf-8")
    assert '"mape": null' in text


# ---------------------------------------------------------------- refusals --
@check("refusal: run-name collision and gapped history are refused")
def _():
    out, cfg, frame = trained("uv")
    with expect_raises(v2.ArtifactExistsError):
        v2.train_v2(cfg, df=frame, verbose=0)     # same run_name, same output_dir

    pack = v2.load_package(out["package_dir"])
    history = frame.tail(10).reset_index(drop=True)

    gapped = history.drop(index=5).reset_index(drop=True)   # one missing hour
    with expect_raises(v2.GapHistoryError):
        pack.forecast(gapped)

    with_nan = history.copy()
    with_nan.loc[3, "traffic_volume"] = np.nan
    with expect_raises(v2.GapHistoryError):
        pack.forecast(with_nan)

    with expect_raises(v2.GapHistoryError):
        pack.forecast(history.tail(4))            # shorter than sequence_length

    duplicated = pd.concat([history, history.tail(2)], ignore_index=True)
    with expect_raises(v2.GapHistoryError):
        pack.forecast(duplicated)

    out_mv, _, frame_mv = trained("mv")
    pack_mv = v2.load_package(out_mv["package_dir"])
    with expect_raises(v2.SchemaError):
        pack_mv.forecast(frame_mv.tail(8).drop(columns=["temp"]))
    with expect_raises(v2.SchemaError):
        pack_mv.forecast(frame_mv.tail(8),
                         declared_units={"traffic_volume": "vehicles/hour",
                                         "temp": "kelvin",      # wrong on purpose
                                         "rain_1h": "millimetre"})


# ------------------------------------------------------------------ legacy ---
@check("legacy: nine archived v1 artifacts parse; window_* are metrics-only")
def _():
    models_dir = ROOT / "models"
    if not models_dir.exists():
        return   # archived models not checked out here; nothing to verify
    paths = sorted(models_dir.glob("*_artifacts.json"))
    assert len(paths) == 11, [p.name for p in paths]

    loadable, metrics_only = [], []
    for path in paths:
        info = v2.read_legacy_artifact(path)
        assert info["pipeline_version"] == "v1", path
        record = info["record"]                     # original semantics, untouched
        assert record["run_name"] == path.name.replace("_artifacts.json", "")
        assert "h1" in record["results"], path
        assert record["results"]["h1"]["lstm"]["mae"] > 0, path
        assert record["scaler"]["data_min"] and record["scaler"]["data_max"], path
        assert record["config"]["sequence_length"] >= 1, path
        (metrics_only if info["metrics_only"] else loadable).append(info["run_name"])

    assert sorted(metrics_only) == ["window_012", "window_024"], metrics_only
    assert len(loadable) == 9, loadable

    # Metrics-only records must never load as models.
    for name in ("window_012", "window_024"):
        with expect_raises(v2.MetricsOnlyArtifactError):
            v2.load_legacy_model(models_dir / f"{name}_artifacts.json")

    # A retained v1 model still loads with its original semantics.
    model = v2.load_legacy_model(models_dir / "baseline_univariate_artifacts.json")
    assert model.input_shape[-1] == 1

    # A v2 manifest is not a legacy artifact.
    out, _, _ = trained("uv")
    with expect_raises(ValueError):
        v2.read_legacy_artifact(out["package_dir"] / "manifest.json")


# --------------------------------------------------------------------------
def test_all_checks_passed():
    """pytest entry point: the module-level checks above must all pass."""
    assert not FAILED, FAILED


if __name__ == "__main__":
    print("\n{} passed, {} failed\n".format(len(PASSED), len(FAILED)))
    for name, reason in FAILED:
        print("  {}: {}".format(name, reason))
    sys.exit(1 if FAILED else 0)
