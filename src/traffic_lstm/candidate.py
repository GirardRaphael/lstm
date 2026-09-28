"""Small portable count forecaster for durable jobs; native tree bytes, no pickle."""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .pipeline_v2 import V2Config, prepare_v2, regression_metrics_v2
from .evaluation_v2 import direct_features, weekly_mean, diagnostic_metrics
from .uncertainty import calibrate, bounds, interval_metrics


def train_candidate(frame, stream_id, *, seed=42):
    if len(frame) < 1000:
        raise ValueError("Training requires at least 1000 observations")
    frame = frame.rename(columns={"timestamp": "date_time"}).copy()
    frame["date_time"] = pd.to_datetime(frame.date_time, utc=True).dt.tz_localize(None)
    cfg = V2Config(datetime_column="date_time", target_column="vehicle_count",
                   units={"vehicle_count": "vehicles/completed_hour"}, site_scope=stream_id,
                   fit_ratio=.64, validation_ratio=.16, calendar_features=True)
    data = prepare_v2(cfg, frame)
    fit, val, post = [data.partitions[k] for k in ("fit", "validation", "test")]
    if min(len(fit), len(val), len(post)) < 100:
        raise ValueError("Insufficient complete windows after exclusions")
    def x(part):
        return direct_features(
            data.series, "date_time", "vehicle_count",
            part.target_stamps[:, 0], part.context_end_stamps,
            cfg.xgb_direct_feature_set)
    def y(part):
        return data.target_scaler.inverse_transform(part.y).ravel()
    tree = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=.05, n_jobs=1,
                        subsample=.8, colsample_bytree=.8, random_state=seed,
                        eval_metric="mae", early_stopping_rounds=30)
    tree.fit(x(fit), y(fit), eval_set=[(x(val), y(val))], verbose=False)
    fit_frame = data.series[data.series.date_time < pd.Timestamp(data.boundaries["validation"]["start"])]
    validation = {
        "xgboost": regression_metrics_v2(y(val), np.maximum(0, tree.predict(x(val)))),
        "hour_of_week_mean": regression_metrics_v2(y(val), weekly_mean(fit_frame, "date_time", "vehicle_count", val.target_stamps[:, 0]))}
    selected = min(validation, key=lambda name: validation[name]["mae"])
    pred = (np.maximum(0, tree.predict(x(post))) if selected == "xgboost" else
            weekly_mean(fit_frame, "date_time", "vehicle_count", post.target_stamps[:, 0]))
    # Split by a RAW timestamp boundary, not the number of eligible windows.
    cal_end = pd.Timestamp(data.series.date_time.iloc[int(len(data.series)*.90)])
    cal = pd.DatetimeIndex(post.target_stamps[:, 0]) < cal_end
    evaluate = ~cal
    if min(cal.sum(), evaluate.sum()) < 30:
        raise ValueError("Need at least 30 windows in calibration and evaluation")
    calibration = calibrate(y(post)[cal], pred[cal])
    metrics = regression_metrics_v2(y(post)[evaluate], pred[evaluate])
    metrics.update(diagnostic_metrics(y(post)[evaluate], pred[evaluate], post.target_stamps[evaluate, 0]))
    metrics["interval"] = interval_metrics(y(post)[evaluate], pred[evaluate], calibration)
    slots = fit_frame.date_time.dt.dayofweek*24 + fit_frame.date_time.dt.hour
    weekly = fit_frame.vehicle_count.groupby(slots).mean()
    payload = bytes(tree.get_booster().save_raw(raw_format="ubj"))
    meta = {"format_version": 1, "stream_id": stream_id, "interval_seconds": 3600,
            "evidence_through": frame.date_time.max().tz_localize("UTC").isoformat(),
            "units": "vehicles/completed_hour", "selected": selected, "validation": validation,
            "calibration": calibration, "evaluation": metrics, "calibration_end": cal_end.isoformat(),
            "splits": data.boundaries, "coverage": data.coverage, "seed": seed,
            "direct_feature_set": cfg.xgb_direct_feature_set,
            "weekly_means": {str(int(k)): float(v) for k,v in weekly.items()},
            "fit_mean": float(fit_frame.vehicle_count.mean()),
            "fit_q01": float(fit_frame.vehicle_count.quantile(.01)),
            "fit_q99": float(fit_frame.vehicle_count.quantile(.99)),
            "model_sha256": hashlib.sha256(payload).hexdigest(),
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("candidate.py", "observations.py", "uncertainty.py", "evaluation_v2.py", "pipeline_v2.py")},
            "python": platform.python_version(),
            "status": "local research candidate; activation is not field validation"}
    return meta, payload


def predict_candidate(meta, payload, frame):
    if hashlib.sha256(payload).hexdigest() != meta["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    history = frame.rename(columns={"timestamp": "date_time"}).copy()
    history.date_time = pd.to_datetime(history.date_time, utc=True).dt.tz_localize(None)
    origin = history.date_time.max()
    target = origin + pd.Timedelta(hours=1)
    if meta["selected"] == "xgboost":
        tree = XGBRegressor()
        tree.load_model(bytearray(payload))
        value = float(tree.predict(direct_features(
            history, "date_time", "vehicle_count", [target], [origin],
            meta.get("direct_feature_set", "seasonal_v1")))[0])
    else:
        value = meta["weekly_means"].get(str(target.dayofweek*24 + target.hour), meta["fit_mean"])
    if not np.isfinite(value):
        raise ValueError("Non-finite model output")
    value = max(0., value)
    low, high = bounds([value], meta["calibration"])
    return {"timestamp": target.tz_localize("UTC").isoformat(), "value": value,
            "lower": float(low[0]), "upper": float(high[0])}
