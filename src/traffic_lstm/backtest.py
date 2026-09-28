"""Recorded rolling-origin, multi-seed retrospective evaluation with calibration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, TrainingConfig
from .evaluation_v2 import direct_features, weekly_mean, diagnostic_metrics, block_interval
from .pipeline_v2 import (
    V2Config,
    prepare_v2,
    regression_metrics_v2,
    seasonal_baseline,
    _early_stopping_roles,
)
from .uncertainty import calibrate, interval_metrics


def run(dataset, output, seeds=(7, 42, 123), epochs=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bike = dataset == "bikes"
    path = PROJECT_ROOT / ("data/samples/bike_sharing_hourly.csv" if bike else "data/raw/Metro_Interstate_Traffic_Volume.csv")
    target = "rentals" if bike else "traffic_volume"
    frame = pd.read_csv(path, usecols=["date_time", target])
    frame.date_time = pd.to_datetime(frame.date_time)
    frame = frame.groupby("date_time", as_index=False)[target].mean().sort_values("date_time")
    # Fractions fixed before any fitting. Expanding fit, disjoint evaluation periods.
    folds = [(0.40, 0.50, 0.55, 0.65), (0.55, 0.65, 0.70, 0.80), (0.70, 0.80, 0.85, 0.95)]
    protocol = {"dataset": dataset, "seeds": list(seeds), "epochs": epochs, "folds": folds,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selection": "validation MAE only", "calibration": "separate next block; nominal 90%",
                "lstm_units": [32, 16], "batch_size": 128, "patience": 5,
                "direct_feature_set": "seasonal_v2",
                "training_roles": "chronological fit core; fit tail for early stopping; validation for candidate selection",
                "status": "retrospective; historical data were inspected previously; no untouched holdout claim"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    from .model import build_model, default_callbacks, set_seeds
    from xgboost import XGBRegressor
    import tensorflow as tf
    results = []
    for fold, fractions in enumerate(folds):
        fit_end, val_end, cal_end, test_end = [int(len(frame) * x) for x in fractions]
        cfg = V2Config(target_column=target, units={target: "counts/hour"}, site_scope=dataset,
                       calendar_features=True, fit_end=str(frame.date_time.iloc[fit_end]),
                       validation_end=str(frame.date_time.iloc[val_end]))
        data = prepare_v2(cfg, frame.iloc[:test_end])
        fit, val, post = [data.partitions[k] for k in ("fit", "validation", "test")]
        train_fit, early_stop, training_roles = _early_stopping_roles(
            fit, cfg.early_stopping_ratio)
        split = pd.Timestamp(frame.date_time.iloc[cal_end])
        calibration_mask = pd.DatetimeIndex(post.target_stamps[:, 0]) < split
        evaluation_mask = ~calibration_mask
        if min(len(fit), len(val), calibration_mask.sum(), evaluation_mask.sum()) < 30:
            raise ValueError("Insufficient eligible windows in a fold")
        def actual(part):
            return data.target_scaler.inverse_transform(part.y).ravel()
        def features(part):
            return direct_features(
                data.series, "date_time", target, part.target_stamps[:, 0],
                part.context_end_stamps, cfg.xgb_direct_feature_set)
        fit_frame = data.series[data.series.date_time < pd.Timestamp(cfg.fit_end)]
        for seed in seeds:
            print(f"{dataset} fold={fold+1} seed={seed}", flush=True)
            tf.keras.backend.clear_session()
            set_seeds(seed)
            shim = TrainingConfig(lstm_units=(32, 16), epochs=epochs, batch_size=128, patience=5,
                                  run_name=f"fold_{fold}_seed_{seed}")
            model = build_model(shim, n_features=len(data.feature_names))
            model.compile(optimizer=model.optimizer, loss="mae")
            history = model.fit(train_fit.X, train_fit.y,
                                validation_data=(early_stop.X, early_stop.y), epochs=epochs,
                                batch_size=128, shuffle=False, callbacks=default_callbacks(shim), verbose=0)
            tree = XGBRegressor(n_estimators=800, max_depth=6, learning_rate=.05,
                                subsample=.8, colsample_bytree=.8, random_state=seed, n_jobs=1,
                                early_stopping_rounds=30, eval_metric="mae")
            tree.fit(features(train_fit), actual(train_fit),
                     eval_set=[(features(early_stop), actual(early_stop))],
                     verbose=False)
            def predictions(part):
                return {"lstm": np.maximum(0, data.target_scaler.inverse_transform(model.predict(part.X, verbose=0)).ravel()),
                        "xgboost": np.maximum(0, tree.predict(features(part))),
                        "hour_of_week_mean": weekly_mean(fit_frame, "date_time", target, part.target_stamps[:, 0]),
                        "yesterday": seasonal_baseline(data.series, "date_time", target, part.target_stamps[:, 0], part.context_end_stamps)}
            validation = {name: regression_metrics_v2(actual(val), pred) for name, pred in predictions(val).items()}
            selected = min(validation, key=lambda name: validation[name]["mae"])
            observed = actual(post)
            stamps = post.target_stamps[:, 0][evaluation_mask]
            row = {"fold": fold+1, "seed": seed, "selected": selected, "validation": validation,
                   "epochs_run": len(history.history["loss"]), "coverage": data.coverage,
                   "training_roles": training_roles,
                   "fit_end": cfg.fit_end, "validation_end": cfg.validation_end,
                   "calibration_end": split.isoformat(), "evaluation_end": str(frame.date_time.iloc[test_end-1]),
                   "evaluation": {}}
            preds = predictions(post)
            prediction_rows = pd.DataFrame({"timestamp": post.target_stamps[:, 0], "actual": observed,
                                           "partition": np.where(calibration_mask, "calibration", "evaluation")})
            for name, pred in preds.items():
                calibration = calibrate(observed[calibration_mask], pred[calibration_mask])
                scored = pred[evaluation_mask]
                metrics = regression_metrics_v2(observed[evaluation_mask], scored)
                metrics.update(diagnostic_metrics(observed[evaluation_mask], scored, stamps, seed=seed))
                metrics["interval"] = interval_metrics(observed[evaluation_mask], scored, calibration)
                peak = (pd.DatetimeIndex(stamps).dayofweek < 5) & np.isin(pd.DatetimeIndex(stamps).hour, [7,8,9,16,17,18])
                metrics["peak_interval"] = interval_metrics(observed[evaluation_mask][peak], scored[peak], calibration) if peak.any() else None
                row["evaluation"][name] = {"metrics": metrics, "calibration": calibration}
                prediction_rows[name] = pred
            diff = np.abs(preds["xgboost"][evaluation_mask]-observed[evaluation_mask]) - np.abs(preds["lstm"][evaluation_mask]-observed[evaluation_mask])
            row["paired_block_sensitivity"] = {str(hours): block_interval(diff, stamps, seed=seed, block_hours=hours) for hours in (24, 72, 168)}
            prediction_rows.to_csv(output / f"fold{fold+1}_seed{seed}.csv", index=False)
            results.append(row)
            (output / "results.json").write_text(json.dumps({"protocol": protocol, "runs": results}, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["motorway", "bikes"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run(args.dataset, args.output, epochs=args.epochs)
