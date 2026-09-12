"""Is the LSTM actually earning its complexity? Benchmark against XGBoost.

"Why an LSTM?" is the question this project will be asked, and the only
convincing answer is a number. Gradient boosting on the same 24 lagged values
is the natural competitor: simpler, faster, easier to interpret, and very
strong on tabular problems.

**The comparison is only worth anything if it is exactly like for like.** So
rather than rebuilding features, this module flattens the *same* tensors the
LSTM was given:

    LSTM     X of shape (n, 24, F)     sequence
    XGBoost  X of shape (n, 24 * F)    the same numbers, order discarded

Same rows, same chronological split, same scaler, same targets, same test
windows. The single difference is that the boosting model cannot see the
ordering of the timesteps except through which column a value landed in.

    python -m traffic_lstm.benchmark
    python -m traffic_lstm.benchmark --run-name multivariate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import MODEL_DIR, TrainingConfig
from .data import build_datasets, inverse_target
from .evaluate import compare_against_baselines, regression_metrics


def flatten(x: np.ndarray) -> np.ndarray:
    """(samples, timesteps, features) -> (samples, timesteps * features)."""
    return x.reshape(len(x), -1)


def feature_labels(feature_names: list, sequence_length: int) -> list:
    """Readable names for the flattened columns, so importances mean something."""
    labels = []
    for t in range(sequence_length):
        for name in feature_names:
            labels.append("{}_t-{}h".format(name, sequence_length - t))
    return labels


def run_benchmark(cfg: TrainingConfig, bundle=None, verbose: bool = True) -> dict:
    """Train XGBoost on the LSTM's own tensors and score it the same way."""
    from xgboost import XGBRegressor

    if bundle is None:
        bundle = build_datasets(cfg)

    X_train, X_test = flatten(bundle.X_train), flatten(bundle.X_test)
    horizon = cfg.horizons[0]
    y_train = bundle.y_train[:, 0]
    y_test = bundle.y_test[:, 0]

    # Keras's validation_split takes the LAST fraction of the training data
    # when shuffle=False. Mirror that exactly so both models see the same
    # amount of data and the same validation window.
    cut = int(len(X_train) * (1 - cfg.validation_split))
    fit_x, val_x = X_train[:cut], X_train[cut:]
    fit_y, val_y = y_train[:cut], y_train[cut:]

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="mae",
        random_state=cfg.seed,
        n_jobs=0,
    )

    started = time.time()
    model.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], verbose=False)
    duration = time.time() - started

    predicted = inverse_target(bundle.scaler, model.predict(X_test))
    actual = inverse_target(bundle.scaler, y_test)

    values = bundle.series[cfg.target_column].to_numpy(dtype="float64")
    bridged = values[bundle.train_size - cfg.sequence_length:]
    comparison = compare_against_baselines(
        actual, predicted, bridged, cfg.sequence_length, horizon,
        indices=bundle.test_indices)

    labels = feature_labels(bundle.feature_names, cfg.sequence_length)
    importances = model.feature_importances_
    order = np.argsort(-importances)[:15]
    top_features = [{"feature": labels[i], "importance": float(importances[i])}
                    for i in order]

    result = {
        "model": "XGBoost {}".format(getattr(__import__("xgboost"), "__version__", "?")),
        "run_name": cfg.run_name,
        "horizon": horizon,
        "n_features_flat": int(X_train.shape[1]),
        "trees_used": int(model.best_iteration + 1),
        "training_seconds": round(duration, 1),
        "metrics": comparison["lstm"],          # the key is generic: it is this model's score
        "naive_persistence": comparison["naive_persistence"],
        "naive_same_hour_yesterday": comparison["naive_same_hour_yesterday"],
        "improvement_over_best_naive_pct": comparison["improvement_over_best_naive_pct"],
        "top_features": top_features,
    }

    if verbose:
        _print(result)
    return {"result": result, "predicted": predicted, "actual": actual,
            "bundle": bundle, "estimator": model}


def _print(result: dict) -> None:
    m = result["metrics"]
    print("\n" + "=" * 62)
    print("  XGBOOST BENCHMARK - {}".format(result["run_name"]))
    print("=" * 62)
    print("  {} flattened inputs, {} trees, {}s".format(
        result["n_features_flat"], result["trees_used"], result["training_seconds"]))
    print("  MAE {:8.1f}   RMSE {:8.1f}   MAPE {:5.1f}%".format(
        m["mae"], m["rmse"], m["mape"]))
    print("  vs best naive baseline: {:+.1f}%".format(
        result["improvement_over_best_naive_pct"]))
    print("\n  Most useful inputs:")
    for item in result["top_features"][:8]:
        print("    {:<28} {:.4f}".format(item["feature"], item["importance"]))
    print("=" * 62 + "\n")


def compare_with_lstm(cfg: TrainingConfig, benchmark_result: dict) -> dict:
    """Put the two models side by side, using the stored LSTM artifacts."""
    if not cfg.artifact_path.exists():
        raise FileNotFoundError(
            "No LSTM artifacts for run '{}'. Train it first.".format(cfg.run_name))
    artifacts = json.loads(cfg.artifact_path.read_text(encoding="utf-8"))
    horizon = cfg.horizons[0]
    block = artifacts["results"]["h{}".format(horizon)]
    lstm, xgb = block["lstm"], benchmark_result["metrics"]

    verdict = ("the LSTM wins" if lstm["mae"] < xgb["mae"]
               else "XGBoost wins" if xgb["mae"] < lstm["mae"] else "a tie")
    gap = (xgb["mae"] - lstm["mae"]) / xgb["mae"] * 100

    return {
        "run_name": cfg.run_name,
        "horizon": horizon,
        "lstm": lstm,
        "xgboost": xgb,
        "naive_same_hour_yesterday": block["naive_same_hour_yesterday"],
        "naive_persistence": block["naive_persistence"],
        "lstm_training_seconds": artifacts["training_seconds"],
        "xgboost_training_seconds": benchmark_result["training_seconds"],
        "lstm_advantage_pct": round(gap, 2),
        "verdict": verdict,
    }


# ------------------------------------------------------------------- cli ----
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline_univariate",
                        help="Which trained LSTM run to match and compare against.")
    args = parser.parse_args(argv)

    probe = TrainingConfig(run_name=args.run_name)
    if not probe.artifact_path.exists():
        raise SystemExit(
            "No artifacts for run '{}'. Train it first:\n"
            "    python -m traffic_lstm.train --run-name {}".format(
                args.run_name, args.run_name))
    artifacts = json.loads(probe.artifact_path.read_text(encoding="utf-8"))
    cfg = TrainingConfig(**artifacts["config"])

    outcome = run_benchmark(cfg)
    side_by_side = compare_with_lstm(cfg, outcome["result"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "benchmark_{}.json".format(cfg.run_name)
    path.write_text(json.dumps(
        {"xgboost": outcome["result"], "comparison": side_by_side}, indent=2),
        encoding="utf-8")

    print("  LSTM    MAE {:8.1f}   ({}s to train)".format(
        side_by_side["lstm"]["mae"], side_by_side["lstm_training_seconds"]))
    print("  XGBoost MAE {:8.1f}   ({}s to train)".format(
        side_by_side["xgboost"]["mae"], side_by_side["xgboost_training_seconds"]))
    print("  -> {} by {:.1f}%".format(
        side_by_side["verdict"], abs(side_by_side["lstm_advantage_pct"])))
    print("  written: {}".format(path))


if __name__ == "__main__":
    main()
