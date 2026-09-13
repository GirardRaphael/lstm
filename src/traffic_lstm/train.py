"""Train, evaluate, and write down everything needed to reproduce the result.

Run it directly:

    python -m traffic_lstm.train
    python -m traffic_lstm.train --data data/raw/my_file.csv --epochs 30
    python -m traffic_lstm.train --horizons 1 3 6 --run-name multi_horizon
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import plots
from .config import MODEL_DIR, REPORT_DIR, TrainingConfig
from .data import DataBundle, build_datasets, inverse_target
from .evaluate import compare_against_baselines, training_diagnosis
from .model import build_model, default_callbacks, set_seeds, summarise


# ------------------------------------------------------------- persistence ---
def scaler_to_dict(scaler) -> dict:
    return {
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "feature_range": list(scaler.feature_range),
    }


def scaler_from_dict(payload: dict):
    """Rebuild a MinMaxScaler from plain JSON - no pickle, no version drift."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=tuple(payload["feature_range"]))
    data_min = np.asarray(payload["data_min"], dtype="float64")
    data_max = np.asarray(payload["data_max"], dtype="float64")
    scaler.fit(np.vstack([data_min, data_max]))
    return scaler


# -------------------------------------------------------------- training ----
def train(cfg: TrainingConfig, bundle: DataBundle | None = None, verbose: int = 1,
          *, extra_callbacks=None) -> dict:
    """Train one model and return a dictionary of everything worth keeping."""
    set_seeds(cfg.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if bundle is None:
        bundle = build_datasets(cfg)

    model = build_model(cfg, n_features=bundle.n_features)
    if verbose:
        print(summarise(model))
        print(f"\nTraining on {len(bundle.X_train):,} sequences, "
              f"testing on {len(bundle.X_test):,}.")
        print(f"Inputs ({bundle.n_features}): {', '.join(bundle.feature_names)}")

    started = time.time()
    history = model.fit(
        bundle.X_train,
        bundle.y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        callbacks=default_callbacks(cfg) + list(extra_callbacks or ()),
        shuffle=False,   # keep the temporal order inside the validation split
        verbose=verbose,
    )
    duration = time.time() - started

    # ---- predictions back in real units (vehicles) ----
    predicted_scaled = model.predict(bundle.X_test, verbose=0)
    predicted = inverse_target(bundle.scaler, predicted_scaled)
    actual = inverse_target(bundle.scaler, bundle.y_test)

    # The unscaled series the baselines need, aligned exactly like the test set.
    values = bundle.series[cfg.target_column].to_numpy(dtype="float64")
    bridged = values[bundle.train_size - cfg.sequence_length:]

    results = {}
    for k, horizon in enumerate(cfg.horizons):
        results[f"h{horizon}"] = compare_against_baselines(
            actual[:, k], predicted[:, k], bridged, cfg.sequence_length, horizon,
            indices=bundle.test_indices,
        )

    model.save(cfg.model_path)

    artifacts = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "data_quality": bundle.quality,
        "feature_names": list(bundle.feature_names),
        "n_features": int(bundle.n_features),
        "train_sequences": int(len(bundle.X_train)),
        "test_sequences": int(len(bundle.X_test)),
        "epochs_run": int(len(history.history["loss"])),
        "training_seconds": round(duration, 1),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "diagnosis": training_diagnosis(history.history, cfg.patience),
        "results": results,
        "model_path": str(cfg.model_path),
        "scaler": scaler_to_dict(bundle.scaler),
        "feature_scaler": scaler_to_dict(bundle.feature_scaler),
        "architecture": summarise(model),
    }
    cfg.artifact_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    if verbose:
        _print_report(cfg, artifacts)

    return {
        "model": model,
        "bundle": bundle,
        "artifacts": artifacts,
        "predicted": predicted,
        "actual": actual,
        "history": history.history,
    }


def _print_report(cfg: TrainingConfig, artifacts: dict) -> None:
    print("\n" + "=" * 62)
    print(f"  RESULTS - {cfg.run_name}")
    print("=" * 62)
    for horizon in cfg.horizons:
        block = artifacts["results"][f"h{horizon}"]
        lstm, seasonal, persistence = (
            block["lstm"], block["naive_same_hour_yesterday"], block["naive_persistence"]
        )
        print(f"\n  Horizon +{horizon}h")
        print(f"    LSTM                 MAE {lstm['mae']:8.1f}   RMSE {lstm['rmse']:8.1f}"
              f"   MAPE {lstm['mape']:5.1f}%")
        print(f"    Same hour yesterday  MAE {seasonal['mae']:8.1f}")
        print(f"    Last hour            MAE {persistence['mae']:8.1f}")
        print(f"    -> {block['improvement_over_best_naive_pct']:+.1f}% vs the best baseline")
    diagnosis = artifacts["diagnosis"]
    print(f"\n  Trained {artifacts['epochs_run']} epochs in "
          f"{artifacts['training_seconds']}s. Model: {artifacts['model_path']}")
    print(f"  Training health: {diagnosis['verdict'].upper()} - {diagnosis['detail']}")
    print("=" * 62 + "\n")


def make_figures(cfg: TrainingConfig, outcome: dict) -> list:
    """Regenerate every figure used by the report and the slides."""
    bundle = outcome["bundle"]
    paths = [
        plots.plot_series(
            bundle.series[cfg.datetime_column].to_numpy(),
            bundle.series[cfg.target_column].to_numpy(),
        ),
        plots.plot_daily_profile(bundle.series, cfg.datetime_column, cfg.target_column),
        plots.plot_history(outcome["history"]),
        plots.plot_predictions(
            outcome["actual"][:, 0],
            outcome["predicted"][:, 0],
            horizon=cfg.horizons[0],
            timestamps=bundle.test_timestamps.to_numpy(),
        ),
        plots.plot_error_distribution(outcome["actual"][:, 0], outcome["predicted"][:, 0]),
        plots.plot_baseline_comparison(outcome["artifacts"]["results"][f"h{cfg.horizons[0]}"]),
    ]
    return paths


# ------------------------------------------------------------------- cli ----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the traffic LSTM.")
    defaults = TrainingConfig()
    p.add_argument("--data", type=Path, default=defaults.data_path)
    p.add_argument("--target", default=defaults.target_column)
    p.add_argument("--datetime", dest="datetime_column", default=defaults.datetime_column)
    p.add_argument("--sequence-length", type=int, default=defaults.sequence_length)
    p.add_argument("--horizons", type=int, nargs="+", default=list(defaults.horizons))
    p.add_argument("--units", type=int, nargs="+", default=list(defaults.lstm_units))
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--patience", type=int, default=defaults.patience,
                   help="EarlyStopping patience, in epochs.")
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--exogenous", nargs="*", default=list(defaults.exogenous_columns),
                   help="Extra input columns, e.g. temp rain_1h snow_1h clouds_all")
    p.add_argument("--calendar", action="store_true",
                   help="Add hour/weekday as sin-cos pairs, plus holiday and weekend flags.")
    p.add_argument("--drop-gapped-windows", action="store_true",
                   help="Discard windows that span a break in the series "
                        "(more correct; costs ~29%% of the windows on this dataset).")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--export-vault", action="store_true",
                   help="Also regenerate the Obsidian vault from the trained model.")
    p.add_argument("--vault-dir", type=Path, default=None,
                   help="Write the vault somewhere other than the default, so a "
                        "second dataset does not overwrite the first one's vault.")
    return p


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = TrainingConfig(
        data_path=args.data,
        target_column=args.target,
        datetime_column=args.datetime_column,
        sequence_length=args.sequence_length,
        horizons=tuple(args.horizons),
        lstm_units=tuple(args.units),
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        exogenous_columns=tuple(args.exogenous),
        calendar_features=args.calendar,
        drop_gapped_windows=args.drop_gapped_windows,
        run_name=args.run_name,
    )
    outcome = train(cfg)
    if not args.no_figures:
        for path in make_figures(cfg, outcome):
            print(f"  figure: {path}")
    if args.export_vault:
        from .config import VAULT_DIR
        from .obsidian_export import export_vault

        export_vault(cfg, outcome["model"], outcome["bundle"], outcome["artifacts"],
                     vault_dir=args.vault_dir or VAULT_DIR)


if __name__ == "__main__":
    main()
