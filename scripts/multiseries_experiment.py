"""Do several correlated series favour the recurrence? Measure it.

This is the second half of the question left open by the benchmark. On a single
series XGBoost won comfortably; the standard defence of a recurrent model is
that it should pull ahead when **several correlated series share one model**,
because the recurrence learns one representation across the channels while a
tree receives one flat column per (series x timestep).

The test: predict PM2.5 at one Beijing station, twice.

* **single** - that station's own history, plus calendar columns
* **multi**  - the same, plus the other 11 stations as input channels

The stations correlate between 0.78 and 0.97, so the extra channels really do
carry information. What matters is not which model wins outright, but **how
much each family gains from the extra series** - that is the quantity the
argument for an LSTM is actually about.

    python scripts/multiseries_experiment.py
    python scripts/multiseries_experiment.py --station Dongsi --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from traffic_lstm.benchmark import run_benchmark  # noqa: E402
from traffic_lstm.config import REPORT_DIR, VAULT_DIR, TrainingConfig  # noqa: E402
from traffic_lstm.data import build_datasets  # noqa: E402
from traffic_lstm.train import train  # noqa: E402

DATA = ROOT / "data" / "samples" / "beijing_pm25_multisite.csv"
RESULTS = REPORT_DIR / "multiseries_experiment.json"


def stations(path: Path) -> list:
    columns = list(pd.read_csv(path, nrows=1).columns)
    return [c for c in columns if c != "date_time"]


def one_run(cfg: TrainingConfig, label: str) -> dict:
    print("\n" + "=" * 66)
    print("  {}".format(label.upper()))
    print("=" * 66)

    started = time.time()
    bundle = build_datasets(cfg)
    print("  inputs ({}): {}".format(
        bundle.n_features, ", ".join(bundle.feature_names[:6])
        + (" ..." if len(bundle.feature_names) > 6 else "")))
    outcome = train(cfg, bundle=bundle, verbose=0)
    lstm_seconds = time.time() - started

    artifacts = outcome["artifacts"]
    block = artifacts["results"]["h1"]
    bench = run_benchmark(cfg, bundle=bundle, verbose=False)["result"]

    row = {
        "label": label,
        "run_name": cfg.run_name,
        "n_features": artifacts["n_features"],
        "flat_columns": bench["n_features_flat"],
        "train_sequences": artifacts["train_sequences"],
        "test_sequences": artifacts["test_sequences"],
        "lstm_mae": block["lstm"]["mae"],
        "lstm_rmse": block["lstm"]["rmse"],
        "lstm_epochs": artifacts["epochs_run"],
        "lstm_seconds": round(lstm_seconds, 1),
        "lstm_diagnosis": artifacts["diagnosis"]["verdict"],
        "xgb_mae": bench["metrics"]["mae"],
        "xgb_rmse": bench["metrics"]["rmse"],
        "xgb_seconds": bench["training_seconds"],
        "naive_persistence": block["naive_persistence"]["mae"],
        "naive_seasonal": block["naive_same_hour_yesterday"]["mae"],
        "top_features": bench["top_features"][:8],
    }
    print("  LSTM    MAE {:7.1f}  ({} epochs, {:,.0f}s, {})".format(
        row["lstm_mae"], row["lstm_epochs"], row["lstm_seconds"], row["lstm_diagnosis"]))
    print("  XGBoost MAE {:7.1f}  ({:,} flat columns, {:,.0f}s)".format(
        row["xgb_mae"], row["flat_columns"], row["xgb_seconds"]))
    return row


def vault_note(single: dict, multi: dict, station: str, others: int) -> str:
    lstm_gain = (single["lstm_mae"] - multi["lstm_mae"]) / single["lstm_mae"] * 100
    xgb_gain = (single["xgb_mae"] - multi["xgb_mae"]) / single["xgb_mae"] * 100

    if lstm_gain > xgb_gain + 1.0:
        verdict = (
            "**The recurrence gains more from the extra series than the tree does**\n"
            "({lg:+.1f}% against {xg:+.1f}%). This is the first result in the project\n"
            "that supports the usual argument for an LSTM on its own terms: the\n"
            "advantage is not that the data is \"sequential\", it is that a shared\n"
            "representation across {n} correlated channels is worth more than {cols:,}\n"
            "independent lag columns."
        ).format(lg=lstm_gain, xg=xgb_gain, n=others + 1, cols=multi["flat_columns"])
    elif xgb_gain > lstm_gain + 1.0:
        verdict = (
            "**The tree gains more from the extra series than the recurrence does**\n"
            "({xg:+.1f}% against {lg:+.1f}%). The prediction was that handing a tree\n"
            "{cols:,} flat columns would drown it while the LSTM folded the same\n"
            "channels through one cell. It did not happen on this data, and the\n"
            "prediction should be recorded as failed rather than quietly dropped."
        ).format(xg=xgb_gain, lg=lstm_gain, cols=multi["flat_columns"])
    else:
        verdict = (
            "**Both families gain about the same amount** ({lg:+.1f}% for the LSTM,\n"
            "{xg:+.1f}% for XGBoost). The extra series carry real information and both\n"
            "models use it, but nothing here supports the claim that a recurrence is\n"
            "specifically better at exploiting correlated channels."
        ).format(lg=lstm_gain, xg=xgb_gain)

    return """---
tags: [results, experiment]
---
# 11 Multi-Series Experiment

On a single series, gradient boosting won comfortably - see
[[09 Model Comparison]]. The standard defence of a recurrent model is that it
should pull ahead once **several correlated series share one model**. This
tests that claim directly instead of arguing it.

**Data.** UCI Beijing Multi-Site Air-Quality: PM2.5 at {total} monitoring
stations across one city, hourly, 2013-2017. The stations correlate between
0.78 and 0.97, so the extra channels genuinely carry information about each
other.

**Task.** Predict the next hour of PM2.5 at **{station}**, twice - once from
its own history plus calendar columns, once with the other {others} stations
added as input channels. Same split, same scaler, same windows.

| Setup | Input channels | Flat columns for XGBoost | LSTM MAE | XGBoost MAE |
| --- | --- | --- | --- | --- |
| Own station only | {s_feat} | {s_cols:,} | {s_lstm:,.1f} | {s_xgb:,.1f} |
| All {total} stations | {m_feat} | {m_cols:,} | {m_lstm:,.1f} | {m_xgb:,.1f} |
| **Gain from the extra series** | | | **{lstm_gain:+.1f}%** | **{xgb_gain:+.1f}%** |

{verdict}

## Why this is the right test

Adding the other stations multiplies the tree's input by {ratio:.1f}x -
from {s_cols:,} columns to {m_cols:,} - while the LSTM's input shape only
grows in its feature axis, from `(24, {s_feat})` to `(24, {m_feat})`. The
parameter count of the recurrence barely moves. If the "shared representation"
argument is worth anything, this is where it should show.

Note that MAE here is in micrograms per cubic metre, so it cannot be compared
with the traffic or bike numbers anywhere else in this vault - only the two
rows above can be compared with each other.

Related: [[09 Model Comparison]] - [[10 Window Length Experiment]] - [[06 Limits and Next Steps]]
""".format(total=others + 1, station=station, others=others,
           s_feat=single["n_features"], s_cols=single["flat_columns"],
           s_lstm=single["lstm_mae"], s_xgb=single["xgb_mae"],
           m_feat=multi["n_features"], m_cols=multi["flat_columns"],
           m_lstm=multi["lstm_mae"], m_xgb=multi["xgb_mae"],
           lstm_gain=lstm_gain, xgb_gain=xgb_gain, verdict=verdict,
           ratio=multi["flat_columns"] / max(single["flat_columns"], 1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--station", default=None,
                        help="Which station to predict (default: the first column).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=24)
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(
            "Missing {}\nBuild it first:\n"
            "  python scripts/prepare_multisite_dataset.py".format(args.data))

    every = stations(args.data)
    station = args.station or every[0]
    if station not in every:
        raise SystemExit("Unknown station {!r}. Available: {}".format(
            station, ", ".join(every)))
    others = [s for s in every if s != station]

    common = dict(data_path=args.data, target_column=station, calendar_features=True,
                  sequence_length=args.sequence_length, epochs=args.epochs,
                  patience=args.patience)

    single = one_run(TrainingConfig(run_name="air_single", exogenous_columns=(), **common),
                     "single series")
    multi = one_run(TrainingConfig(run_name="air_multi", exogenous_columns=tuple(others),
                                   **common),
                    "all {} stations".format(len(every)))

    lstm_gain = (single["lstm_mae"] - multi["lstm_mae"]) / single["lstm_mae"] * 100
    xgb_gain = (single["xgb_mae"] - multi["xgb_mae"]) / single["xgb_mae"] * 100

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "station": station, "n_stations": len(every),
        "single": single, "multi": multi,
        "lstm_gain_pct": round(lstm_gain, 2), "xgb_gain_pct": round(xgb_gain, 2),
    }, indent=2), encoding="utf-8")

    vault = Path(VAULT_DIR)
    if vault.exists():
        (vault / "11 Multi-Series Experiment.md").write_text(
            vault_note(single, multi, station, len(others)), encoding="utf-8")
        print("  vault note written")

    print("\n" + "=" * 66)
    print("  Predicting PM2.5 at {}".format(station))
    print("-" * 66)
    print("  {:<22} {:>12} {:>12}".format("", "LSTM", "XGBoost"))
    print("  {:<22} {:>12,.1f} {:>12,.1f}".format(
        "own station only", single["lstm_mae"], single["xgb_mae"]))
    print("  {:<22} {:>12,.1f} {:>12,.1f}".format(
        "all {} stations".format(len(every)), multi["lstm_mae"], multi["xgb_mae"]))
    print("  {:<22} {:>11.1f}% {:>11.1f}%".format("gain", lstm_gain, xgb_gain))
    print("-" * 66)
    print("  written: {}".format(RESULTS))


if __name__ == "__main__":
    main()
