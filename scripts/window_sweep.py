"""Does a longer input window favour the recurrence? Measure it.

Every comparison in this project so far used a 24-hour window, which is close
to the best case for a tree ensemble: 24 lag columns, nearly all the signal in
two or three of them. The standard argument for an LSTM is that it should pull
ahead when the window grows, because a tree gets one column per
(timestep x feature) and drowns in mostly redundant ones, while a recurrence
folds the whole window through the same cell.

That is a testable claim, so this tests it. For each window length both models
are trained on **the same bundle** - same rows, same chronological split, same
scaler - and the only thing that changes is how many hours they are handed.

    python scripts/window_sweep.py
    python scripts/window_sweep.py --lengths 12 24 48 --epochs 40

Runtime is dominated by the LSTM and grows roughly linearly with the window,
so the default sweep takes around ninety minutes on a CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_lstm.benchmark import run_benchmark  # noqa: E402
from traffic_lstm.config import MODEL_DIR, REPORT_DIR, VAULT_DIR, TrainingConfig  # noqa: E402
from traffic_lstm.data import build_datasets  # noqa: E402
from traffic_lstm.plots import plot_window_sweep  # noqa: E402
from traffic_lstm.train import train  # noqa: E402

RESULTS = REPORT_DIR / "window_sweep.json"

# The best-performing feature set from the ablation: past traffic plus the
# calendar columns, and no weather.
CALENDAR = True
EXOGENOUS: tuple = ()


def sweep(lengths, epochs: int, patience: int, keep_models: bool) -> list:
    rows = []
    for length in lengths:
        cfg = TrainingConfig(
            sequence_length=length,
            calendar_features=CALENDAR,
            exogenous_columns=EXOGENOUS,
            epochs=epochs,
            patience=patience,
            run_name="window_{:03d}".format(length),
        )
        print("\n" + "=" * 66)
        print("  WINDOW {} HOURS".format(length))
        print("=" * 66)

        started = time.time()
        bundle = build_datasets(cfg)
        outcome = train(cfg, bundle=bundle, verbose=0)
        lstm_seconds = time.time() - started

        artifacts = outcome["artifacts"]
        block = artifacts["results"]["h1"]

        # Same bundle, so the two models score identical windows.
        bench = run_benchmark(cfg, bundle=bundle, verbose=False)["result"]

        row = {
            "sequence_length": length,
            "n_features": artifacts["n_features"],
            "flat_columns": bench["n_features_flat"],
            "train_sequences": artifacts["train_sequences"],
            "test_sequences": artifacts["test_sequences"],
            "lstm_mae": block["lstm"]["mae"],
            "lstm_rmse": block["lstm"]["rmse"],
            "lstm_seconds": round(lstm_seconds, 1),
            "lstm_epochs": artifacts["epochs_run"],
            "lstm_diagnosis": artifacts["diagnosis"]["verdict"],
            "xgb_mae": bench["metrics"]["mae"],
            "xgb_rmse": bench["metrics"]["rmse"],
            "xgb_seconds": bench["training_seconds"],
            "naive_persistence": block["naive_persistence"]["mae"],
            "naive_seasonal": block["naive_same_hour_yesterday"]["mae"],
        }
        row["gap_pct"] = round(
            (row["lstm_mae"] - row["xgb_mae"]) / row["xgb_mae"] * 100, 2)
        rows.append(row)

        print("  LSTM    MAE {:7.1f}  ({} epochs, {:,.0f}s, {})".format(
            row["lstm_mae"], row["lstm_epochs"], row["lstm_seconds"],
            row["lstm_diagnosis"]))
        print("  XGBoost MAE {:7.1f}  ({:,} flat columns, {:,.0f}s)".format(
            row["xgb_mae"], row["flat_columns"], row["xgb_seconds"]))
        print("  gap: {:+.1f}% (negative means the LSTM is ahead)".format(row["gap_pct"]))

        if not keep_models:
            # A sweep produces one .keras per length and none of them is the
            # model anyone will use; keep the artifacts, drop the weights.
            cfg.model_path.unlink(missing_ok=True)

    return rows


def summarise(rows) -> dict:
    gaps = [r["gap_pct"] for r in rows]
    crossed = any(g < 0 for g in gaps)
    narrowing = len(gaps) > 1 and gaps[-1] < gaps[0]
    return {
        "crossed": crossed,
        "narrowing": narrowing,
        "gap_at_shortest": gaps[0],
        "gap_at_longest": gaps[-1],
        "best_lstm": min(rows, key=lambda r: r["lstm_mae"]),
        "best_xgb": min(rows, key=lambda r: r["xgb_mae"]),
    }


def vault_note(rows, summary) -> str:
    table = "\n".join(
        "| {h}h | {cols:,} | {lstm:,.1f} | {xgb:,.1f} | **{gap:+.1f}%** | {ls:,.0f}s / {xs:,.0f}s |".format(
            h=r["sequence_length"], cols=r["flat_columns"], lstm=r["lstm_mae"],
            xgb=r["xgb_mae"], gap=r["gap_pct"], ls=r["lstm_seconds"], xs=r["xgb_seconds"])
        for r in rows)

    if summary["crossed"]:
        verdict = (
            "**The curves cross.** At the longest window the LSTM is ahead, which is\n"
            "the first direct evidence in this project that the recurrence buys\n"
            "something a lag table cannot. The advantage is not about traffic being\n"
            "\"sequential\" in the abstract - it is about what happens to a tree model\n"
            "when you hand it {cols:,} columns, most of them redundant.".format(
                cols=rows[-1]["flat_columns"]))
    elif summary["narrowing"]:
        verdict = (
            "**The gap narrows but never closes.** The LSTM gains ground as the window\n"
            "grows - from {a:+.1f}% to {b:+.1f}% - which is the direction the theory\n"
            "predicts, but on this series it does not get far enough to win. Extrapolating\n"
            "past the longest window tested would be guessing; the honest statement is\n"
            "that the trend exists and the crossover was not reached here.".format(
                a=summary["gap_at_shortest"], b=summary["gap_at_longest"]))
    else:
        verdict = (
            "**The gap does not narrow.** Lengthening the window does not help the\n"
            "recurrence on this series, so the usual argument - \"an LSTM compresses a\n"
            "long history where a tree drowns in columns\" - simply does not apply here.\n"
            "That is worth stating plainly rather than quietly dropping: a prediction\n"
            "was made, it was tested, and it failed.")

    return """---
tags: [results, experiment]
---
# 10 Window Length Experiment

Every other comparison in this vault uses a 24-hour window. That is close to
the best case for a tree ensemble: 24 lag columns with nearly all the signal in
two or three of them. The standard argument for a recurrent model is that it
should pull ahead once the window grows, because a tree receives one column per
timestep and feature while an LSTM folds the same window through one cell.

This is that experiment. Both models see **the same bundle** at every length -
same rows, same split, same scaler - and the only thing that changes is how
many hours they are handed.

| Window | Flat columns for XGBoost | LSTM MAE | XGBoost MAE | Gap | Train time |
| --- | --- | --- | --- | --- | --- |
{table}

Gap is `(LSTM - XGBoost) / XGBoost`; negative means the LSTM is ahead.

![[12_window_sweep.png]]

{verdict}

## What this does and does not test

A longer window on this dataset mostly adds **more daily cycles**, not longer
dependencies - the structure in motorway traffic is daily, so hours 25 to 96
largely repeat information hours 1 to 24 already carried. So this measures
whether the recurrence handles a long, largely redundant window better than a
lag table does. It does **not** test genuinely long-range dependencies, which
would need a series that actually has them.

The stronger test - several correlated series sharing one model - is still
unrun. See [[06 Limits and Next Steps]].

Related: [[09 Model Comparison]] - [[05 Results]]
""".format(table=table, verdict=verdict)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[12, 24, 48, 96])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--keep-models", action="store_true",
                        help="Keep every .keras file instead of only the artifacts.")
    args = parser.parse_args(argv)

    lengths = sorted(set(args.lengths))
    rows = sweep(lengths, args.epochs, args.patience, args.keep_models)
    summary = summarise(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"rows": rows, "summary": {
        k: v for k, v in summary.items() if k not in {"best_lstm", "best_xgb"}}},
        indent=2), encoding="utf-8")

    plot_window_sweep(lengths, [r["lstm_mae"] for r in rows],
                      [r["xgb_mae"] for r in rows])

    vault = Path(VAULT_DIR)
    if vault.exists():
        (vault / "10 Window Length Experiment.md").write_text(
            vault_note(rows, summary), encoding="utf-8")
        import shutil
        source = REPORT_DIR / "figures" / "12_window_sweep.png"
        if source.exists():
            shutil.copy2(source, vault / "Figures" / source.name)

    print("\n" + "=" * 66)
    print("  {:<10} {:>10} {:>10} {:>9}".format("WINDOW", "LSTM", "XGBOOST", "GAP"))
    print("-" * 66)
    for row in rows:
        print("  {:<10} {:>10,.1f} {:>10,.1f} {:>8.1f}%".format(
            "{}h".format(row["sequence_length"]), row["lstm_mae"], row["xgb_mae"],
            row["gap_pct"]))
    print("-" * 66)
    print("  crossed: {}   narrowing: {}   {:+.1f}% -> {:+.1f}%".format(
        summary["crossed"], summary["narrowing"],
        summary["gap_at_shortest"], summary["gap_at_longest"]))
    print("  written: {}".format(RESULTS))


if __name__ == "__main__":
    main()
