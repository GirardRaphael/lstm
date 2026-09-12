"""Collect every trained run, benchmark each one, and rank them honestly.

Reads whatever is in `models/`, runs the XGBoost benchmark for any run that
does not have one yet, and produces:

    reports/figures/10_model_comparison.png   every model on the same axis
    reports/figures/11_horizons.png           MAE vs forecast horizon
    reports/model_comparison.json             the raw table
    obsidian_vault/.../09 Model Comparison.md the write-up

    python scripts/compare_runs.py
    python scripts/compare_runs.py --skip-benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_lstm.config import MODEL_DIR, REPORT_DIR, VAULT_DIR, TrainingConfig  # noqa: E402
from traffic_lstm.plots import plot_horizon_degradation, plot_model_comparison  # noqa: E402

FRIENDLY = {
    "baseline_univariate": "LSTM - past traffic only",
    "multivariate": "LSTM - traffic + weather + calendar",
    "gap_guarded": "LSTM - gap-guarded windows",
    "multi_horizon": "LSTM - multi-horizon (+1h head)",
    "bike_sharing": "LSTM - past rentals only",
    "bike_sharing_mv": "LSTM - rentals + weather + calendar",
}

# MAE is in the units of whatever is being predicted, so runs on different
# datasets must never share a ranking or a chart. Everything below groups by
# the source file first.
DATASET_LABELS = {
    "Metro_Interstate_Traffic_Volume": "Motorway traffic (UCI Metro Interstate)",
    "bike_sharing_hourly": "Bike rentals (UCI Bike Sharing)",
}
DATASET_UNITS = {
    "Metro_Interstate_Traffic_Volume": "vehicles per hour",
    "bike_sharing_hourly": "rentals per hour",
}


def dataset_key(cfg) -> str:
    return Path(cfg.data_path).stem


def load_runs() -> list:
    runs = []
    for path in sorted(MODEL_DIR.glob("*_artifacts.json")):
        artifacts = json.loads(path.read_text(encoding="utf-8"))
        runs.append((TrainingConfig(**artifacts["config"]), artifacts))
    return runs


def ensure_benchmark(cfg: TrainingConfig, skip: bool) -> dict | None:
    path = MODEL_DIR / "benchmark_{}.json".format(cfg.run_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if skip:
        return None
    from traffic_lstm.benchmark import compare_with_lstm, run_benchmark

    print("  benchmarking {} ...".format(cfg.run_name))
    outcome = run_benchmark(cfg, verbose=False)
    payload = {"xgboost": outcome["result"],
               "comparison": compare_with_lstm(cfg, outcome["result"])}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_table(runs, benchmarks) -> dict:
    rows = []
    naive_seen = {}
    for (cfg, artifacts), benchmark in zip(runs, benchmarks):
        horizon = cfg.horizons[0]
        key = dataset_key(cfg)
        block = artifacts["results"]["h{}".format(horizon)]
        rows.append({
            "label": FRIENDLY.get(cfg.run_name, cfg.run_name),
            "run_name": cfg.run_name,
            "dataset": key,
            "kind": "lstm",
            "mae": block["lstm"]["mae"],
            "rmse": block["lstm"]["rmse"],
            "mape": block["lstm"]["mape"],
            "features": artifacts.get("n_features", 1),
            "train_sequences": artifacts["train_sequences"],
            "seconds": artifacts["training_seconds"],
            "epochs": artifacts["epochs_run"],
        })
        naive_seen[(key, "Naive - same hour yesterday")] = \
            block["naive_same_hour_yesterday"]["mae"]
        naive_seen[(key, "Naive - last hour")] = block["naive_persistence"]["mae"]
        if benchmark:
            xgb = benchmark["xgboost"]
            rows.append({
                "label": "XGBoost - {}".format(
                    FRIENDLY.get(cfg.run_name, cfg.run_name).replace("LSTM - ", "")),
                "run_name": cfg.run_name,
                "dataset": key,
                "kind": "xgboost",
                "mae": xgb["metrics"]["mae"],
                "rmse": xgb["metrics"]["rmse"],
                "mape": xgb["metrics"]["mape"],
                "features": xgb["n_features_flat"],
                "train_sequences": None,
                "seconds": xgb["training_seconds"],
                "epochs": xgb["trees_used"],
            })
    for (key, label), mae in naive_seen.items():
        rows.append({"label": label, "run_name": "-", "dataset": key, "kind": "naive",
                     "mae": mae, "rmse": None, "mape": None, "features": 0,
                     "train_sequences": None, "seconds": 0, "epochs": 0})
    rows.sort(key=lambda r: (r["dataset"], r["mae"]))
    return {"rows": rows}


def group_by_dataset(rows) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: r["mae"])
    return grouped


def horizon_curve(runs) -> tuple:
    for cfg, artifacts in runs:
        if len(cfg.horizons) > 1:
            maes = [artifacts["results"]["h{}".format(h)]["lstm"]["mae"] for h in cfg.horizons]
            return list(cfg.horizons), maes
    return [], []


def _cross_dataset_section(grouped: dict) -> str:
    """Does the LSTM-vs-XGBoost verdict hold across datasets? Usually not."""
    if len(grouped) < 2:
        return ""
    lines = []
    flips = set()
    for key, rows in grouped.items():
        lstms = [r for r in rows if r["kind"] == "lstm"]
        xgbs = [r for r in rows if r["kind"] == "xgboost"]
        if not lstms or not xgbs:
            continue
        best_lstm, best_xgb = min(lstms, key=lambda r: r["mae"]), min(xgbs, key=lambda r: r["mae"])
        winner = "LSTM" if best_lstm["mae"] < best_xgb["mae"] else "XGBoost"
        flips.add(winner)
        margin = abs(best_lstm["mae"] - best_xgb["mae"]) / max(
            best_lstm["mae"], best_xgb["mae"]) * 100
        lines.append("| {ds} | {lstm:,.1f} | {xgb:,.1f} | **{w}** by {m:.1f}% | {u} |".format(
            ds=DATASET_LABELS.get(key, key), lstm=best_lstm["mae"], xgb=best_xgb["mae"],
            w=winner, m=margin, u=DATASET_UNITS.get(key, "units per hour")))

    verdict = (
        "**The verdict flips between the two datasets.** That is the single most\n"
        "useful thing this project found, and it is only visible because the same\n"
        "pipeline was run on a second, unrelated series.\n\n"
        "Anyone who concludes \"gradient boosting beats LSTMs on time series\" from\n"
        "the traffic result alone would be wrong on the bike data, and vice versa.\n"
        "The right conclusion is narrower and more useful: *this* comparison is\n"
        "cheap to run, so run it on your data instead of inheriting someone else's\n"
        "answer."
        if len(flips) > 1 else
        "The same family wins on both datasets, which is weak evidence that the\n"
        "result is about the method rather than about one particular series.")

    return """
## Does the verdict hold on a second dataset?

The same code, the same architecture, the same evaluation - pointed at hourly
**bike rentals in Washington DC** instead of motorway traffic. Only two
command-line arguments changed.

| Dataset | Best LSTM | Best XGBoost | Winner | Unit |
| --- | --- | --- | --- | --- |
{rows}

{verdict}

*MAE is in the units of the target, so the two rows must never be compared to
each other - only within a row.*

""".format(rows="\n".join(lines), verdict=verdict)


def vault_note(grouped: dict, primary: str, horizons, maes) -> str:
    header = ("| Model | Inputs | MAE | RMSE | MAPE | Train time |\n"
              "| --- | --- | --- | --- | --- | --- |\n")
    lines = []
    for row in grouped[primary]:
        lines.append("| {label} | {feat} | **{mae:,.1f}** | {rmse} | {mape} | {secs} |".format(
            label=row["label"], feat=row["features"] or "-", mae=row["mae"],
            rmse="{:,.1f}".format(row["rmse"]) if row["rmse"] else "-",
            mape="{:.1f}%".format(row["mape"]) if row["mape"] else "-",
            secs="{:,.0f}s".format(row["seconds"]) if row["seconds"] else "-"))

    best = grouped[primary][0]
    lstms = [r for r in grouped[primary] if r["kind"] == "lstm"]
    xgbs = [r for r in grouped[primary] if r["kind"] == "xgboost"]
    best_lstm = min(lstms, key=lambda r: r["mae"]) if lstms else None
    best_xgb = min(xgbs, key=lambda r: r["mae"]) if xgbs else None

    verdict = ""
    if best_lstm and best_xgb:
        if best_xgb["mae"] < best_lstm["mae"]:
            gap = (best_lstm["mae"] - best_xgb["mae"]) / best_lstm["mae"] * 100
            speed = best_lstm["seconds"] / max(best_xgb["seconds"], 0.1)
            verdict = (
                "## The uncomfortable result\n\n"
                "> Gradient boosting on the **same tensors** - the same windows, the same\n"
                "> split, the same scaler, just flattened - scores **{:,.0f}** against the\n"
                "> best LSTM's **{:,.0f}**. That is **{:.0f}% more accurate**, trained\n"
                "> **{:.0f}x faster**.\n\n"
                "This is the single most useful thing in the project, and it should be said\n"
                "out loud rather than buried. It does not mean the LSTM was a mistake - it\n"
                "means *this problem* does not need one. One strongly periodic series with a\n"
                "24-hour window is close to the ideal case for a tree ensemble: the useful\n"
                "signal is almost entirely in `t-1h`, `t-2h` and `t-24h`, and a tree can\n"
                "split on those directly without learning a recurrence.\n\n"
                "**When the LSTM would start to win:**\n\n"
                "- dependencies longer than the input window, where lag columns run out\n"
                "- many correlated series (several junctions) sharing one model\n"
                "- irregular or variable-length sequences, which a fixed lag table cannot express\n"
                "- learning a representation to reuse elsewhere, rather than one number\n\n"
                "Answering *\"why an LSTM?\"* with **\"I measured it, and for this dataset it is\n"
                "not the best tool - here is the number\"** is a stronger answer than any\n"
                "amount of theory.\n\n".format(
                    best_xgb["mae"], best_lstm["mae"], gap, speed))
        else:
            gap = (best_xgb["mae"] - best_lstm["mae"]) / best_xgb["mae"] * 100
            verdict = (
                "## The verdict\n\n"
                "> The LSTM beats gradient boosting on identical data by **{:.1f}%**\n"
                "> ({:,.0f} against {:,.0f}). The sequence structure is earning its cost.\n\n"
                .format(gap, best_lstm["mae"], best_xgb["mae"]))

    horizon_section = ""
    if horizons:
        rows = "\n".join("| +{}h | {:,.1f} |".format(h, m) for h, m in zip(horizons, maes))
        horizon_section = (
            "## Forecasting further ahead\n\n"
            "![[11_horizons.png]]\n\n"
            "| Horizon | MAE |\n| --- | --- |\n{}\n\n"
            "One model, three outputs. Accuracy decays as the horizon grows, which is\n"
            "the expected and honest result: the further ahead you look, the less the\n"
            "last 24 hours determine the answer.\n\n".format(rows))

    return """---
tags: [results, comparison]
---
# 09 Model Comparison

Every model tried on the motorway dataset, scored on **the same test windows**.

![[10_model_comparison.png]]

{header}{rows}

The best result on this dataset is **{best_label}** at **{best_mae:,.1f}**.

{verdict}{cross}{horizon_section}## How the comparison is kept fair

The benchmark does not rebuild features. It takes the exact tensors the LSTM
was trained on, shape `(n, timesteps, features)`, and flattens them to
`(n, timesteps * features)`. Same rows, same chronological split, same scaler,
same targets, same validation cut. The only difference is that the tree model
cannot see the ordering of the timesteps except through column position.

Related: [[05 Results]] - [[06 Limits and Next Steps]] - [[08 Exam Questions]]
""".format(header=header, rows="\n".join(lines), best_label=best["label"],
           best_mae=best["mae"], verdict=verdict, horizon_section=horizon_section,
           cross=_cross_dataset_section(grouped))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-benchmarks", action="store_true")
    args = parser.parse_args(argv)

    runs = load_runs()
    if not runs:
        raise SystemExit("No trained runs found in {}".format(MODEL_DIR))
    print("Found {} run(s): {}".format(len(runs), ", ".join(c.run_name for c, _ in runs)))

    benchmarks = [ensure_benchmark(cfg, args.skip_benchmarks) for cfg, _ in runs]
    table = build_table(runs, benchmarks)
    grouped = group_by_dataset(table["rows"])
    horizons, maes = horizon_curve(runs)

    # The primary dataset is whichever the vault was built from.
    primary = ("Metro_Interstate_Traffic_Volume"
               if "Metro_Interstate_Traffic_Volume" in grouped else next(iter(grouped)))

    figure_names = {}
    for key, rows in grouped.items():
        suffix = "" if key == primary else "_" + key.split("_")[0]
        name = "10_model_comparison{}.png".format(suffix)
        plot_model_comparison(
            [(r["label"], r["mae"], r["kind"]) for r in rows], filename=name,
            unit=DATASET_UNITS.get(key, "units per hour"),
            title="{} - same test windows".format(DATASET_LABELS.get(key, key)))
        figure_names[key] = name
    if horizons:
        plot_horizon_degradation(horizons, maes)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "model_comparison.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8")

    vault = Path(VAULT_DIR)
    if vault.exists():
        (vault / "09 Model Comparison.md").write_text(
            vault_note(grouped, primary, horizons, maes), encoding="utf-8")
        figures = vault / "Figures"
        figures.mkdir(parents=True, exist_ok=True)
        import shutil
        for name in list(figure_names.values()) + ["11_horizons.png"]:
            source = REPORT_DIR / "figures" / name
            if source.exists():
                shutil.copy2(source, figures / name)
        print("  vault note: {}".format(vault / "09 Model Comparison.md"))

    for key, rows in grouped.items():
        print("\n{}  (MAE in {})".format(
            DATASET_LABELS.get(key, key), DATASET_UNITS.get(key, "units")))
        print("{:<46} {:>10} {:>10} {:>9}".format("MODEL", "MAE", "RMSE", "TIME"))
        print("-" * 78)
        for row in rows:
            print("{:<46} {:>10,.1f} {:>10} {:>9}".format(
                row["label"][:46], row["mae"],
                "{:,.1f}".format(row["rmse"]) if row["rmse"] else "-",
                "{:,.0f}s".format(row["seconds"]) if row["seconds"] else "-"))


if __name__ == "__main__":
    main()
