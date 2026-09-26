"""Summarize completed rolling-origin artifacts without choosing by test score."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def render():
    lines=["# Rolling-origin and calibration evidence", "",
        "Retrospective evaluation on previously inspected historical datasets. No untouched future/site confirmation is claimed.", "",
        "Three expanding-origin folds × three seeds per dataset. Fixed 20-epoch maximum for the 32/16-unit LSTM, "
        "validation early stopping, direct-lag XGBoost and seasonal baselines. Fit, validation, calibration and "
        "evaluation timestamps are separated. Calibration uses the next block after validation; model selection "
        "uses only validation. All candidates share eligible windows. Counts and calendar only; no weather ablation.", "",
        "The table reports mean MAE across seeds ± sample standard deviation within a fold. These are not confidence "
        "intervals. Daily/3-day/weekly paired bootstrap sensitivity and per-seed metrics are in each results.json. "
        "Identical baseline results repeated across seeds are not independent evidence. Fold boundaries use sorted "
        "observed-row fractions, so wall-clock durations differ. No equal-duration comparison is implied.", ""]
    for path in sorted((ROOT/"reports/backtests").glob("*/results.json")):
        data=json.loads(path.read_text())
        lines += [f"## {data['protocol']['dataset']}", "", f"[Protocol and all runs](backtests/{path.parent.name}/results.json)", "",
                  "| Fold | Candidate | MAE mean ± seed SD | MAPE mean | Bias mean | Peak MAE mean | 90% interval coverage range | Mean interval width |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for fold in sorted({row["fold"] for row in data["runs"]}):
            runs=[row for row in data["runs"] if row["fold"]==fold]
            for name in ("xgboost","lstm","hour_of_week_mean","yesterday"):
                metrics=[row["evaluation"][name]["metrics"] for row in runs]
                maes=[m["mae"] for m in metrics]
                coverage=[m["interval"]["coverage"] for m in metrics]
                avg=lambda key:np.mean([m[key] for m in metrics])
                lines.append(f"| {fold} | {name} | {np.mean(maes):.2f} ± {np.std(maes,ddof=1):.2f} | {avg('mape'):.2f}% | "
                    f"{avg('bias'):.2f} | {avg('peak_hour_mae'):.2f} | {min(coverage):.1%}–{max(coverage):.1%} | "
                    f"{np.mean([m['interval']['mean_width'] for m in metrics]):.2f} |")
        lines += ["", "Validation selections: " + "; ".join(f"fold {r['fold']} / seed {r['seed']}: {r['selected']}" for r in data["runs"]), ""]
        coverages=[r["evaluation"][r["selected"]]["metrics"]["interval"]["coverage"] for r in data["runs"]]
        lines += [f"Selected-candidate observed coverage spans **{min(coverages):.1%}–{max(coverages):.1%}** against nominal 90%.",
                  "Calibration does not establish a coverage guarantee under shift. Review undercoverage and width before any pilot.", ""]
    lines += ["## Remaining external gates", "", "Authorized local observations, certified interval/timezone semantics, untouched future/site tests, "
              "operator acceptance and field interval coverage remain unverified. These artifacts support research review, not street readiness.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    (ROOT/"reports/BACKTEST_REPORT.md").write_text(render(),encoding="utf-8")
