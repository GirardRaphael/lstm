"""Generate the current report exclusively from verified v2 manifests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traffic_lstm.pipeline_v2 import load_package


def render():
    lines = ["# Hourly forecast evidence", "",
             "Status: research workbench, not a deployed traffic product. No signal control.", "",
             "These are retrospective reruns on historically inspected data. Model selection uses validation MAE, "
             "including baselines, before test scoring. This does not make the old test period untouched.", "",
             "One seed per run. MAE intervals resample observed calendar-day blocks (500 replicates). "
             "They do not measure training-seed variability or provide prediction intervals. "
             "Peak hours are weekdays 07–09 and 16–18 inclusive in the dataset clock. "
             "MAPE excludes actuals <= 1; its sample count is shown. Bias is prediction minus actual.", ""]
    for path in sorted((ROOT / "models/v2").glob("*/manifest.json")):
        m = load_package(path.parent).manifest
        lines += [f"## {m['run_name']}", "", f"Source: [{path.parent.name} manifest](../models/v2/{path.parent.name}/manifest.json).",
                  f"Validation-selected candidate: **{m.get('selected_on_validation')}**. Units: {m['units']}.",
                  f"LSTM epochs: {m['training']['epochs_run']}/{m['config']['epochs']} maximum. "
                  "Fixed-budget candidates; no claim of exhaustive tuning or the best attainable LSTM.",
                  f"XGBoost layout: `{m['config'].get('xgb_layout', 'flattened')}`. "
                  "The direct tree uses longer seasonal lags and the target clock; this is a practical candidate comparison, not an architecture-only ablation.",
                  f"Eligible windows: `{m['coverage']['eligible']}`. Excluded: `{m['coverage']['excluded']}`.", ""]
        for horizon, block in m["evaluation"]["test"].items():
            lines += [f"Test {horizon}:", "", "| Candidate | MAE [95% CI] | RMSE | MAPE | MAPE n | Bias | Peak MAE | n |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for name in ("xgboost", "lstm", "hour_of_week_mean", "naive_seasonal", "naive_persistence"):
                values = block[name]
                ci = values.get("mae_ci")
                interval = f" [{ci['low']:.2f}, {ci['high']:.2f}]" if ci else ""
                def number(key):
                    v = values.get(key)
                    return "undefined" if v is None else f"{v:.2f}"
                lines.append(f"| {name} | {number('mae')}{interval} | {number('rmse')} | {number('mape')}% | "
                             f"{values.get('mape_n')} | {number('bias')} | {number('peak_hour_mae')} | {values['n']} |")
            paired = block.get("paired_xgboost_minus_lstm_mae")
            lines += ["", f"Paired XGBoost minus LSTM absolute-error difference: `{json.dumps(paired)}`. Negative favors XGBoost.", ""]
    lines += ["## Delivery status", "",
              "Rolling-origin, three-seed evaluation and separate interval calibration are now recorded in [BACKTEST_REPORT.md](BACKTEST_REPORT.md). "
              "A persistent local workflow with access tokens, isolated workers, forecasts/actuals, model activation, monitoring gates and backup/restore is implemented. "
              "See [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) and [operations](../docs/OPERATIONS.md).", "",
              "Remaining gates:", "",
              "- Untouched future/site holdouts and field interval calibration; historical reanalysis is not confirmation.",
              "- Resolve observed bike interval undercoverage under shift before relying on those intervals.",
              "- Authorized local observations, interval semantics, sensor provenance and operator utility validation.",
              "- Managed identity/TLS, deployment isolation, encrypted secrets/backups and alert delivery before shared hosting.",
              "- Legacy educational UI paths remain v1 and cannot substantiate product accuracy.", "",
              "Historical v1 evidence is in [LEGACY_REPORT.md](LEGACY_REPORT.md). The aborted two-epoch window sweep is not model-selection evidence.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    (ROOT / "reports/REPORT.md").write_text(render(), encoding="utf-8")
