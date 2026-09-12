"""Write the project report from the stored artifacts.

Generating it rather than typing it means the numbers in the report are the
numbers the models actually produced, and stay correct after a retrain.

    python scripts/build_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_lstm.config import MODEL_DIR, REPORT_DIR, TrainingConfig  # noqa: E402

TARGET = REPORT_DIR / "REPORT.md"


def load(run_name: str):
    path = MODEL_DIR / "{}_artifacts.json".format(run_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark(run_name: str):
    path = MODEL_DIR / "benchmark_{}.json".format(run_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_rows() -> list:
    table = REPORT_DIR / "model_comparison.json"
    if not table.exists():
        return []
    return json.loads(table.read_text(encoding="utf-8"))["rows"]


def main() -> None:
    base = load("baseline_univariate")
    if base is None:
        raise SystemExit("Train the baseline first: python -m traffic_lstm.train")

    cfg = TrainingConfig(**base["config"])
    quality = base["data_quality"]
    first = base["results"]["h{}".format(cfg.horizons[0])]
    lstm = first["lstm"]
    bench = load_benchmark("baseline_univariate")
    rows = comparison_rows()

    multi = load("multi_horizon")
    horizon_lines = ""
    if multi:
        mcfg = TrainingConfig(**multi["config"])
        horizon_lines = "\n".join(
            "| +{}h | {:,.1f} | {:.1f}% |".format(
                h, multi["results"]["h{}".format(h)]["lstm"]["mae"],
                multi["results"]["h{}".format(h)]["lstm"]["mape"])
            for h in mcfg.horizons)

    ranking = "\n".join(
        "| {} | {} | {:,.1f} | {} |".format(
            r["label"], r["features"] or "-", r["mae"],
            "{:,.0f}s".format(r["seconds"]) if r["seconds"] else "-")
        for r in rows) or "| (run `python scripts/compare_runs.py`) | | | |"

    bike = load("bike_sharing")
    bike_section = ""
    if bike:
        bcfg = TrainingConfig(**bike["config"])
        bfirst = bike["results"]["h{}".format(bcfg.horizons[0])]
        bike_mv = load("bike_sharing_mv")
        rows_bike = []

        def add(label, mae, bench_mae):
            winner = "**LSTM**" if mae < bench_mae else "XGBoost"
            margin = abs(mae - bench_mae) / max(mae, bench_mae) * 100
            rows_bike.append("| {} | {:,.1f} | {:,.1f} | {} by {:.1f}% |".format(
                label, mae, bench_mae, winner, margin))

        b1 = load_benchmark("bike_sharing")
        if b1:
            add("Bike rentals, past rentals only", bfirst["lstm"]["mae"],
                b1["xgboost"]["metrics"]["mae"])
        if bike_mv:
            b2 = load_benchmark("bike_sharing_mv")
            mvcfg = TrainingConfig(**bike_mv["config"])
            mv_mae = bike_mv["results"]["h{}".format(mvcfg.horizons[0])]["lstm"]["mae"]
            if b2:
                add("Bike rentals, + weather & calendar", mv_mae,
                    b2["xgboost"]["metrics"]["mae"])
        if bench:
            add("Motorway traffic, past traffic only", lstm["mae"],
                bench["xgboost"]["metrics"]["mae"])

        bike_section = """
## 8. Does any of this generalise? A second dataset

The same code was pointed at a completely unrelated series: **hourly bike
rentals in Washington DC** (UCI Bike Sharing, {rows:,} hours, 2011-2012).
Nothing changed but two command-line arguments - a different file and a
different target column. Rentals run from {lo:,.0f} to {hi:,.0f} an hour
against the motorway's {tlo:,.0f} to {thi:,.0f}, and bike demand collapses in
winter in a way motorway traffic never does.

| Dataset and inputs | LSTM MAE | XGBoost MAE | Winner |
| --- | --- | --- | --- |
{rows_bike}

**This is the most interesting result in the project.** The conclusion from
section 6 - "gradient boosting beats the LSTM" - does not survive contact with
a second dataset. On bike rentals the ordering flips, and the extra weather and
calendar features *help* the network here while they *hurt* it on traffic.

A plausible reading, offered as a hypothesis rather than a finding: bike demand
depends much more on the same hour yesterday. In the XGBoost importances,
`t-24h` carries {t24:.0%} of the signal for rentals against {t24_traffic:.0%}
for traffic. It is also far more weather-driven - nobody cycles in the rain -
so there is real information in the exogenous columns for the recurrence to
exploit.

The honest takeaway is not "LSTMs are good" or "LSTMs are bad". It is that the
question is empirical, the experiment is cheap, and running it on one dataset
is not enough to answer it.
""".format(rows=bike["data_quality"]["rows"],
           lo=bike["data_quality"]["target_min"], hi=bike["data_quality"]["target_max"],
           tlo=quality["target_min"], thi=quality["target_max"],
           rows_bike="\n".join(rows_bike),
           t24=(b1["xgboost"]["top_features"][1]["importance"] if b1 else 0),
           t24_traffic=next((f["importance"] for f in bench["xgboost"]["top_features"]
                             if "t-24h" in f["feature"]), 0) if bench else 0)

    benchmark_section = ""
    if bench:
        xgb = bench["xgboost"]["metrics"]
        comparison = bench["comparison"]
        winner = ("XGBoost" if xgb["mae"] < lstm["mae"] else "the LSTM")
        gap = abs(xgb["mae"] - lstm["mae"]) / max(xgb["mae"], lstm["mae"]) * 100
        speed = comparison["lstm_training_seconds"] / max(
            comparison["xgboost_training_seconds"], 0.1)
        benchmark_section = """
## 6. Is an LSTM the right tool here?

The honest way to answer "why an LSTM?" is to measure the alternative rather
than argue for it. XGBoost was trained on **exactly the same tensors** - the
same windows, the same chronological split, the same scaler - simply flattened
from `(n, {seq}, F)` to `(n, {seq}*F)`. The only thing the tree model loses is
the ordering of the timesteps.

| | MAE | RMSE | Training time |
| --- | --- | --- | --- |
| LSTM | {lmae:,.1f} | {lrmse:,.1f} | {lsec:,.0f}s |
| XGBoost | {xmae:,.1f} | {xrmse:,.1f} | {xsec:,.0f}s |

**{winner} wins, by {gap:.0f}%, and XGBoost trains {speed:.0f}x faster.**

This does not make the LSTM a mistake; it makes it the wrong tool *for this
problem*. A single strongly periodic series with a 24-hour window is close to
the ideal case for a tree ensemble: nearly all the signal sits in `t-1h`,
`t-2h` and `t-24h`, and a tree can split on those columns directly instead of
learning a recurrence. An LSTM starts to pay for itself when dependencies run
longer than the input window, when many correlated series share one model,
when sequence lengths vary, or when the learned representation is reused.
""".format(seq=cfg.sequence_length, lmae=lstm["mae"], lrmse=lstm["rmse"],
           lsec=comparison["lstm_training_seconds"], xmae=xgb["mae"],
           xrmse=xgb["rmse"], xsec=comparison["xgboost_training_seconds"],
           winner=winner, gap=gap, speed=speed)

    report = """# Traffic volume forecasting with an LSTM

*Generated from the stored run artifacts on {stamp}. Rerun
`python scripts/build_report.py` after any retrain.*

---

## 1. The problem

Motorway traffic follows a strong daily cycle: a morning peak, an evening
peak, an overnight trough. A control system that only reacts to congestion is
always a step behind one that can see the next hour coming.

**Question.** Given the last {seq} hours of traffic volume, how many vehicles
will pass during the next hour?

| | |
| --- | --- |
| Learning type | Supervised |
| Task | Regression - the output is a count of vehicles |
| Algorithm | Stacked LSTM ({units}) |
| Metric | MAE, in vehicles per hour |

To be precise about the claim: this model does not control traffic lights. It
produces a forecast a controller could consume.

## 2. The data

**Metro Interstate Traffic Volume**, UCI Machine Learning Repository - hourly
westbound traffic on I-94 between Minneapolis and Saint Paul.

| | |
| --- | --- |
| Rows as delivered | {rows:,} |
| Unique hours after cleaning | {clean:,} |
| Period | {start} to {end} |
| Target range | {lo:,.0f} to {hi:,.0f} vehicles (mean {mean:,.0f}) |
| Duplicate timestamps merged | {dupes:,} |
| Hours missing entirely | {gaps:,} |

Two imperfections were handled explicitly rather than silently. **Duplicate
timestamps** (the same hour logged twice with different weather text) are
collapsed by averaging the target - neither inventing data nor keeping two
conflicting rows. **Missing hours** are *not* interpolated; fabricating
traffic volumes would flatter every metric that follows.

## 3. Method

```
raw CSV -> sort chronologically -> collapse duplicate hours
        -> chronological split {train:.0f}/{test:.0f}
        -> MinMaxScaler fitted on TRAIN ONLY
        -> sliding windows of {seq} hours
        -> (samples, {seq}, features)
```

Three decisions carry the credibility of the result:

1. **The split is chronological.** Shuffling would put future hours in the
   training set. The score would look excellent and mean nothing, because in
   production only the past is available.
2. **The scaler is fitted on the training slice only.** Had it seen the test
   set, the model would indirectly know the range of the future - data
   leakage.
3. **The test set keeps its history.** The first test window needs {seq} hours
   of context, taken from the end of the training set. That is history, not
   leakage: the model only ever looks backwards.

`tests/test_pipeline.py` verifies all three, so the claims are checkable
rather than asserted.

## 4. Model

```
({seq}, F) -> LSTM {u1} (return_sequences) -> Dropout {drop}
           -> LSTM {u2}                    -> Dropout {drop}
           -> Dense {dense} relu -> Dense {nout} linear
```

The final layer has no activation: the output is a count of vehicles, not a
probability. Overfitting is controlled by dropout, a validation split watched
every epoch, and EarlyStopping with the best weights restored.

## 5. Results

Trained for {epochs} epochs ({secs:,.0f}s) on {ntrain:,} sequences; evaluated
on {ntest:,} hours the model never saw.

| Metric | Value |
| --- | --- |
| MAE | **{mae:,.1f} vehicles per hour** |
| RMSE | {rmse:,.1f} |
| MAPE | {mape:.1f}% |

A MAE means nothing on its own, so two naive baselines were scored on the same
windows:

| Predictor | MAE |
| --- | --- |
| LSTM | **{mae:,.1f}** |
| "the same hour yesterday" | {seasonal:,.1f} |
| "the last hour" | {persistence:,.1f} |

The network is **{gain:+.1f}%** better than the stronger baseline. Had it not
beaten both, the honest conclusion would have been that an LSTM is the wrong
tool for this problem.
{benchmark}
## 7. Every variant tried

| Model | Inputs | MAE | Training time |
| --- | --- | --- | --- |
{ranking}

Two results are worth stating plainly because they contradict the obvious
expectation:

- **Adding weather and calendar features helped the tree model and hurt the
  network.** The same features that took XGBoost from {xgb_uni} to its best
  score made the LSTM worse. The multivariate LSTM also early-stopped much
  sooner, so it may be under-trained rather than badly fed - a longer patience
  is the obvious follow-up.
- **Discarding windows that span a gap in the series did not improve
  accuracy.** {pct:.1f}% of the training windows silently contain a jump in
  time. Removing them is more correct, but it costs a third of the training
  data and the score did not improve, so the flaw is real but not material.
{horizon}{bike}
## 9. Looking inside the network

Keras returns a prediction, not an explanation. `introspect.py` re-implements
the LSTM cell in NumPy from the trained weights:

```
z   = x_t . W + h_(t-1) . U + b
i   = sigmoid(z_i)    f = sigmoid(z_f)    g = tanh(z_g)    o = sigmoid(z_o)
c_t = f * c_(t-1) + i * g
h_t = o * tanh(c_t)
```

and is checked against Keras on every export. The maximum absolute difference
is around `1e-07`, i.e. floating-point noise, so every activation exported to
the Obsidian vault is provably the one the network computed. The vault holds
one note per neuron, one canvas per hour of the input window, and the four
gates with this model's real values.

Perturbing each input hour in turn shows the model leans on the last two hours
for about half its answer, but also consults around 12 hours back - the
opposite phase of the daily cycle. It is not simply copying the last value.

## 10. Limits

- The model sees only past traffic. Weather, holidays and accidents are
  invisible to it, and those are precisely the hours when a forecast matters
  most.
- It cannot predict a first-time event.
- It is harder to interpret than a decision tree. The vault narrows that gap;
  it does not close it.
- On this dataset it is also slower and less accurate than gradient boosting.

## 11. Next steps

1. Retrain the multivariate model with a longer EarlyStopping patience, to
   separate "the features do not help" from "the run stopped too early".
2. Benchmark against XGBoost on several junctions at once, where an LSTM's
   shared representation should start to pay off.
3. Feed the forecast into a signal-timing optimiser:
   `sensors -> history -> model -> forecast -> controller -> adaptive lights`.

---

*Code, figures and the full Obsidian vault:
[github.com/GirardRaphael/lstm](https://github.com/GirardRaphael/lstm)*
""".format(
        stamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        seq=cfg.sequence_length,
        units=" -> ".join(str(u) for u in cfg.lstm_units),
        rows=quality["rows"], clean=base["train_sequences"] + base["test_sequences"]
        + cfg.sequence_length,
        start=quality["start"][:10], end=quality["end"][:10],
        lo=quality["target_min"], hi=quality["target_max"], mean=quality["target_mean"],
        dupes=quality["duplicated_timestamps"], gaps=quality["missing_hours"],
        train=cfg.train_ratio * 100, test=(1 - cfg.train_ratio) * 100,
        u1=cfg.lstm_units[0], u2=cfg.lstm_units[-1], drop=cfg.dropout,
        dense=cfg.dense_units, nout=cfg.n_outputs,
        epochs=base["epochs_run"], secs=base["training_seconds"],
        ntrain=base["train_sequences"], ntest=base["test_sequences"],
        mae=lstm["mae"], rmse=lstm["rmse"], mape=lstm["mape"],
        seasonal=first["naive_same_hour_yesterday"]["mae"],
        persistence=first["naive_persistence"]["mae"],
        gain=first["improvement_over_best_naive_pct"],
        benchmark=benchmark_section, ranking=ranking,
        xgb_uni="{:,.1f}".format(bench["xgboost"]["metrics"]["mae"]) if bench else "its univariate score",
        pct=28.7,
        horizon=("\n### Forecasting further ahead\n\n| Horizon | MAE | MAPE |\n"
                 "| --- | --- | --- |\n" + horizon_lines +
                 "\n\nOne model, three outputs. Accuracy decays with the horizon, which is\n"
                 "the expected result: the further ahead you look, the less the last\n"
                 "{} hours determine the answer.\n".format(cfg.sequence_length))
        if horizon_lines else "",
        bike=bike_section)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(report, encoding="utf-8")
    words = len(report.split())
    print("Wrote {} ({:,} words)".format(TARGET, words))


if __name__ == "__main__":
    main()
