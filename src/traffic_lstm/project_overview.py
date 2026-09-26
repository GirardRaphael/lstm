"""Public project briefing, sourced from committed research evidence only.

Never read the operational workspace here: this briefing is available before login.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .config import PROJECT_ROOT


@dataclass
class Slide:
    title: str
    takeaway: str
    body: str
    facts: list[tuple[str, str, str]] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    evidence: str = ""
    sources: list[str] = field(default_factory=list)


def _read(root, relative):
    try:
        return json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _comparison(root, dataset, title, unit):
    source = f"models/v2/{dataset}_v2_20260925/manifest.json"
    manifest = _read(root, source)
    if manifest is None:
        return Slide(title, "Result evidence is unavailable.",
                     "The saved run could not be read. No substitute accuracy figures are displayed.", sources=[source])
    try:
        block = manifest["evaluation"]["test"]["h1"]
        tree, lstm = block["xgboost"], block["lstm"]
        paired = block["paired_xgboost_minus_lstm_mae"]
        ci = paired["ci"]
        if ci is None:
            conclusion = "The point estimates alone do not establish a reliable ordering."
        elif ci["high"] < 0:
            conclusion = "The paired MAE interval favors XGBoost in this recorded experiment."
        elif ci["low"] > 0:
            conclusion = "The paired MAE interval favors the LSTM in this recorded experiment."
        else:
            conclusion = "The paired MAE interval crosses zero: no clear MAE advantage is established."
        rows = []
        for key, label in [("xgboost", "XGBoost"), ("lstm", "LSTM"),
                           ("hour_of_week_mean", "Hour-of-week mean"),
                           ("naive_seasonal", "Timestamp-true yesterday"),
                           ("naive_persistence", "Last observed hour")]:
            m = block[key]
            rows.append({"Candidate": label, "MAE": round(m["mae"], 2),
                         "RMSE": round(m["rmse"], 2),
                         "MAPE (%)": round(m["mape"], 2) if m["mape"] is not None else None,
                         "Bias": round(m["bias"], 2), "Peak MAE": round(m["peak_hour_mae"], 2)})
        interval = "unavailable" if ci is None else f"[{ci['low']:.2f}, {ci['high']:.2f}]"
        cfg = manifest["config"]
        return Slide(
            title, conclusion,
            f"**{tree['n']:,} matched test hours; errors in {unit}.** "
            f"Validation selected **{manifest['selected_on_validation']}** before test scoring.\n\n"
            "MAE is the average absolute miss; lower is better. RMSE gives more weight to large misses.\n\n"
            "Compare candidates within this experiment. These results do not establish performance on a new city, "
            "sensor or future period. On bikes, also inspect RMSE and peak errors: the lowest MAE need not minimize every operational cost.",
            facts=[("XGBoost MAE", f"{tree['mae']:.2f}", unit),
                   ("LSTM MAE", f"{lstm['mae']:.2f}", unit)], rows=rows,
            evidence=f"Single chronological split; {cfg['sequence_length']}-hour sequence; seed {manifest['seed']}; "
                     f"LSTM {manifest['training']['epochs_run']}/{cfg['epochs']} epochs. "
                     f"Paired XGBoost-minus-LSTM MAE difference: {paired['mean']:.2f}, 95% daily-block bootstrap interval {interval}. "
                     "Negative favors XGBoost. This interval measures sampling uncertainty conditional on one fit, not training-seed variability. "
                     "MAE/RMSE/bias/peak MAE use physical count units; MAPE is percent on actuals > 1. "
                     "Bias is prediction minus actual. Peak hours are weekdays 07–09 and 16–18 in the dataset clock. "
                     "LSTM and direct-tree information sets differ. Historically inspected test data make this a retrospective reanalysis.",
            sources=[source, "reports/REPORT.md"])
    except (KeyError, TypeError, ValueError):
        return Slide(title, "Result evidence is incomplete.",
                     "The saved run does not contain the metrics required for this comparison. No replacement numbers are shown.", sources=[source])


def build_slides(root: Path = PROJECT_ROOT):
    slides = [
        Slide("The project in one minute", "A local observatory for hourly vehicle counts and forecast accountability.",
              "The operator can inspect observations, queue training, select a model version, record a next-hour forecast, "
              "and compare it with the count observed later. The central question is: **is this forecast useful and supported by evidence?**\n\n"
              "The delivered software runs locally. Traffic analytics are read-only: forecasts do not actuate lights. "
              "A vehicle count is not a measurement of congestion, speed or queue length.",
              facts=[("Forecast horizon", "Next hour", "Operational hourly-count model"),
                     ("Delivery stage", "Local research", "Field and shared-hosting validation remain open")],
              sources=["README.md", "ROAD_PRODUCT_PLAN.md"]),
        Slide("What the operator actually does", "Follow each forecast from its source observations to its measured error.",
              "1. **Import:** validate the stream, timestamps, cadence and counts; retain a fixed snapshot.\n"
              "2. **Train:** queue a persistent job; a separate worker fits and evaluates candidates.\n"
              "3. **Review:** inspect validation scores, calibration and evaluation diagnostics.\n"
              "4. **Activate:** an administrator selects a version that passes the local release gate.\n"
              "5. **Forecast:** save the prediction, interval, timestamp, source and model version.\n"
              "6. **Measure:** import subsequent observations to calculate errors and interval coverage.",
              evidence="Jobs, models, forecasts, actuals and audit events persist in SQLite. A failed or unavailable forecast is explicit; "
                       "it is never replaced with an invented observation. Training does not run inside a UI request.",
              sources=["docs/OPERATIONS.md", "reports/observatory_acceptance.json"]),
        Slide("What the data mean", "Historical benchmarks and operational observations have different contracts.",
              "**Motorway research:** hourly westbound I-94 vehicle counts from the bundled UCI dataset.\n\n"
              "**Bike research:** hourly Washington DC rentals, used as a separate forecasting benchmark—not vehicle observations.\n\n"
              "**Operational imports:** one stream and cadence; completed-interval timestamps with explicit UTC offsets; finite nonnegative integer vehicle counts. "
              "Missing intervals stay missing. The current operational model supports hourly counts only.",
              evidence="Bundled motorway replay assigns UTC to the original clock labels solely to exercise the workflow. "
                       "It does not certify the source timezone or interval-start/end convention. A source/provenance declaration does not authenticate a sensor. "
                       "Conflicting duplicates, off-grid or future timestamps, and mixed stream/cadence inputs are refused.",
              sources=["docs/OPERATIONS.md", "src/traffic_lstm/observations.py"]),
        Slide("Why compare trees, sequences and simple rules?", "Choose the model from evidence; the LSTM is a challenger, not a requirement.",
              "**Hour-of-week mean:** learn the average count for each weekly time slot using fit data only. "
              "It measures how much can be explained by the weekly clock.\n\n"
              "**XGBoost:** predict from counts at t−1, t−24 and t−168 plus the target hour/day-of-week clock. "
              "Unavailable historical lags remain missing.\n\n"
              "**LSTM:** learn a sequential representation from a 24-hour history; inspect its gates and hidden states in the educational workbench.\n\n"
              "Research also scores persistence and timestamp-true yesterday. The operational worker currently selects between the tree and weekly mean.",
              evidence="Direct XGBoost has longer seasonal lags and the target clock, while the LSTM uses sequence inputs. "
                       "This is a practical candidate comparison, not an architecture-only ablation. Fixed-budget runs are not claims of optimal tuning.",
              sources=["src/traffic_lstm/candidate.py", "src/traffic_lstm/backtest.py"]),
        Slide("How leakage is prevented", "Fit, selection, calibration and evaluation have distinct jobs.",
              "**Fit:** learn model parameters and transformations from past fitting rows only.\n\n"
              "**Validation:** select candidates and early-stopping points without ranking on test error.\n\n"
              "**Calibration:** estimate interval width on a later, separate block.\n\n"
              "**Evaluation:** measure accuracy, bias, peak errors, coverage and width after those decisions.\n\n"
              "Incomplete or nonconsecutive sequence windows are excluded. Multi-horizon windows crossing partition boundaries are purged.",
              rows=[{"Experiment": "Original v2 comparison", "Protocol": "64% fit / 16% validation / 20% test; one seed; no forecast interval calibration"},
                    {"Experiment": "Operational candidate", "Protocol": "64% fit / 16% validation / 10% calibration / 10% evaluation"},
                    {"Experiment": "Rolling-origin research", "Protocol": "3 expanding folds × 3 seeds; separate calibration and evaluation in each fold"}],
              evidence="Fractions apply to chronologically sorted raw/cleaned timestamps before eligible-window exclusions, not to equal wall-clock durations. "
                       "Later one-step forecasts may use earlier observed evaluation counts as history. This is rolling one-step evaluation, "
                       "not a single forecast of the entire evaluation period. Historical test inspection still prevents an untouched-confirmation claim.",
              sources=["src/traffic_lstm/pipeline_v2.py", "src/traffic_lstm/candidate.py", "src/traffic_lstm/backtest.py"]),
        _comparison(root, "motorway", "Motorway: the tree earns its place", "vehicles per hour"),
        _comparison(root, "bikes", "Bikes: a small MAE gap needs restraint", "rentals per hour"),
    ]
    rows = []
    coverage_rows = []
    sources = []
    for dataset in ("motorway", "bikes"):
        manifest_path = f"models/v2/{dataset}_v2_20260925/manifest.json"
        manifest = _read(root, manifest_path)
        if manifest is not None:
            try:
                quality = manifest["data_quality"]
                slides[2].rows.append({"Dataset": dataset, "Recorded date range": f"{quality['start'][:10]} to {quality['end'][:10]}",
                                       "Observed hours": quality["rows_clean"], "Missing hourly intervals": quality["missing_intervals"]})
                slides[2].sources.append(manifest_path)
            except (KeyError, TypeError):
                pass  # Absence is not zero missing data; do not invent a row.
        path = f"reports/backtests/{dataset}_20260925/results.json"
        sources.append(path)
        artifact = _read(root, path)
        if artifact is None:
            rows.append({"Dataset": dataset, "Evidence": "Unavailable"})
            continue
        try:
            runs = artifact["runs"]
            if not runs:
                raise ValueError("No completed comparisons")
            counts = {}
            for run in runs:
                counts[run["selected"]] = counts.get(run["selected"], 0) + 1
            rows.append({"Dataset": dataset, "Completed comparisons": len(runs),
                         "Folds": len({r["fold"] for r in runs}), "Seeds": len({r["seed"] for r in runs}),
                         "Validation selections": "; ".join(f"{name}: {n}" for name, n in sorted(counts.items())),
                         "LSTM epoch cap": artifact["protocol"]["epochs"]})
            coverages = [r["evaluation"][r["selected"]]["metrics"]["interval"]["coverage"] for r in runs]
            coverage_rows.append({"Dataset": dataset, "Nominal coverage": "90%",
                                  "Observed selected-model coverage": f"{min(coverages):.1%}–{max(coverages):.1%}"})
        except (KeyError, TypeError, ValueError):
            rows.append({"Dataset": dataset, "Evidence": "Incomplete"})
    slides.extend([
        Slide("Does the result survive other time periods?", "Repeat the comparison across forecast origins and training seeds.",
              "Expanding-origin experiments fit on earlier history and evaluate later periods. "
              "The recorded evaluation blocks do not overlap across folds. Different seeds probe training variation.\n\n"
              "The per-run evidence retains MAE, RMSE, MAPE, bias, peak MAE, interval coverage/width and all calibration/evaluation predictions. "
              "Paired bootstrap sensitivity uses 24-, 72- and 168-hour blocks.", rows=rows,
              evidence="These experiments use a 32/16-unit LSTM and a fixed epoch cap, distinct from the previous single-split runs. "
                       "Do not attribute a score change solely to the fold or the algorithm. Three seeds probe variability; they do not establish universal robustness. "
                       "Neither dataset supplies untouched future or independent-site confirmation.", sources=sources + ["reports/BACKTEST_REPORT.md"]),
        Slide("What forecast intervals can promise", "Nominal 90% is a target; measured coverage decides whether the interval is credible.",
              "A prediction interval surrounds an individual forecast. It is different from a bootstrap confidence interval around average error.\n\n"
              "The operational model sizes its intervals using residuals from a separate calibration period. "
              "It then measures **coverage**—how often the observed count falls inside—and **width**—how broad the interval is. "
              "Intervals that are very wide may have little operational value.\n\n"
              "The recorded bike results show undercoverage. Changes in demand can make yesterday’s calibration unreliable.", rows=coverage_rows,
              evidence="Intervals are based on absolute-residual split calibration, clipped at zero for nonnegative counts. "
                       "Temporal dependence and shift prevent a blanket coverage guarantee. The 85% local activation floor is an engineering gate "
                       "on 30+ evaluation windows, not a claim that 85% or 90% field coverage has been certified.", sources=sources + ["src/traffic_lstm/uncertainty.py"]),
        Slide("When the system refuses to forecast", "An explicit unavailable result is preferable to unsupported certainty.",
              "Serving checks the active model, data/model integrity, stream, cadence and observation mode. "
              "A replay origin cannot predate the model’s evidence. Live inputs cannot bypass freshness checks through replay.\n\n"
              "The system blocks incomplete recent history, stale observations and severe departures from the training count range. "
              "After enough actuals arrive, it checks recent error and interval coverage.\n\n"
              "Administrators can disable a stream or activate a reviewed previous model. Tokens, role checks, "
              "bounded background jobs, audit records and backup/restore support local operation.",
              evidence="Freshness limit: two declared intervals. Shift screen: over half of the last 24 counts outside the fit 1st–99th percentile range. "
                       "With at least 30 scored forecasts, the latest 100 trigger a block if coverage is below 85% or MAE exceeds "
                       "2 × max(1, evaluation MAE). This is a simple monitoring policy, not a comprehensive drift detector. "
                       "Alerts are visible in the app/audit trail; external notification routing is not configured.",
              sources=["docs/OPERATIONS.md", "src/traffic_lstm/observatory.py"]),
        Slide("What is delivered—and what remains open", "A tested local workflow is delivered; a field-ready traffic product is not yet established.",
              "**Implemented and exercised:** strict imports, persistent jobs, local access roles, model version selection, "
              "forecast/actual review, monitoring gates, and backup/restore. The bundled I-94 acceptance workflow "
              "checks the complete path through restored prediction parity.\n\n"
              "**Still required for a pilot:** an authorized local feed; certified sensor identity, timezone and aggregation semantics; "
              "untouched future/site evaluation; interval performance under local conditions; and a traffic engineer’s assessment of usefulness.\n\n"
              "**Before shared hosting:** managed identity/TLS, deployment isolation, secure secrets/backups and alert delivery. "
              "Local token access is not a substitute for those controls.",
              evidence="V1 reports remain archived exploratory evidence, not retroactively repaired results. No reduced-delay, congestion-control "
                       "or signal-actuation claim is supported. Automated application tests are recorded; external-browser visual review was unavailable "
                       "in the implementation session. Read the implementation log for verification scope.",
              sources=["reports/IMPLEMENTATION_LOG.md", "reports/observatory_acceptance.json", "ROAD_PRODUCT_PLAN.md"]),
        Slide("Start a transparent historical replay", "Use the bundled workflow to learn the product before introducing a local feed.",
              "**1. Sign in** to the initialized workspace.\n\n"
              "**2. Import and queue** the bundled motorway snapshot with later observations withheld.\n\n"
              "**3. Run the worker and review** the completed model’s evaluation and interval diagnostics.\n\n"
              "**4. Activate and forecast** with historical replay explicitly selected.\n\n"
              "**5. Import the withheld actuals** and inspect error and interval coverage.\n\n"
              "The operations guide contains the commands, role permissions, recovery steps and limits. "
              "A successful replay proves the workflow works; it does not validate a new city’s traffic model.",
              sources=["docs/OPERATIONS.md", "README.md"]),
    ])
    return slides


def briefing_markdown(slides):
    parts = ["# Traffic Observatory — project briefing\n",
             "Figures are drawn from the named saved experiments. Different protocols are not interchangeable.\n"]
    for i, slide in enumerate(slides, 1):
        parts.extend([f"## {i}. {slide.title}", f"**{slide.takeaway}**", slide.body])
        for label, value, unit in slide.facts:
            parts.append(f"- {label}: {value} ({unit})")
        if slide.rows:
            # Keep this export dependency-free; pandas Markdown would require tabulate.
            keys = list(dict.fromkeys(key for row in slide.rows for key in row))
            parts.extend(["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"])
            parts.extend("| " + " | ".join(str(row.get(key, "—")) for key in keys) + " |" for row in slide.rows)
        if slide.evidence:
            parts.append("Details: " + slide.evidence)
        parts.append("Sources: " + ", ".join(f"`{source}`" for source in slide.sources))
    return "\n\n".join(parts) + "\n"


def render_project_slideshow():
    slides = build_slides()
    key = "project_briefing_slide"
    st.session_state.setdefault(key, 0)
    if st.session_state[key] not in range(len(slides)):
        st.session_state[key] = 0

    def move(delta):
        st.session_state[key] = max(0, min(len(slides)-1, st.session_state[key] + delta))

    st.subheader("Project briefing")
    st.caption("Explore the purpose, methods, recorded results and delivery limits. Advance at your own pace.")
    previous, jump, following = st.columns([1, 5, 1])
    previous.button("Previous", key="briefing_previous", disabled=st.session_state[key] == 0,
                    on_click=move, args=(-1,), width="stretch")
    jump.selectbox("Jump to a section", range(len(slides)), key=key,
                   format_func=lambda i: f"{i+1:02d} · {slides[i].title}", label_visibility="collapsed")
    following.button("Next", key="briefing_next", disabled=st.session_state[key] == len(slides)-1,
                     on_click=move, args=(1,), width="stretch")
    index = st.session_state[key]
    slide = slides[index]
    st.progress((index+1)/len(slides), text=f"Slide {index+1} of {len(slides)}")
    with st.container(border=True):
        st.header(slide.title)
        st.markdown(f"**{slide.takeaway}**")
        if slide.facts:
            for col, (label, value, unit) in zip(st.columns(len(slide.facts)), slide.facts):
                col.metric(label, value, help=unit)
        st.markdown(slide.body)
        if slide.rows:
            st.dataframe(pd.DataFrame(slide.rows), hide_index=True, width="stretch")
        with st.expander("Details and source evidence"):
            if slide.evidence:
                st.write(slide.evidence)
            for source in slide.sources:
                st.caption(source)
    st.download_button("Download complete project briefing", briefing_markdown(slides),
                       file_name="traffic_observatory_briefing.md", mime="text/markdown", key="briefing_download")
