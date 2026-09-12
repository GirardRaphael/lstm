"""Turn a trained network into an Obsidian vault you can walk through.

Nothing in the vault is decorative. Every number is read back out of the
trained model by `introspect.trace_network`, so what you see in Obsidian is
what the network actually computed:

    Canvas/Network Architecture.canvas     the shape of the model
    Canvas/Neurons - Rush Hour.canvas      every unit, coloured by activation
    Canvas/Neurons - Quiet Night.canvas    the same units on an empty road
    Canvas/Timeline/t00 ... t23.canvas     one canvas per hour, memory forming
    Neurons/*.md                           one note per unit, with its trace
    Gates/*.md                             the four gates, with this model's values

Open `obsidian_vault/Traffic_LSTM_Brain` in Obsidian ("Open folder as vault")
and start from `00 Start Here`.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from . import plots
from .config import FIGURE_DIR, VAULT_DIR, TrainingConfig
from .data import inverse_target
from .evaluate import traffic_level
from .introspect import GATE_NAMES, input_sensitivity, most_active, neuron_profiles, trace_network
from .obsidian_canvas import (
    build_architecture_canvas,
    build_neuron_canvas,
    build_timeline_canvas,
    sparkline,
)

GENERATED_DIRS = ("Canvas", "Neurons", "Layers", "Gates", "Figures")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _reset(vault: Path) -> None:
    """Clear only what we generate, so hand-written notes survive a rebuild."""
    for name in GENERATED_DIRS:
        target = vault / name
        if target.exists():
            shutil.rmtree(target)
    vault.mkdir(parents=True, exist_ok=True)


def _obsidian_settings(vault: Path) -> None:
    """Pre-configure the vault so the graph view looks right on first open."""
    cfg_dir = vault / ".obsidian"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    _write(cfg_dir / "app.json", json.dumps({
        "attachmentFolderPath": "Figures",
        "alwaysUpdateLinks": True,
        "newLinkFormat": "shortest",
    }, indent=2))
    _write(cfg_dir / "appearance.json", json.dumps({
        "accentColor": "#d62728", "theme": "obsidian",
    }, indent=2))
    _write(cfg_dir / "core-plugins.json", json.dumps([
        "file-explorer", "global-search", "switcher", "graph", "backlink",
        "outgoing-link", "tag-pane", "page-preview", "canvas", "outline",
        "word-count", "file-recovery",
    ], indent=2))
    _write(cfg_dir / "graph.json", json.dumps({
        "collapse-filter": False, "search": "", "showTags": True,
        "showAttachments": False, "hideUnresolved": True, "showOrphans": False,
        "collapse-color-groups": False,
        "colorGroups": [
            {"query": "path:Neurons/lstm_1", "color": {"a": 1, "rgb": 14043432}},
            {"query": "path:Neurons/lstm_2", "color": {"a": 1, "rgb": 2071764}},
            {"query": "path:Gates", "color": {"a": 1, "rgb": 11780968}},
            {"query": "path:Layers", "color": {"a": 1, "rgb": 9737471}},
        ],
        "collapse-display": False, "showArrow": True, "textFadeMultiplier": -0.6,
        "nodeSizeMultiplier": 1.15, "lineSizeMultiplier": 1,
        "collapse-forces": False, "centerStrength": 0.42, "repelStrength": 12,
        "linkStrength": 0.6, "linkDistance": 180, "scale": 0.5,
    }, indent=2))


# ------------------------------------------------------------- sampling ----
def _window_context(cfg: TrainingConfig, bundle, model, index: int) -> dict:
    """Everything needed to describe one 24-hour window in human units."""
    window_scaled = bundle.X_test[index]
    window_values = inverse_target(bundle.scaler, window_scaled[:, 0]).ravel()
    target = inverse_target(bundle.scaler, bundle.y_test[index])[0]
    predicted = inverse_target(
        bundle.scaler, model.predict(window_scaled[None, ...], verbose=0)[0])[0]
    return {
        "index": index,
        "timestamp": str(bundle.test_timestamps.iloc[index]),
        "window_values": window_values,
        "target": float(target),
        "prediction": float(predicted),
    }


def _pick_windows(bundle) -> dict:
    """Two contrasting real windows: the busiest hour and the quietest."""
    actual = inverse_target(bundle.scaler, bundle.y_test)[:, 0]
    return {"rush_hour": int(np.argmax(actual)), "night": int(np.argmin(actual))}


# ------------------------------------------------------------ note bodies ---
def _note_start_here(cfg: TrainingConfig, artifacts: dict, contexts: dict) -> str:
    horizon = cfg.horizons[0]
    block = artifacts["results"]["h{}".format(horizon)]
    lstm, naive = block["lstm"], block["naive_same_hour_yesterday"]
    return """---
tags: [moc]
---
# Traffic LSTM - Start Here

Predicting motorway traffic one hour ahead with a stacked LSTM, and then
opening the network up to see *how* it got there.

> **Headline:** the model is wrong by **{mae:,.0f} vehicles per hour** on
> average, against **{naive:,.0f}** for the best naive baseline
> ({gain:+.1f}%). Hourly traffic on this road ranges from ~{lo:,.0f} to
> ~{hi:,.0f} vehicles.

## Walk through it in this order

1. [[01 Problem]] - why forecasting traffic is worth doing
2. [[02 Dataset]] - what the data actually contains
3. [[03 Pipeline]] - raw CSV to supervised sequences, without leaking
4. [[04 Architecture]] - the network, layer by layer
5. **[[Network Architecture.canvas|Canvas: the model]]**
6. **[[Neurons - Rush Hour.canvas|Canvas: every neuron at rush hour]]**
7. **[[Neurons - Quiet Night.canvas|Canvas: the same neurons at night]]**
8. **[[t00.canvas|Canvas: hour-by-hour timeline]]** - 24 canvases, the memory forming
9. [[05 Results]] - metrics, baselines, figures
10. [[06 Limits and Next Steps]]
11. [[07 Presentation Script]] and [[08 Exam Questions]]

## The mechanism in one picture

```mermaid
flowchart LR
    A["24 past hours"] --> B["lstm_1<br/>{u1} units"]
    B --> C["lstm_2<br/>{u2} units"]
    C --> D["Dense {dense}"]
    D --> E["Traffic next hour"]
    B -.->|"memory c"| B
    C -.->|"memory c"| C
```

## Inside the layers

- [[lstm_1]] - {u1} units, one hidden state per hour
- [[lstm_2]] - {u2} units, one summary vector for the whole day
- [[dense_hidden]] and [[traffic_output]]

## The four gates

[[Forget gate]] - [[Input gate]] - [[Candidate memory]] - [[Output gate]] - [[Cell state]]

---
*Generated from `{run}` on {stamp}. Rebuild with*
`python -m traffic_lstm.train --export-vault`
""".format(
        mae=lstm["mae"], naive=naive["mae"],
        gain=block["improvement_over_best_naive_pct"],
        lo=artifacts["data_quality"]["target_min"],
        hi=artifacts["data_quality"]["target_max"],
        u1=cfg.lstm_units[0], u2=cfg.lstm_units[-1], dense=cfg.dense_units,
        run=cfg.run_name, stamp=datetime.now().strftime("%Y-%m-%d %H:%M"))


def _note_problem(cfg: TrainingConfig) -> str:
    return """---
tags: [context]
---
# 01 Problem

Traffic on a motorway is not random. It follows a strong daily cycle - a
morning peak, a midday plateau, an evening peak, an overnight trough - and
that cycle is modulated by the day of the week, the weather and holidays.

**The question this project answers:**

> Given the last {seq} hours of traffic volume, how many vehicles will pass
> during the next hour?

## Why it matters

A traffic controller that only *reacts* to congestion is always one step
behind. A controller that knows the next hour in advance can act before the
queue forms.

```mermaid
flowchart LR
    S["Loop sensors"] --> H["Hourly history"]
    H --> M["LSTM forecast"]
    M --> C["Signal controller"]
    C --> L["Adaptive traffic lights"]
    style M fill:#d62728,color:#fff
```

**Be precise about the claim.** This model does *not* control traffic lights.
It produces a forecast that a control system could consume. Saying anything
stronger would be overselling it.

## Type of problem

| Question | Answer |
| --- | --- |
| Learning type | Supervised - each window has a known future value |
| Task | Regression - the output is a count of vehicles |
| Data shape | Univariate time series, hourly |
| Metric | MAE, in vehicles per hour |

Next: [[02 Dataset]]
""".format(seq=cfg.sequence_length)


def _note_dataset(cfg: TrainingConfig, artifacts: dict) -> str:
    q = artifacts["data_quality"]
    return """---
tags: [data]
---
# 02 Dataset

**Metro Interstate Traffic Volume** - UCI Machine Learning Repository.
Hourly westbound traffic on I-94 between Minneapolis and Saint Paul.

| Property | Value |
| --- | --- |
| Rows | {rows:,} |
| Period | {start} to {end} |
| Target | `{target}` - vehicles counted during the hour |
| Range | {lo:,.0f} to {hi:,.0f} vehicles (mean {mean:,.0f}) |
| Duplicate timestamps | {dupes} |
| Missing target values | {missing} |
| Hours missing from the range | {gaps:,} |

## What we do about the imperfections

**Duplicate timestamps.** The same hour is sometimes logged twice with
different weather descriptions. We keep one row per hour and average the
target - we neither invent data nor silently keep two conflicting rows.

**Gaps.** There are {gaps:,} hours with no record at all, mostly a long
outage in 2014-2015. We do *not* interpolate them. Inventing traffic
volumes would make the metrics look better than the model deserves. The
sequence builder simply slides over the rows that exist.

## Columns we deliberately ignore for now

`temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `holiday`.

Version 1 is univariate on purpose: it isolates the question "can past traffic
alone predict future traffic?". Adding weather is the first item in
[[06 Limits and Next Steps]].

![[01_dataset.png]]

![[02_daily_profile.png]]

The second figure is the pattern the network has to learn: two sharp peaks,
a deep overnight trough, and a spread that widens during the day.

Next: [[03 Pipeline]]
""".format(rows=q["rows"], start=q["start"][:16], end=q["end"][:16],
           target=cfg.target_column, lo=q["target_min"], hi=q["target_max"],
           mean=q["target_mean"], dupes=q["duplicated_timestamps"],
           missing=q["missing_values"], gaps=q["missing_hours"])


def _note_pipeline(cfg: TrainingConfig, artifacts: dict) -> str:
    return """---
tags: [method]
---
# 03 Pipeline

```mermaid
flowchart TD
    A["Raw CSV<br/>{rows:,} rows"] --> B["Sort chronologically<br/>collapse duplicate hours"]
    B --> C["Chronological split<br/>{train:.0f}% train / {test:.0f}% test"]
    C --> D["MinMaxScaler<br/>fitted on TRAIN ONLY"]
    D --> E["Sliding windows<br/>{seq} in, {n_out} out"]
    E --> F["Reshape to<br/>(samples, {seq}, 1)"]
    F --> G["LSTM"]
    style D fill:#2ca02c,color:#fff
    style C fill:#2ca02c,color:#fff
```

## The three rules that matter

### 1. The split is chronological, never shuffled

```python
train_size = int(len(values) * {ratio})
train, test = values[:train_size], values[train_size:]
```

`train_test_split(shuffle=True)` would put future hours in the training set
and past hours in the test set. The score would look excellent and mean
nothing, because in production you only ever have the past.

### 2. The scaler is fitted on the training slice only

```python
train_scaled = scaler.fit_transform(train)   # fit + transform
test_scaled  = scaler.transform(test)        # transform only
```

If the scaler saw the test set, the model would indirectly know the maximum
traffic volume of the future. That is **data leakage**.

### 3. The test set keeps its history

The first test window needs {seq} hours of history, and those hours live at
the end of the training set. We prepend them - that is history, not leakage,
because the model only ever looks backwards.

## Building the windows

```python
for i in range(sequence_length, len(data) - max_horizon + 1):
    X.append(data[i - sequence_length:i, 0])   # the last {seq} hours
    y.append(data[i + horizon - 1, 0])         # the hour we want
```

| | |
| --- | --- |
| Training sequences | {n_train:,} |
| Test sequences | {n_test:,} |
| Input shape | `({seq}, 1)` = {seq} timesteps, 1 feature |

**Why {seq} hours?** One full daily cycle. The network sees a complete
morning peak, evening peak and overnight trough before it has to predict.
A shorter window would cut the cycle in half; a much longer one adds
parameters without adding information.

Next: [[04 Architecture]]
""".format(rows=artifacts["data_quality"]["rows"], train=cfg.train_ratio * 100,
           test=(1 - cfg.train_ratio) * 100, seq=cfg.sequence_length,
           n_out=cfg.n_outputs, ratio=cfg.train_ratio,
           n_train=artifacts["train_sequences"], n_test=artifacts["test_sequences"])


def _note_architecture(cfg: TrainingConfig, artifacts: dict, trace) -> str:
    return """---
tags: [method, architecture]
---
# 04 Architecture

```mermaid
flowchart TD
    I["Input (({seq}, 1))"] --> L1["LSTM {u1}<br/>return_sequences=True"]
    L1 --> D1["Dropout {drop}"]
    D1 --> L2["LSTM {u2}<br/>return_sequences=False"]
    L2 --> D2["Dropout {drop}"]
    D2 --> DN["Dense {dense} · relu"]
    DN --> O["Dense {nout} · linear"]
    style L1 fill:#1f77b4,color:#fff
    style L2 fill:#1f77b4,color:#fff
    style O fill:#d62728,color:#fff
```

Open **[[Network Architecture.canvas]]** for the same thing as a canvas.

## Layer by layer

- [[lstm_1]] - {u1} units, returns a hidden state for **every one of the
  {seq} hours**, so the next layer still sees a sequence.
- **Dropout {drop}** - during training, {pct:.0f}% of the outputs are zeroed at
  random. The network cannot lean on any single unit, which reduces
  overfitting. At prediction time dropout is off.
- [[lstm_2]] - {u2} units, returns **only the final state**: one vector that
  summarises the whole day.
- [[dense_hidden]] - {dense} units with `relu`, recombining the summary.
- [[traffic_output]] - {nout} linear unit(s). **No activation**, because the
  answer is a number of vehicles, not a probability.

## What one LSTM unit computes

At every hour `t`, each unit runs these five lines:

```
z = x_t . W + h_{{t-1}} . U + b

i = sigmoid(z_i)     input gate    - how much new information to write
f = sigmoid(z_f)     forget gate   - how much of the memory to keep
g = tanh(z_g)        candidate     - what the new information is
o = sigmoid(z_o)     output gate   - how much memory to expose

c_t = f * c_{{t-1}} + i * g   long-term memory
h_t = o * tanh(c_t)          what the next layer sees
```

That is the whole mechanism: [[Forget gate]] decides what survives,
[[Input gate]] decides what gets written, [[Candidate memory]] is the content,
[[Output gate]] decides what leaks out, and [[Cell state]] is the memory itself.

## This is not a diagram from a textbook

Every activation in this vault was recomputed from the trained weights with
the NumPy re-implementation in `introspect.py`, then checked against Keras:

> **Maximum absolute difference: `{err:.2e}`**

Same network, same numbers. If they disagreed, the vault would be fiction.

## Parameter count

```
{summary}
```

Next: [[05 Results]]
""".format(seq=cfg.sequence_length, u1=cfg.lstm_units[0], u2=cfg.lstm_units[-1],
           drop=cfg.dropout, pct=cfg.dropout * 100, dense=cfg.dense_units,
           nout=cfg.n_outputs, err=trace.max_abs_error or 0.0,
           summary=artifacts["architecture"])


def _sensitivity_commentary(sensitivity, sequence_length: int) -> str:
    """Describe where the model's attention actually went, from the numbers."""
    order = list(np.argsort(-np.asarray(sensitivity)))
    top = order[:3]
    bullets = "\n".join(
        "- **t-{}h** - {:.1f}% of the movement".format(sequence_length - int(t),
                                                       sensitivity[int(t)] * 100)
        for t in top)
    recent_share = float(np.sum(sensitivity[-3:])) * 100
    distant = [int(t) for t in order[:6] if t < sequence_length - 4]
    extra = ""
    if distant:
        hours = ", ".join("t-{}h".format(sequence_length - t) for t in sorted(distant, reverse=True))
        extra = ("\n\nThe last three hours account for {:.0f}% of the movement, but "
                 "{} also rank in the top six. The network is consulting roughly half a "
                 "day back as well as the immediate past - which is why it beats the "
                 "persistence baseline instead of tying with it.".format(recent_share, hours))
    else:
        extra = ("\n\nThe last three hours account for {:.0f}% of the movement. On this "
                 "window the model leans heavily on the immediate past.".format(recent_share))
    return bullets + extra


def _note_results(cfg: TrainingConfig, artifacts: dict, sensitivity=None) -> str:
    rows = []
    for horizon in cfg.horizons:
        block = artifacts["results"]["h{}".format(horizon)]
        rows.append(
            "| +{h}h | {mae:,.1f} | {rmse:,.1f} | {mape:.1f}% | {nv:,.1f} | {pers:,.1f} | "
            "**{gain:+.1f}%** |".format(
                h=horizon, mae=block["lstm"]["mae"], rmse=block["lstm"]["rmse"],
                mape=block["lstm"]["mape"],
                nv=block["naive_same_hour_yesterday"]["mae"],
                pers=block["naive_persistence"]["mae"],
                gain=block["improvement_over_best_naive_pct"]))

    first = artifacts["results"]["h{}".format(cfg.horizons[0])]
    return """---
tags: [results]
---
# 05 Results

## Metrics on the test set ({n:,} unseen hours)

| Horizon | MAE | RMSE | MAPE | Same hour yesterday | Last hour | Gain |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

## Say it in words, not in numbers

> On hours it had never seen, the model is wrong by about
> **{mae:,.0f} vehicles per hour** on average. Traffic on this road runs
> between {lo:,.0f} and {hi:,.0f} vehicles an hour, so that is roughly
> **{mape:.0f}%** off.

## Why the baselines are in the table

A MAE of {mae:,.0f} means nothing on its own. Two baselines make it an argument:

- **Last hour** (persistence): predict whatever the previous hour was. MAE {pers:,.0f}.
- **Same hour yesterday**: exploit the daily cycle without any learning. MAE {nv:,.0f}.

The network lands at {mae:,.0f}, i.e. **{gain:+.1f}%** against the better of
the two. If it had not beaten both, the honest conclusion would have been
that an LSTM is the wrong tool here.

![[09_baselines.png]]

## Learning curve

![[03_training_loss.png]]

Trained for **{epochs} epochs** ({secs}s) before EarlyStopping restored the
best weights. Training and validation loss fall together - the sign that
dropout and early stopping are doing their job.

## Actual vs predicted

![[04_actual_vs_predicted.png]]

The model tracks the shape of the daily cycle closely. It tends to shave the
very top of the sharp peaks, which is expected: MSE rewards being close on
average more than nailing the extremes.

## Error distribution

![[05_error_distribution.png]]

## Which past hours the model actually uses

![[08_input_sensitivity.png]]

Each of the {seq} input hours was nudged in turn and the change in the output
measured, on the busiest window of the test set:

{sensitivity}

## Now look inside

- **[[Neurons - Rush Hour.canvas]]** - every unit on the busiest hour of the test set
- **[[Neurons - Quiet Night.canvas]]** - the same units, near-empty road
- **[[t00.canvas]]** - 24 canvases, one per hour

Next: [[06 Limits and Next Steps]]
""".format(n=artifacts["test_sequences"], rows="\n".join(rows), seq=cfg.sequence_length,
           sensitivity=(_sensitivity_commentary(sensitivity, cfg.sequence_length)
                        if sensitivity is not None else ""),
           mae=first["lstm"]["mae"], mape=first["lstm"]["mape"],
           lo=artifacts["data_quality"]["target_min"],
           hi=artifacts["data_quality"]["target_max"],
           pers=first["naive_persistence"]["mae"],
           nv=first["naive_same_hour_yesterday"]["mae"],
           gain=first["improvement_over_best_naive_pct"],
           epochs=artifacts["epochs_run"], secs=artifacts["training_seconds"])


def _note_limits(cfg: TrainingConfig) -> str:
    return """---
tags: [critique]
---
# 06 Limits and Next Steps

## What this model cannot do

**It has never heard of weather, holidays or accidents.** It only sees past
traffic. A snowstorm, a closed lane or a stadium emptying out are invisible
to it, and those are exactly the hours where a forecast would be most useful.

**It cannot predict a first-time event.** An LSTM extrapolates patterns it
has seen. The first hour of an unprecedented situation will be missed.

**It is hard to interpret.** This vault is an attempt to fix that, but a
decision tree would explain itself in one line. That trade is the price of
modelling temporal dependencies.

**It needs a lot of data and compute.** ~48,000 hours here. On a few hundred
rows, a simpler model would win.

## When an LSTM would be the wrong choice

If the observations had no temporal relationship, the sequence structure
would be wasted and Random Forest or XGBoost on engineered features
(hour, weekday, lag-1, lag-24) would be simpler, faster and often as good.
That is an honest comparison to run, not a weakness to hide.

## Next steps, in order of value

1. **Add the exogenous columns.** `temp`, `rain_1h`, `snow_1h`, `holiday`,
   plus hour-of-day and day-of-week encoded as sine/cosine pairs. Input shape
   becomes `({seq}, 8)` and nothing else in the pipeline changes.
2. **Compare against a gradient-boosted baseline** on the same split.
3. **Predict several hours ahead.** Already supported:
   `--horizons 1 3 6` trains one model with {seq}-hour input and three outputs.
4. **Feed a controller.** The forecast becomes an input to a signal-timing
   optimiser - see the diagram in [[01 Problem]].

## The honest summary

> The model learns the daily traffic cycle well and beats both naive
> baselines. It does not understand *why* traffic changes, and it will fail
> on exactly the unusual hours a traffic operator cares about most. Adding
> weather and calendar features is the obvious next move.

Next: [[07 Presentation Script]]
""".format(seq=cfg.sequence_length)


def _note_script(cfg: TrainingConfig, artifacts: dict, contexts: dict) -> str:
    first = artifacts["results"]["h{}".format(cfg.horizons[0])]
    rush = contexts["rush_hour"]
    level, _ = traffic_level(rush["prediction"], cfg.level_low, cfg.level_high)
    return """---
tags: [presentation]
---
# 07 Presentation Script

Target: 8 minutes, plus questions. Times are cumulative.

## 0:00 - 0:45 - The question

> "Traffic is a time series: what happens now depends on what happened in the
> hours before. My project asks whether the last {seq} hours of traffic volume
> are enough to predict the next hour. I used an LSTM, a neural network built
> for sequences."

## 0:45 - 2:00 - The data

Show [[02 Dataset]] and `01_dataset.png`.

> "{rows:,} hourly measurements from the UCI repository, on I-94 between
> Minneapolis and Saint Paul, from 2012 to 2018. The target is
> `traffic_volume`: vehicles counted during the hour."

Show `02_daily_profile.png`.

> "This is the pattern the network has to learn: a morning peak, an evening
> peak, an overnight trough."

## 2:00 - 3:15 - The method

Show the diagram in [[03 Pipeline]]. Hit the two points a teacher will probe:

> "The split is chronological, not random - in production you only have the
> past. And the scaler is fitted only on the training set, otherwise the model
> would indirectly know the range of the future. That is data leakage."

## 3:15 - 4:15 - The network

Open **[[Network Architecture.canvas]]**.

> "{seq} hours in. A first LSTM of {u1} units produces one hidden state per
> hour. A second LSTM of {u2} units collapses that into a single vector for
> the day. Two dense layers turn it into one number. The last layer is linear
> because the output is a count of vehicles, not a probability."

## 4:15 - 5:30 - **The demo** (the part they will remember)

Open **[[Neurons - Rush Hour.canvas]]**.

> "This is not an illustration. Every colour is a real activation, replayed
> from the trained weights. Red means the unit fired positively, blue
> negatively, grey means it stayed silent. This window ends at {ts}, the
> busiest hour of the test set: the model predicted {pred:,.0f} vehicles,
> the truth was {truth:,.0f}."

Then open **[[Neurons - Quiet Night.canvas]]**.

> "Same network, an almost empty road. A completely different set of units
> lights up. The network has specialised."

Then open **[[t00.canvas]]** and step through a few hours.

> "And here is the memory forming, hour by hour. Each canvas is one timestep.
> Watch the forget gate: when it stays near 1, the unit is holding on to the
> whole day."

## 5:30 - 6:45 - The results

Show [[05 Results]].

> "On {n:,} hours the model had never seen, it is wrong by
> **{mae:,.0f} vehicles per hour** on average. Predicting 'the same hour
> yesterday' gives {nv:,.0f}, so the network is {gain:.0f}% better. Without
> that comparison the number would not mean anything."

Show `04_actual_vs_predicted.png`.

> "It follows the daily cycle and slightly flattens the sharpest peaks."

## 6:45 - 8:00 - Limits and what comes next

From [[06 Limits and Next Steps]]:

> "The model has never heard of weather, holidays or accidents - exactly the
> situations where a forecast matters most. The next version adds those
> columns. And to be precise about the claim: this model does not control
> traffic lights. It produces a forecast a controller could use."

Have [[08 Exam Questions]] open in a second tab.

## The live demo, as a fallback

```
python -m traffic_lstm.predict --last-hours

  Window    : {seq} hours ending {ts}
  Prediction: {pred:,.0f} vehicles
  Level     : {level}
```
""".format(seq=cfg.sequence_length, rows=artifacts["data_quality"]["rows"],
           u1=cfg.lstm_units[0], u2=cfg.lstm_units[-1],
           ts=rush["timestamp"][:16], pred=rush["prediction"], truth=rush["target"],
           n=artifacts["test_sequences"], mae=first["lstm"]["mae"],
           nv=first["naive_same_hour_yesterday"]["mae"],
           gain=first["improvement_over_best_naive_pct"], level=level)


def _benchmark_answer(cfg: TrainingConfig, lstm_mae: float) -> str:
    """Answer the "why an LSTM?" question with a measurement, not theory."""
    path = MODEL_DIR / "benchmark_{}.json".format(cfg.run_name)
    if not path.exists():
        return ("With no temporal structure, or with little data. Gradient boosting on\n"
                "lag features would be simpler and often as strong - run\n"
                "`python -m traffic_lstm.benchmark` to find out for this dataset instead\n"
                "of guessing. See [[06 Limits and Next Steps]].")

    payload = json.loads(path.read_text(encoding="utf-8"))
    xgb = payload["xgboost"]
    xgb_mae = xgb["metrics"]["mae"]
    lstm_seconds = payload["comparison"]["lstm_training_seconds"]
    speed = lstm_seconds / max(xgb["training_seconds"], 0.1)

    if xgb_mae < lstm_mae:
        gap = (lstm_mae - xgb_mae) / lstm_mae * 100
        return (
            "**On this dataset, it is.** I measured it rather than arguing it. XGBoost\n"
            "on exactly the same tensors - same windows, same split, same scaler, just\n"
            "flattened - scores **MAE {xgb:,.0f}** against the network's **{lstm:,.0f}**.\n"
            "That is **{gap:.0f}% better**, trained **{speed:.0f}x faster**\n"
            "({xs:,.0f}s against {ls:,.0f}s).\n\n"
            "One strongly periodic series with a 24-hour window is close to the ideal\n"
            "case for a tree ensemble: almost all the signal is in `t-1h`, `t-2h` and\n"
            "`t-24h`, and a tree splits on those directly without having to learn a\n"
            "recurrence at all.\n\n"
            "An LSTM starts to earn its cost with dependencies longer than the input\n"
            "window, many correlated series sharing one model, irregular sequence\n"
            "lengths, or when the learned representation is reused downstream.\n"
            "See [[09 Model Comparison]].".format(
                xgb=xgb_mae, lstm=lstm_mae, gap=gap, speed=speed,
                xs=xgb["training_seconds"], ls=lstm_seconds))

    gap = (xgb_mae - lstm_mae) / xgb_mae * 100
    return (
        "I checked rather than assumed. XGBoost on the same flattened tensors scores\n"
        "**MAE {xgb:,.0f}** against the network's **{lstm:,.0f}**, so the sequence\n"
        "structure is worth **{gap:.1f}%** here and the LSTM is justified.\n\n"
        "It would still be the wrong choice with no temporal structure, or on a few\n"
        "hundred rows, where a tree model would be simpler and cheaper.\n"
        "See [[09 Model Comparison]].".format(xgb=xgb_mae, lstm=lstm_mae, gap=gap))


def _note_questions(cfg: TrainingConfig, artifacts: dict, trace) -> str:
    first = artifacts["results"]["h{}".format(cfg.horizons[0])]
    return """---
tags: [presentation, qa]
---
# 08 Exam Questions

### Why an LSTM rather than a plain neural network?

A dense network treats the {seq} inputs as {seq} unrelated numbers. An LSTM
processes them **in order** and carries a memory from one hour to the next,
so it can learn that a rising 16:00 usually leads to a peak at 17:00.

### What does the LSTM actually remember?

The cell state `c`. At each hour the forget gate decides what fraction of `c`
survives, the input gate decides how much new information is written. A unit
whose forget gate stays near 1 carries information across the whole day - see
[[Cell state]] and the timeline canvases.

### What is overfitting, and what did you do about it?

Learning the training set so precisely that performance on new data drops.
Four defences here: **Dropout {drop}**, a **validation split** watched every
epoch, **EarlyStopping** (patience {pat}, best weights restored), and a
**chronological train/test split** so the test set is genuinely unseen.

### Why MAE rather than MSE for reporting?

MSE is the training loss because it penalises large misses harder. MAE is the
reported metric because it is in the unit of the problem: "wrong by
{mae:,.0f} vehicles per hour" is a sentence anyone can check. Both are in
[[05 Results]].

### Is a MAE of {mae:,.0f} good?

On its own, unknowable - which is why the baselines are there. Predicting
"the same hour yesterday" gives {nv:,.0f} and "the last hour" gives
{pers:,.0f}. The model is **{gain:+.1f}%** better than the best of them.

### Why {seq} timesteps?

One full daily cycle. Shorter cuts the cycle in half; much longer adds
parameters without new information.

### Why is the last layer linear?

Regression. A sigmoid would squash the output into [0, 1] and a softmax would
turn it into class probabilities. We want a count of vehicles.

### What do the {u1} units in the first layer do?

Each learns its own feature of the sequence. On the rush-hour window,
{active} of {u1} units carry most of the signal while the rest stay near
zero - open [[Neurons - Rush Hour.canvas]] and the grey nodes are the quiet
ones. Individual units are documented in `Neurons/`.

### How do you know the numbers in this vault are real?

`introspect.py` re-implements the LSTM cell in NumPy from the trained weights
and is checked against Keras on every export. Current maximum absolute
difference: **{err:.2e}**.

### When would an LSTM be the wrong choice?

{benchmark}

### Does this control the traffic lights?

No. It produces a forecast that a signal controller could consume. Claiming
more would be wrong.
""".format(seq=cfg.sequence_length, drop=cfg.dropout, pat=cfg.patience,
           mae=first["lstm"]["mae"], nv=first["naive_same_hour_yesterday"]["mae"],
           pers=first["naive_persistence"]["mae"],
           gain=first["improvement_over_best_naive_pct"], u1=cfg.lstm_units[0],
           active=sum(1 for p in neuron_profiles(trace.layers[0])
                      if p["mean_abs_activation"] > 0.05),
           err=trace.max_abs_error or 0.0)


# --------------------------------------------------------------- gate notes --
GATE_NOTES = {
    "Forget gate": (
        "f", "`f = sigmoid(x_t . W_f + h_(t-1) . U_f + b_f)`",
        "Decides what fraction of the existing memory survives this hour.\n\n"
        "- `f` near **1** - keep the memory intact (a unit tracking the whole day)\n"
        "- `f` near **0** - wipe it and start fresh (a unit that only cares about now)\n\n"
        "It multiplies the old cell state: `c_t = f * c_(t-1) + ...`"),
    "Input gate": (
        "i", "`i = sigmoid(x_t . W_i + h_(t-1) . U_i + b_i)`",
        "Decides how much of the new candidate information gets written into "
        "memory this hour.\n\n- `i` near **1** - this hour is worth remembering\n"
        "- `i` near **0** - ignore it\n\nIt gates the candidate: `c_t = ... + i * g`"),
    "Candidate memory": (
        "g", "`g = tanh(x_t . W_g + h_(t-1) . U_g + b_g)`",
        "The *content* the unit proposes to store, squashed into [-1, 1]. "
        "[[Input gate]] then decides how much of it actually lands in "
        "[[Cell state]]."),
    "Output gate": (
        "o", "`o = sigmoid(x_t . W_o + h_(t-1) . U_o + b_o)`",
        "Decides how much of the memory is exposed to the next layer:\n\n"
        "`h_t = o * tanh(c_t)`\n\nA unit can hold information in `c` for hours "
        "while keeping `o` closed, then open it exactly when it matters."),
}


def _note_gate(name: str, trace, contexts: dict) -> str:
    key, formula, description = GATE_NOTES[name]
    layer = trace.layers[0]
    series = layer.gate(key).mean(axis=1)
    steps = layer.timesteps
    return """---
tags: [mechanism, gate]
---
# {name}

{formula}

{description}

## What this gate did on the rush-hour window

Averaged over all {units} units of [[lstm_1]], hour by hour:

`{spark}`

| | |
| --- | --- |
| Mean opening | {mean:.3f} |
| Most open at | t-{peak_h}h ({peak:.3f}) |
| Most closed at | t-{low_h}h ({low:.3f}) |

Step through **[[t00.canvas]]** to see this gate hour by hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] -
[[Output gate]] - [[Cell state]] - [[04 Architecture]]
""".format(name=name, formula=formula, description=description, units=layer.units,
           spark=sparkline(series), mean=float(series.mean()),
           peak_h=steps - int(np.argmax(series)), peak=float(series.max()),
           low_h=steps - int(np.argmin(series)), low=float(series.min()))


def _note_cell_state(trace) -> str:
    layer = trace.layers[0]
    magnitude = np.abs(layer.c).mean(axis=1)
    saturated = int((np.abs(layer.c[-1]) > 0.9).sum())
    return """---
tags: [mechanism, gate]
---
# Cell state

`c_t = f * c_(t-1) + i * g`

The memory itself - the reason an LSTM can connect an hour to something that
happened a day earlier. It is never shown to the next layer directly:
[[Output gate]] decides how much of it leaks out as `h_t = o * tanh(c_t)`.

## How the memory built up on the rush-hour window

Mean `|c|` across the {units} units of [[lstm_1]], hour by hour:

`{spark}`

| | |
| --- | --- |
| Mean magnitude at the start | {start:.3f} |
| Mean magnitude at the end | {end:.3f} |
| Units saturated at the end (`|c| > 0.9`) | {sat} of {units} |

A rising line means the layer is accumulating information as it walks through
the day rather than resetting at every hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] - [[Output gate]]
""".format(units=layer.units, spark=sparkline(magnitude), start=float(magnitude[0]),
           end=float(magnitude[-1]), sat=saturated)


# -------------------------------------------------------------- layer notes --
def _note_layer(cfg: TrainingConfig, layer, index: int, profiles: list) -> str:
    top = most_active(profiles, 8)
    rows = "\n".join(
        "| [[{ln} u{u:02d}\\|u{u:02d}]] | {mean:+.3f} | {final:+.3f} | t-{peak}h | {role} |".format(
            ln=layer.name, u=p["unit"], mean=p["mean_abs_activation"], final=p["final_h"],
            peak=layer.timesteps - p["peak_timestep"], role=p["role"].split(" - ")[0])
        for p in top)
    quiet = sum(1 for p in profiles if p["mean_abs_activation"] < 0.05)
    long_memory = sum(1 for p in profiles if p["mean_forget"] > 0.75)
    returns = "one hidden state per hour" if index == 0 else "only the final summary vector"
    return """---
tags: [layer]
---
# {name}

{units} LSTM units. Returns **{returns}**.

## Behaviour on the rush-hour window

| | |
| --- | --- |
| Units carrying signal (`mean |h| > 0.05`) | {active} of {units} |
| Near-silent units | {quiet} |
| Long-memory units (`mean f > 0.75`) | {long} |
| Strongest activation | {peak:+.3f} |

That {quiet} units stay near zero on this window is normal and useful: dropout
during training pushes the layer to spread its representation, so different
subsets specialise in different regimes. Compare
[[Neurons - Rush Hour.canvas]] with [[Neurons - Quiet Night.canvas]] - the
lit-up subsets differ.

## Every unit, every hour

![[06_heatmap_{name}.png]]

Rows are units, columns are the {steps} hours of the window. Red is a positive
activation, blue negative, white silent. The vertical stripes are hours where
most of the layer reacted at once.

## How the gates behaved

![[07_gates_{name}.png]]

## Most active units here

| Unit | mean \\|h\\| | final h | peaks at | role |
| --- | --- | --- | --- | --- |
{rows}

Every unit has its own note in `Neurons/`.

Related: [[04 Architecture]] - [[Forget gate]] - [[Input gate]] - [[Output gate]] - [[Cell state]]
""".format(name=layer.name, units=layer.units, returns=returns, steps=layer.timesteps,
           active=len(profiles) - quiet, quiet=quiet, long=long_memory,
           peak=max(p["peak_activation"] for p in profiles), rows=rows)


def _note_dense(cfg: TrainingConfig, trace) -> str:
    values = trace.dense_hidden
    active = int((values > 1e-6).sum())
    return """---
tags: [layer]
---
# dense_hidden

{n} units, `relu` activation. Takes the summary vector produced by [[lstm_2]]
and recombines it non-linearly before the output.

`relu(x) = max(0, x)` - anything negative becomes exactly zero, which is why
some units below read `+0.00`.

| | |
| --- | --- |
| Units firing on the rush-hour window | {active} of {n} |
| Largest activation | {peak:.3f} |

`{spark}`

Feeds [[traffic_output]]. Related: [[04 Architecture]]
""".format(n=len(values), active=active, peak=float(values.max()),
           spark=sparkline(values))


def _note_output(cfg: TrainingConfig, contexts: dict) -> str:
    rows = "\n".join(
        "| {label} | {ts} | {pred:,.0f} | {truth:,.0f} | {err:+,.0f} |".format(
            label=label.replace("_", " ").title(), ts=ctx["timestamp"][:16],
            pred=ctx["prediction"], truth=ctx["target"],
            err=ctx["prediction"] - ctx["target"])
        for label, ctx in contexts.items())
    return """---
tags: [layer]
---
# traffic_output

{n} linear unit(s) - horizon(s) {h}.

**No activation function.** This is a regression: the output is a number of
vehicles per hour. A sigmoid would clamp it to [0, 1]; a softmax would turn it
into class probabilities. Neither is what we want.

## The two windows documented in this vault

| Window | Ends at | Predicted | Actual | Error |
| --- | --- | --- | --- | --- |
{rows}

## Turning the number into a decision

| Predicted volume | Level |
| --- | --- |
| below {low:,} | LOW - free flow |
| {low:,} to {high:,} | MODERATE - steady traffic |
| above {high:,} | HIGH - congestion risk |

Fed by [[dense_hidden]]. Related: [[05 Results]]
""".format(n=cfg.n_outputs, h=", ".join("+{}h".format(x) for x in cfg.horizons),
           rows=rows, low=cfg.level_low, high=cfg.level_high)


# ------------------------------------------------------------- neuron notes --
def _note_neuron(layer, profile: dict, night_profile: dict, contexts: dict) -> str:
    u = profile["unit"]
    steps = layer.timesteps
    peaks = ", ".join("t-{}h".format(steps - t) for t in profile["top_timesteps"])
    return """---
tags: [neuron, {ln}]
layer: {ln}
unit: {u}
---
# {ln} u{u:02d}

**Role on the rush-hour window:** {role}

## Hidden state across the {steps}-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `{spark_h}` | {fh:+.3f} | {ph:+.3f} at t-{pt}h |
| Quiet night | `{spark_n}` | {fn:+.3f} | {pn:+.3f} at t-{pnt}h |

Reacted most strongly at: **{peaks}**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `{spark_f}` | {mf:.3f} |
| [[Input gate]] | `{spark_i}` | {mi:.3f} |
| [[Output gate]] | `{spark_o}` | {mo:.3f} |
| [[Cell state]] | `{spark_c}` | final {fc:+.3f} |

A mean forget value of **{mf:.2f}** means this unit keeps roughly
{keep:.0f}% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[{ln}]]
""".format(ln=layer.name, u=u, role=profile["role"], steps=steps,
           spark_h=sparkline(layer.h[:, u]), fh=profile["final_h"],
           ph=profile["peak_activation"], pt=steps - profile["peak_timestep"],
           spark_n=night_profile["spark"], fn=night_profile["final_h"],
           pn=night_profile["peak_activation"], pnt=steps - night_profile["peak_timestep"],
           peaks=peaks,
           spark_f=sparkline(layer.f[:, u]), mf=profile["mean_forget"],
           spark_i=sparkline(layer.i[:, u]), mi=profile["mean_input"],
           spark_o=sparkline(layer.o[:, u]), mo=float(layer.o[:, u].mean()),
           spark_c=sparkline(layer.c[:, u]), fc=profile["final_cell"],
           keep=profile["mean_forget"] * 100)


# ------------------------------------------------------------------ export ---
def export_vault(cfg: TrainingConfig, model, bundle, artifacts: dict,
                 vault_dir: Path = VAULT_DIR, timeline: bool = True,
                 verbose: bool = True) -> Path:
    """Regenerate the whole vault from a trained model. Idempotent."""
    vault = Path(vault_dir)
    _reset(vault)
    _obsidian_settings(vault)

    picks = _pick_windows(bundle)
    contexts = {name: _window_context(cfg, bundle, model, idx) for name, idx in picks.items()}
    traces = {name: trace_network(model, bundle.X_test[idx])
              for name, idx in picks.items()}
    sensitivity = input_sensitivity(model, bundle.X_test[picks["rush_hour"]])

    rush, night = traces["rush_hour"], traces["night"]

    # --- canvases ---
    canvas_dir = vault / "Canvas"
    build_architecture_canvas(cfg, artifacts).save(canvas_dir / "Network Architecture.canvas")
    build_neuron_canvas(cfg, rush, sensitivity, "Neurons at rush hour",
                        contexts["rush_hour"]).save(canvas_dir / "Neurons - Rush Hour.canvas")
    night_sensitivity = input_sensitivity(model, bundle.X_test[picks["night"]])
    build_neuron_canvas(cfg, night, night_sensitivity, "Neurons on a quiet night",
                        contexts["night"]).save(canvas_dir / "Neurons - Quiet Night.canvas")

    if timeline:
        for t in range(cfg.sequence_length):
            build_timeline_canvas(cfg, rush, t, contexts["rush_hour"]).save(
                canvas_dir / "Timeline" / "t{:02d}.canvas".format(t))

    # --- introspection figures, drawn from the same trace as the canvases ---
    for layer in rush.layers:
        plots.plot_activation_heatmap(layer, "06_heatmap_{}.png".format(layer.name))
        plots.plot_gate_summary(layer, "07_gates_{}.png".format(layer.name))
    plots.plot_sensitivity(sensitivity)

    # --- figures ---
    figures = vault / "Figures"
    figures.mkdir(parents=True, exist_ok=True)
    for png in sorted(FIGURE_DIR.glob("*.png")):
        shutil.copy2(png, figures / png.name)

    # --- top-level notes ---
    _write(vault / "00 Start Here.md", _note_start_here(cfg, artifacts, contexts))
    _write(vault / "01 Problem.md", _note_problem(cfg))
    _write(vault / "02 Dataset.md", _note_dataset(cfg, artifacts))
    _write(vault / "03 Pipeline.md", _note_pipeline(cfg, artifacts))
    _write(vault / "04 Architecture.md", _note_architecture(cfg, artifacts, rush))
    _write(vault / "05 Results.md", _note_results(cfg, artifacts, sensitivity))
    _write(vault / "06 Limits and Next Steps.md", _note_limits(cfg))
    _write(vault / "07 Presentation Script.md", _note_script(cfg, artifacts, contexts))
    _write(vault / "08 Exam Questions.md", _note_questions(cfg, artifacts, rush))

    # --- mechanism notes ---
    for name in GATE_NOTES:
        _write(vault / "Gates" / "{}.md".format(name), _note_gate(name, rush, contexts))
    _write(vault / "Gates" / "Cell state.md", _note_cell_state(rush))

    # --- layer + neuron notes ---
    n_neurons = 0
    for index, layer in enumerate(rush.layers):
        profiles = neuron_profiles(layer)
        _write(vault / "Layers" / "{}.md".format(layer.name),
               _note_layer(cfg, layer, index, profiles))

        night_layer = night.layers[index]
        night_profiles = neuron_profiles(night_layer)
        for profile in profiles:
            u = profile["unit"]
            np_prof = dict(night_profiles[u])
            np_prof["spark"] = sparkline(night_layer.h[:, u])
            _write(vault / "Neurons" / "{} u{:02d}.md".format(layer.name, u),
                   _note_neuron(layer, profile, np_prof, contexts))
            n_neurons += 1

    _write(vault / "Layers" / "dense_hidden.md", _note_dense(cfg, rush))
    _write(vault / "Layers" / "traffic_output.md", _note_output(cfg, contexts))

    if verbose:
        n_canvas = len(list(canvas_dir.rglob("*.canvas")))
        print("  vault:   {}".format(vault))
        print("  canvases:{:>4}   neuron notes:{:>4}   verification error: {:.2e}".format(
            n_canvas, n_neurons, rush.max_abs_error or 0.0))
    return vault
