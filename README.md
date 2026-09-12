# Traffic LSTM

Forecast the next hour of motorway traffic from the last 24 — and then open the
network up and look at what every neuron actually did.

```
24 past hours  ->  LSTM 64  ->  LSTM 32  ->  Dense 16  ->  vehicles next hour
```

Three ways to use it:

| | |
| --- | --- |
| **Notebook** | `notebooks/traffic_lstm.ipynb` — run cell by cell, built for a presentation |
| **App** | `streamlit run app/streamlit_app.py` — load any CSV, train it, inspect the neurons |
| **Obsidian vault** | `obsidian_vault/Traffic_LSTM_Brain` — the trained network as a browsable set of canvases and notes |
| **Written up** | [`reports/REPORT.md`](reports/REPORT.md) and [`reports/web/results.html`](reports/web/results.html), both generated from the stored run artifacts |

---

## The project in one paragraph

Traffic volume is a time series: the number of vehicles in the next hour
depends on the hours before it. An LSTM is a recurrent network built for
exactly that — it walks through a sequence carrying a memory it can choose to
keep or overwrite. This project trains one on ~48,000 hourly measurements from
the UCI *Metro Interstate Traffic Volume* dataset, evaluates it honestly
against two naive baselines, and then reconstructs every gate of every unit so
the mechanism can be seen rather than asserted.

| | |
| --- | --- |
| Learning type | Supervised |
| Task | Regression — the output is a count of vehicles |
| Algorithm | Stacked LSTM (64 → 32 units) |
| Input | 24 timesteps × 1 feature |
| Metric | MAE, in vehicles per hour |
| Baselines | "last hour" and "same hour yesterday" |

---

## Install

TensorFlow needs **Python 3.12** (no Windows wheels for 3.13 yet).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `.ps1` scripts use the project-local `.venv` created above. For backward
compatibility, they also recognize an older environment at
`~\.venvs\traffic_lstm`.

## Run

```powershell
.\train.ps1              # train, draw the figures, rebuild the vault
.\run_app.ps1            # the interactive workbench
.\notebook.ps1           # JupyterLab, for the presentation
.\test.ps1               # verify the claims made below
```

`test.ps1` checks the core pipeline: the scaler never sees the test set, every
`(X, y)` pair lines up with the timestamps it claims, and the NumPy replay of
the LSTM cell reproduces Keras exactly. GitHub Actions additionally loads every
committed model and smoke-tests the Streamlit interface on each push and pull
request.

Or without the wrappers:

```powershell
$env:PYTHONPATH = "src"

# train
python -m traffic_lstm.train --export-vault
python -m traffic_lstm.train --data data/raw/my_file.csv --target my_column --epochs 30

# the variants
python -m traffic_lstm.train --run-name multivariate --calendar `
       --exogenous temp rain_1h snow_1h clouds_all
python -m traffic_lstm.train --run-name gap_guarded --drop-gapped-windows
python -m traffic_lstm.train --run-name multi_horizon --horizons 1 3 6

# is the LSTM worth it? measure, do not argue
python -m traffic_lstm.benchmark --run-name baseline_univariate
python scripts/compare_runs.py          # ranks every model, writes the vault note

# use it
python -m traffic_lstm.predict --last-hours
streamlit run app/streamlit_app.py
python scripts/rebuild_vault.py         # regenerate the vault without retraining
```

---

## Using your own dataset

Any CSV with **a timestamp column and a numeric column** works — energy
consumption, website traffic, sales per hour. Either:

* drop it in `data/raw/` and pass `--data` / `--target` to the training
  command, or
* upload it on the **Data** tab of the Streamlit app and pick the two columns
  from the dropdowns.

The pipeline sorts chronologically, collapses duplicate timestamps by
averaging, and never interpolates missing periods.

---

## What is in `src/traffic_lstm/`

| Module | Responsibility |
| --- | --- |
| `config.py` | every hyper-parameter in one dataclass, with the reasoning attached |
| `data.py` | CSV → cleaned series → chronological split → scaling → sequences |
| `features.py` | weather + cyclical calendar features; the target stays column 0 |
| `benchmark.py` | XGBoost on the **same tensors**, flattened — the honest comparison |
| `model.py` | the stacked LSTM, the optimiser, the callbacks |
| `train.py` | orchestration, artifacts, the CLI |
| `evaluate.py` | MAE / RMSE / MAPE **and** the two naive baselines |
| `predict.py` | load a saved model and forecast from 24 values |
| `introspect.py` | replays the LSTM cell in NumPy to recover every gate |
| `plots.py` | the figures used by the report and the slides |
| `obsidian_canvas.py` | JSON Canvas builders |
| `obsidian_export.py` | generates the whole vault from a trained model |

### Three decisions worth defending

**The split is chronological.** `train_test_split(shuffle=True)` would put
future hours in the training set. The score would look excellent and mean
nothing, because in production you only ever have the past.

**The scaler is fitted on the training slice only.** If it saw the test set,
the model would indirectly know the range of the future — data leakage.

**Gaps in the series are disclosed, not hidden.** This dataset has 2,588
breaks in time, including a 308-day outage. Sliding windows are built over the
rows that exist, so **28.7% of the training windows silently span a jump** —
"the last 24 hours" is really 24 readings spread over longer. Only 5.3% of the
*test* windows are affected, so the reported metrics barely move, but the flaw
is real. `--drop-gapped-windows` discards those windows (21,185 training
windows instead of 32,436); it is off by default so the documented run stays
reproducible. Turning it on and retraining is the first thing to try.

**The baselines are not optional.** A MAE of ~300 vehicles is meaningless
until you know that predicting *"the same hour yesterday"* scores worse. If
the network had not beaten both naive rules, the honest conclusion would have
been that an LSTM is the wrong tool for this problem.

---

## The Obsidian vault

**Open folder as vault** → select `obsidian_vault/Traffic_LSTM_Brain` → start
from `00 Start Here`.

```
00 Start Here                     the map
01 Problem  02 Dataset  03 Pipeline  04 Architecture
05 Results  06 Limits and Next Steps
07 Presentation Script            a timed 8-minute script
08 Exam Questions                 the questions a teacher will ask, answered

Canvas/Network Architecture.canvas    the model
Canvas/Neurons - Rush Hour.canvas     every unit, coloured by its activation
Canvas/Neurons - Quiet Night.canvas   the same units on an empty road
Canvas/Timeline/t00 … t23.canvas      one canvas per hour: the memory forming

Layers/    one note per layer, with its behaviour on a real window
Gates/     forget, input, candidate, output, cell state — with this model's values
Neurons/   one note per unit (96 of them), each with its own trace
```

**None of it is illustrative.** `introspect.py` re-implements the LSTM cell in
NumPy from the trained weights:

```
z = x_t · W + h_(t-1) · U + b
i = σ(z_i)   f = σ(z_f)   g = tanh(z_g)   o = σ(z_o)
c_t = f · c_(t-1) + i · g
h_t = o · tanh(c_t)
```

and checks the result against Keras on every export. The current maximum
absolute difference is printed at the end of each run (~`7e-08`, i.e.
floating-point noise). If the replay and Keras disagreed, the vault would be
fiction — so the check is part of the build, not an afterthought.

Two contrasting windows are documented side by side: the busiest hour of the
test set and the quietest. Comparing the two canvases shows that different
subsets of units light up in different traffic regimes — the network has
specialised.

---

## Limits

* The model sees **only past traffic**. Weather, holidays and accidents are
  invisible to it, and those are exactly the hours when a forecast matters
  most. The dataset ships those columns; using them is the first item on the
  list below.
* It cannot predict a first-time event.
* It is harder to interpret than a decision tree — this vault is an attempt to
  narrow that gap, not to close it.
* With no temporal structure, or with little data, gradient boosting on lag
  features would be simpler and often as strong.

## Next steps

1. Use rolling-origin cross-validation instead of relying on one chronological
   holdout period.
2. Tune the LSTM against the already-supported multivariate and XGBoost runs;
   on the current motorway data, XGBoost is both faster and more accurate.
3. Add prediction intervals so the forecast communicates uncertainty.
4. Test longer seasonal windows (48, 72 and 168 hours) and direct multi-horizon
   forecasts with `--horizons 1 3 6`.
5. Feed the forecast to a signal-timing optimiser only after monitoring drift
   and defining a safety fallback.

**To be precise about the claim:** this model does not control traffic lights.
It produces a forecast that a control system could consume.

```
sensors → hourly history → LSTM → forecast → controller → adaptive lights
```

---

## Data

Metro Interstate Traffic Volume — UCI Machine Learning Repository.
Hourly westbound traffic on I-94 between Minneapolis and Saint Paul,
2012-10-02 to 2018-09-30. 48,204 rows; 40,575 unique hours after duplicates
are collapsed.
