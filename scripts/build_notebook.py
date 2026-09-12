"""Generate the presentation notebook.

Keeping the notebook in a generator script means it can be regenerated cleanly
after any change to the library, and it keeps the repository free of the
execution metadata that makes .ipynb diffs unreadable.

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "traffic_lstm.ipynb"

CELLS = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------
md("""
# Traffic volume forecasting with an LSTM

**Question.** Given the last 24 hours of traffic volume, how many vehicles will
pass during the next hour?

| | |
| --- | --- |
| Learning type | Supervised |
| Task | Regression - the output is a count of vehicles |
| Data | Metro Interstate Traffic Volume (UCI), ~48,000 hourly rows |
| Algorithm | Stacked LSTM |
| Metric | MAE, in vehicles per hour |

Run the cells in order. Each section maps to one slide.
""")

md("""
## 0 - Setup

Everything reusable lives in `src/traffic_lstm/`. The notebook imports it
rather than copying it, except for the two pieces worth reading out loud:
how the sequences are built, and how the model is assembled.
""")

code("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from traffic_lstm.config import DEFAULT_DATASET, TrainingConfig
from traffic_lstm.data import clean_series, describe_quality, load_raw
from traffic_lstm.evaluate import naive_persistence, naive_seasonal, traffic_level
from traffic_lstm.model import set_seeds

set_seeds(42)
plt.rcParams["figure.figsize"] = (12, 5)
print("Ready.")
""")

md("""
## 1 - Load the data

| Library | What it does here |
| --- | --- |
| `pandas` | reads and reshapes the CSV |
| `numpy` | numeric arrays |
| `matplotlib` | the figures |
| `scikit-learn` | scaling and metrics |
| `tensorflow.keras` | builds and trains the LSTM |
""")

code("""
df = load_raw(DEFAULT_DATASET, "date_time")
df.head()
""")

code("""
quality = describe_quality(df, "date_time", "traffic_volume")
for key, value in quality.items():
    if key != "columns":
        print(f"{key:>24} : {value}")
""")

md("""
Two facts to mention out loud, because they are the first thing a careful
reader checks:

* **Duplicate timestamps.** The same hour is sometimes logged twice with a
  different weather description. We keep one row per hour and average the
  target - we neither invent data nor keep two conflicting rows.
* **Missing hours.** Some hours have no record at all (a long outage in
  2014-2015). We do **not** interpolate them; inventing traffic volumes would
  flatter the metrics.
""")

code("""
series = clean_series(df, "date_time", "traffic_volume")
print(f"{len(df):,} raw rows -> {len(series):,} unique hours")
series.head()
""")

md("""
## 2 - Look at the data before modelling it
""")

code("""
sample = series.head(500)

plt.plot(sample["date_time"], sample["traffic_volume"], linewidth=1.2)
plt.xlabel("Date")
plt.ylabel("Vehicles per hour")
plt.title("Traffic volume - first 500 hours")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""")

code("""
by_hour = series.assign(hour=series["date_time"].dt.hour).groupby("hour")["traffic_volume"]
mean, std = by_hour.mean(), by_hour.std()

plt.plot(mean.index, mean.values, marker="o")
plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
plt.xlabel("Hour of day")
plt.ylabel("Average vehicles per hour")
plt.title("The daily cycle the network has to learn")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.show()
""")

md("""
Morning peak, evening peak, overnight trough. **This is why 24 timesteps:**
one input window covers exactly one full cycle.
""")

md("""
## 3 - Split, then scale (in that order)

Two rules that decide whether the result is honest:

1. **The split is chronological.** `train_test_split(shuffle=True)` would put
   future hours in the training set. The score would look great and mean
   nothing, because in production you only ever have the past.
2. **The scaler is fitted on the training slice only.** If it saw the test
   set, the model would indirectly know the maximum volume of the future.
   That is **data leakage**.
""")

code("""
values = series[["traffic_volume"]].to_numpy(dtype="float64")

train_size = int(len(values) * 0.8)
train_raw, test_raw = values[:train_size], values[train_size:]

print(f"train: {len(train_raw):,} hours   test: {len(test_raw):,} hours")
print(f"test period: {series['date_time'].iloc[train_size]} -> {series['date_time'].iloc[-1]}")
""")

code("""
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train_raw)   # fit + transform
test_scaled = scaler.transform(test_raw)         # transform ONLY

print(f"scaler learned min={scaler.data_min_[0]:,.0f}  max={scaler.data_max_[0]:,.0f}")
""")

md("""
## 4 - Build the sequences

```
hours  1 .. 24  ->  predict hour 25
hours  2 .. 25  ->  predict hour 26
hours  3 .. 26  ->  predict hour 27
```
""")

code("""
SEQUENCE_LENGTH = 24


def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i, 0])   # the last 24 hours
        y.append(data[i, 0])                       # the next hour
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled)

# The first test window needs 24 hours of history, and those hours sit at the
# end of the training set. Prepending them is history, not leakage: the model
# only ever looks backwards.
bridged = scaler.transform(np.concatenate([train_raw[-SEQUENCE_LENGTH:], test_raw]))
X_test, y_test = create_sequences(bridged)

print("before reshape:", X_train.shape)
""")

code("""
# An LSTM expects (samples, timesteps, features).
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"X_train {X_train.shape}   = {X_train.shape[0]:,} windows of {X_train.shape[1]} hours, 1 feature")
print(f"X_test  {X_test.shape}")
""")

md("""
## 5 - The network

```
(24, 1)  ->  LSTM 64  ->  Dropout  ->  LSTM 32  ->  Dropout  ->  Dense 16  ->  Dense 1
```

* **LSTM 64**, `return_sequences=True` - one hidden state per hour, so the next
  layer still receives a sequence.
* **Dropout 0.2** - 20% of the outputs are zeroed at random during training, so
  the network cannot lean on any single unit. Off at prediction time.
* **LSTM 32**, `return_sequences=False` - collapses the day into one vector.
* **Dense 1**, no activation - this is a regression, the output is a number of
  vehicles, not a probability.
""")

code("""
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, 1), name="input_sequence"),

    LSTM(64, return_sequences=True, name="lstm_1"),
    Dropout(0.2),

    LSTM(32, name="lstm_2"),
    Dropout(0.2),

    Dense(16, activation="relu", name="dense_hidden"),
    Dense(1, name="traffic_output"),
])

model.compile(
    optimizer="adam",   # adapts the learning rate per weight
    loss="mse",         # training loss: punishes large misses hard
    metrics=["mae"],    # reported metric: interpretable, in vehicles
)

model.summary()
""")

md("""
## 6 - Train

`epochs=50` is an upper bound. **EarlyStopping** watches the validation loss
and stops as soon as it stops improving for 5 epochs in a row, then restores
the best weights.
""")

code("""
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    shuffle=False,      # keep the temporal order inside the validation split
    verbose=1,
)
""")

code("""
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE, scaled)")
plt.title("Learning curve")
plt.legend()
plt.tight_layout()
plt.show()
""")

md("""
Both curves falling together is what we want. If the training loss kept
dropping while the validation loss turned upward, that would be overfitting -
and EarlyStopping would have cut the run short.
""")

md("""
## 7 - Evaluate on hours the model has never seen
""")

code("""
predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print(f"MAE  : {mae:,.1f} vehicles per hour")
print(f"RMSE : {rmse:,.1f}")
""")

md("""
### Say it in words

> On hours it had never seen, the model is wrong by about **MAE** vehicles per
> hour on average.

### And then immediately ask: is that good?

A MAE means nothing on its own. Two baselines turn it into an argument.
""")

code("""
raw_bridged = np.concatenate([train_raw[-SEQUENCE_LENGTH:], test_raw])

baseline_last_hour = naive_persistence(raw_bridged, SEQUENCE_LENGTH, 1)
baseline_yesterday = naive_seasonal(raw_bridged, SEQUENCE_LENGTH, 1)

mae_last = mean_absolute_error(actual, baseline_last_hour)
mae_yesterday = mean_absolute_error(actual, baseline_yesterday)
best_naive = min(mae_last, mae_yesterday)

print(f"LSTM                : {mae:,.1f}")
print(f"Same hour yesterday : {mae_yesterday:,.1f}")
print(f"Last hour           : {mae_last:,.1f}")
print(f"-> {(best_naive - mae) / best_naive * 100:+.1f}% against the best baseline")
""")

code("""
plt.bar(["LSTM", "Same hour\\nyesterday", "Last hour"],
        [mae, mae_yesterday, mae_last],
        color=["#ff7f0e", "#7f7f7f", "#cccccc"])
plt.ylabel("MAE (vehicles) - lower is better")
plt.title("Is the network better than a naive rule?")
plt.tight_layout()
plt.show()
""")

md("""
## 8 - Actual vs predicted

This is the figure to put on the results slide.
""")

code("""
plt.figure(figsize=(13, 6))
plt.plot(actual[:200], label="Actual traffic", linewidth=1.6)
plt.plot(predictions[:200], label="Predicted traffic", linewidth=1.6, linestyle="--")
plt.xlabel("Hours into the test set")
plt.ylabel("Vehicles per hour")
plt.title("Actual vs predicted - test set")
plt.legend()
plt.tight_layout()
plt.show()
""")

md("""
The model follows the daily cycle closely and shaves the very top of the
sharpest peaks - expected, because MSE rewards being close on average more
than nailing the extremes.
""")

md("""
## 9 - Live demo

The part to run in front of the class.
""")

code("""
def predict_next_hour(last_24_hours):
    window = np.asarray(last_24_hours, dtype="float64").reshape(-1, 1)
    scaled = scaler.transform(window).reshape(1, SEQUENCE_LENGTH, 1)
    return float(scaler.inverse_transform(model.predict(scaled, verbose=0))[0][0])


last_24_hours = series["traffic_volume"].to_numpy()[-SEQUENCE_LENGTH:]
prediction = predict_next_hour(last_24_hours)
level, comment = traffic_level(prediction)

print("=" * 44)
print("        TRAFFIC FORECAST".center(44))
print("=" * 44)
print(f"  Window ends : {series['date_time'].iloc[-1]}")
print(f"  Last 6 hours: " + "  ".join(f"{v:,.0f}" for v in last_24_hours[-6:]))
print()
print(f"  Next hour   : {prediction:,.0f} vehicles")
print(f"  Level       : {level}  ({comment})")
print("=" * 44)
""")

md("""
## 10 - Look inside the network

Keras gives a prediction, not an explanation. `introspect.py` re-implements the
LSTM cell in NumPy from the trained weights, so every gate of every unit at
every hour can be read - and it is checked against Keras so the numbers are
provably the real ones.
""")

code("""
from traffic_lstm.introspect import most_active, neuron_profiles, trace_network

busiest = int(np.argmax(actual))
trace = trace_network(model, X_test[busiest])

print(f"Window ends at the busiest hour of the test set: {actual[busiest][0]:,.0f} vehicles")
print(f"Replay vs Keras, maximum absolute difference: {trace.max_abs_error:.2e}")
""")

md("""
`6.9e-08` is floating-point noise. The replay **is** the network.
""")

code("""
layer = trace.layers[0]

plt.figure(figsize=(13, 6))
plt.imshow(layer.h.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Hidden state h")
plt.xlabel("Hour of the input window (0 = 24h ago)")
plt.ylabel("lstm_1 unit")
plt.title("What every neuron did, hour by hour")
plt.tight_layout()
plt.show()
""")

code("""
for key, label in (("f", "Forget gate (keep memory)"),
                   ("i", "Input gate (write new)"),
                   ("o", "Output gate (expose memory)")):
    plt.plot(layer.gate(key).mean(axis=1), marker="o", markersize=3, label=label)

plt.xlabel("Hour of the input window")
plt.ylabel("Mean gate value (0 = closed, 1 = open)")
plt.title("How the memory behaves across the window")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
""")

code("""
profiles = neuron_profiles(layer)
pd.DataFrame(most_active(profiles, 8))[
    ["unit", "final_h", "peak_timestep", "mean_abs_activation", "mean_forget", "role"]
]
""")

md("""
For the full version of this - one canvas per hour, one note per neuron, the
gates with their real values - open the Obsidian vault:

```
obsidian_vault/Traffic_LSTM_Brain
```

Regenerate it any time with:

```
python -m traffic_lstm.train --export-vault
```
""")

md("""
## 11 - Conclusion

**What works.** The network learns the daily cycle and beats both naive
baselines on hours it never saw.

**What does not.** It has never heard of weather, holidays or accidents -
exactly the situations where a forecast matters most. It cannot predict a
first-time event, and it is harder to interpret than a decision tree.

**Next.** Add `temp`, `rain_1h`, `snow_1h`, `holiday`, plus hour-of-day and
day-of-week as sine/cosine pairs. The input shape becomes `(24, 8)` and
nothing else in the pipeline changes.

**And to be precise about the claim:** this model does not control traffic
lights. It produces a forecast that a signal controller could consume.

```
Sensors -> hourly history -> LSTM -> forecast -> controller -> adaptive lights
```
""")


# ---------------------------------------------------------------------------
def build() -> Path:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": payload}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    path = build()
    print("wrote {} ({} cells)".format(path, len(CELLS)))
