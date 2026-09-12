"""Figures for the report and the slide deck.

Every function saves to `reports/figures/` and returns the path, so the
notebook, the training script and the Streamlit app all produce the exact
same images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in scripts, notebooks and Streamlit alike
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURE_DIR

PALETTE = {
    "actual": "#1f77b4",
    "predicted": "#ff7f0e",
    "train": "#2ca02c",
    "val": "#d62728",
    "grid": "#dddddd",
}


def _finish(fig, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_series(timestamps, values, hours: int = 500, filename: str = "01_dataset.png") -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps[:hours], values[:hours], color=PALETTE["actual"], linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Vehicles per hour")
    ax.set_title(f"Traffic volume - first {hours} hours of the dataset")
    ax.grid(color=PALETTE["grid"])
    fig.autofmt_xdate(rotation=45)
    return _finish(fig, filename)


def plot_daily_profile(series, datetime_column: str, target_column: str,
                       filename: str = "02_daily_profile.png") -> Path:
    """Average traffic by hour of day - the pattern the LSTM has to learn."""
    hourly = series.assign(hour=series[datetime_column].dt.hour).groupby("hour")[target_column]
    mean, std = hourly.mean(), hourly.std()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean.index, mean.values, color=PALETTE["actual"], marker="o")
    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=PALETTE["actual"])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average vehicles per hour")
    ax.set_title("Daily traffic cycle (mean +/- 1 standard deviation)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(color=PALETTE["grid"])
    return _finish(fig, filename)


def plot_history(history: dict, filename: str = "03_training_loss.png") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["loss"], label="Training", color=PALETTE["train"])
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Validation", color=PALETTE["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, scaled)")
    ax.set_title("Learning curve")
    ax.legend()
    ax.grid(color=PALETTE["grid"])
    return _finish(fig, filename)


def plot_predictions(actual, predicted, hours: int = 200, horizon: int = 1,
                     timestamps=None, filename: str = "04_actual_vs_predicted.png") -> Path:
    actual = np.asarray(actual).ravel()[:hours]
    predicted = np.asarray(predicted).ravel()[:hours]
    x = timestamps[:hours] if timestamps is not None else np.arange(len(actual))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, actual, label="Actual traffic", color=PALETTE["actual"], linewidth=1.6)
    ax.plot(x, predicted, label=f"Predicted (+{horizon}h)", color=PALETTE["predicted"],
            linewidth=1.6, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Vehicles per hour")
    ax.set_title(f"Actual vs predicted traffic - test set, horizon +{horizon}h")
    ax.legend()
    ax.grid(color=PALETTE["grid"])
    if timestamps is not None:
        fig.autofmt_xdate(rotation=45)
    return _finish(fig, filename)


def plot_error_distribution(actual, predicted, filename: str = "05_error_distribution.png") -> Path:
    errors = np.asarray(predicted).ravel() - np.asarray(actual).ravel()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=60, color=PALETTE["predicted"], edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Prediction error (vehicles)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error distribution - mean {errors.mean():.0f}, std {errors.std():.0f}")
    ax.grid(color=PALETTE["grid"])
    return _finish(fig, filename)


def plot_activation_heatmap(layer_trace, filename: str = "06_neuron_heatmap.png") -> Path:
    """The money shot: every neuron, every hour, one image.

    The colour scale is fitted to the data rather than pinned to the tanh
    bounds of [-1, 1]. Hidden states are usually well inside those bounds, and
    a fixed scale washes the whole picture out. The limit used is written into
    the title so the image cannot overstate how hard the units are firing.
    """
    values = layer_trace.h.T
    limit = float(np.percentile(np.abs(values), 99))
    limit = max(limit, 1e-3)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                   interpolation="nearest")
    last = layer_trace.timesteps - 1
    ax.set_xlabel(f"Hour of the input window (0 = {layer_trace.timesteps}h ago, "
                  f"{last} = most recent)")
    ax.set_ylabel(f"{layer_trace.name} unit")
    ax.set_title(f"Hidden state of every neuron in {layer_trace.name}, hour by hour "
                 f"(scale ±{limit:.2f})")
    fig.colorbar(im, ax=ax, label="Activation h")
    return _finish(fig, filename)


def plot_gate_summary(layer_trace, filename: str = "07_gates.png") -> Path:
    """Average gate opening over time - how the memory behaves across the day."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, color in (
        ("f", "Forget gate (keep memory)", "#8c564b"),
        ("i", "Input gate (write new)", "#2ca02c"),
        ("o", "Output gate (expose memory)", "#9467bd"),
    ):
        ax.plot(layer_trace.gate(key).mean(axis=1), label=label, color=color, marker="o",
                markersize=3)
    ax.set_xlabel("Hour of the input window")
    ax.set_ylabel("Mean gate value (0 = closed, 1 = open)")
    ax.set_title(f"Gate behaviour in {layer_trace.name}, averaged over all units")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(color=PALETTE["grid"])
    return _finish(fig, filename)


def plot_sensitivity(scores, filename: str = "08_input_sensitivity.png") -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(scores)), scores, color=PALETTE["predicted"])
    ax.set_xlabel("Hour of the input window (23 = most recent)")
    ax.set_ylabel("Share of the prediction moved")
    ax.set_title("Which past hours the model actually relies on")
    ax.grid(axis="y", color=PALETTE["grid"])
    return _finish(fig, filename)


def plot_baseline_comparison(comparison: dict, filename: str = "09_baselines.png") -> Path:
    labels = ["LSTM", "Same hour\nyesterday", "Last hour\n(persistence)"]
    values = [
        comparison["lstm"]["mae"],
        comparison["naive_same_hour_yesterday"]["mae"],
        comparison["naive_persistence"]["mae"],
    ]
    colors = [PALETTE["predicted"], "#7f7f7f", "#bbbbbb"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("MAE (vehicles per hour) - lower is better")
    ax.set_title("Is the network actually better than doing nothing clever?")
    ax.grid(axis="y", color=PALETTE["grid"])
    return _finish(fig, filename)


def plot_model_comparison(rows, filename: str = "10_model_comparison.png",
                          unit: str = "vehicles per hour",
                          title: str = "Every model tried, on the same test windows") -> Path:
    """Every trained variant and every baseline, ranked by MAE.

    `rows` is a list of (label, mae, kind) where kind is one of
    "lstm", "xgboost" or "naive" - the colour carries the family.

    One chart per dataset, always: MAE is in the units of whatever is being
    predicted, so putting two datasets on one axis would be meaningless.
    """
    rows = sorted(rows, key=lambda r: r[1])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = {"lstm": PALETTE["predicted"], "xgboost": "#2ca02c", "naive": "#bbbbbb"}
    bars_colour = [colours.get(r[2], "#888888") for r in rows]

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 2.2))
    bars = ax.barh(labels, values, color=bars_colour)
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,.0f}",
                va="center", fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel(f"MAE ({unit}) - lower is better")
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", color=PALETTE["grid"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (colours["lstm"], colours["xgboost"], colours["naive"])]
    # Bars are sorted shortest-first, so the top-right corner is the empty one.
    ax.legend(handles, ["LSTM", "XGBoost", "Naive baseline"], loc="upper right",
              framealpha=0.95)
    return _finish(fig, filename)


def plot_horizon_degradation(horizons, maes, filename: str = "11_horizons.png") -> Path:
    """How fast accuracy decays as the forecast reaches further ahead."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(horizons, maes, marker="o", markersize=9, linewidth=2.5,
            color=PALETTE["predicted"])
    for h, m in zip(horizons, maes):
        ax.annotate(f"{m:,.0f}", (h, m), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontweight="bold")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("MAE (vehicles per hour)")
    ax.set_title("Accuracy decays with the forecast horizon")
    ax.set_xticks(list(horizons))
    ax.set_ylim(0, max(maes) * 1.25)
    ax.grid(color=PALETTE["grid"])
    return _finish(fig, filename)
