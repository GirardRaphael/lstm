"""Traffic LSTM - interactive workbench.

Load any hourly CSV, train the network on it, watch the loss fall live, then
open the model up and look at what every neuron did.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable whichever directory Streamlit was launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from traffic_lstm.config import DEFAULT_DATASET, TrainingConfig
from traffic_lstm.data import build_datasets, clean_series, inverse_target, load_raw
from traffic_lstm.evaluate import traffic_level
from traffic_lstm.introspect import (
    input_sensitivity,
    most_active,
    neuron_profiles,
    trace_network,
)

st.set_page_config(page_title="Traffic LSTM", page_icon="🚦", layout="wide")

ACCENT = "#d62728"
BLUE = "#1f77b4"
GREY = "#8c8c8c"


# --------------------------------------------------------------- helpers ----
@st.cache_data(show_spinner=False)
def read_csv(source, datetime_column: str) -> pd.DataFrame:
    return load_raw(source, datetime_column)


def state(key, default=None):
    return st.session_state.get(key, default)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


class StreamlitProgress:
    """Keras callback that draws the learning curve while it trains."""

    def __init__(self, total_epochs: int):
        from tensorflow.keras.callbacks import Callback

        self.bar = st.progress(0.0, text="Starting…")
        self.chart = st.empty()
        self.losses, self.val_losses = [], []
        outer = self

        class _Callback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                outer.losses.append(float(logs.get("loss", np.nan)))
                outer.val_losses.append(float(logs.get("val_loss", np.nan)))
                done = (epoch + 1) / total_epochs
                outer.bar.progress(
                    min(done, 1.0),
                    text="Epoch {}/{} — loss {:.5f} · val_loss {:.5f}".format(
                        epoch + 1, total_epochs, outer.losses[-1], outer.val_losses[-1]))
                fig = go.Figure()
                fig.add_scatter(y=outer.losses, name="Training", line=dict(color=BLUE))
                fig.add_scatter(y=outer.val_losses, name="Validation", line=dict(color=ACCENT))
                fig.update_layout(height=320, margin=dict(t=30, b=10),
                                  xaxis_title="Epoch", yaxis_title="Loss (MSE, scaled)",
                                  title="Learning curve")
                outer.chart.plotly_chart(fig, width="stretch")

        self.callback = _Callback()


# ------------------------------------------------------------------ pages ---
def page_data():
    st.header("1 · Data")
    st.caption("Any CSV with a timestamp column and a numeric column works — "
               "not just the bundled traffic dataset.")

    choice = st.radio("Source", ["Bundled dataset (Metro Interstate)", "Upload a CSV"],
                      horizontal=True)

    if choice.startswith("Upload"):
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is None:
            st.info("Waiting for a file.")
            return
        source = uploaded
    else:
        if not DEFAULT_DATASET.exists():
            st.error("Bundled dataset not found at {}".format(DEFAULT_DATASET))
            return
        source = DEFAULT_DATASET

    peek = pd.read_csv(source, nrows=5)
    if hasattr(source, "seek"):
        source.seek(0)
    columns = list(peek.columns)

    col1, col2 = st.columns(2)
    datetime_column = col1.selectbox(
        "Timestamp column", columns,
        index=columns.index("date_time") if "date_time" in columns else 0)
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(peek[c])] or columns
    target_column = col2.selectbox(
        "Column to predict", numeric,
        index=numeric.index("traffic_volume") if "traffic_volume" in numeric else 0)

    try:
        raw = read_csv(source, datetime_column)
    except Exception as exc:  # surfacing the real parsing error is more useful than a stack trace
        st.error("Could not read that file: {}".format(exc))
        return

    series = clean_series(raw, datetime_column, target_column)
    st.session_state.update(raw=raw, series=series, datetime_column=datetime_column,
                            target_column=target_column)

    metric_row([
        ("Raw rows", "{:,}".format(len(raw)), "Rows in the file as delivered"),
        ("Unique hours", "{:,}".format(len(series)), "After collapsing duplicate timestamps"),
        ("Duplicates merged", "{:,}".format(len(raw) - len(series)),
         "Same hour logged more than once — averaged, not dropped at random"),
        ("Range", "{:,.0f} → {:,.0f}".format(series[target_column].min(),
                                             series[target_column].max()), "Target range"),
    ])

    st.dataframe(raw.head(8), width="stretch")

    fig = go.Figure()
    sample = series.head(700)
    fig.add_scatter(x=sample[datetime_column], y=sample[target_column],
                    line=dict(color=BLUE, width=1.4), name=target_column)
    fig.update_layout(height=340, margin=dict(t=40, b=10),
                      title="First 700 hours", xaxis_title="", yaxis_title=target_column)
    st.plotly_chart(fig, width="stretch")

    hourly = series.assign(hour=series[datetime_column].dt.hour).groupby("hour")[target_column]
    mean, std = hourly.mean(), hourly.std()
    fig2 = go.Figure()
    fig2.add_scatter(x=mean.index, y=mean + std, line=dict(width=0), showlegend=False)
    fig2.add_scatter(x=mean.index, y=mean - std, fill="tonexty", line=dict(width=0),
                     fillcolor="rgba(31,119,180,0.18)", name="±1 std")
    fig2.add_scatter(x=mean.index, y=mean, line=dict(color=BLUE, width=3), name="Mean")
    fig2.update_layout(height=340, margin=dict(t=40, b=10),
                       title="Average by hour of day — the cycle the network must learn",
                       xaxis_title="Hour of day", yaxis_title=target_column)
    st.plotly_chart(fig2, width="stretch")


def page_train():
    st.header("2 · Train")
    if state("series") is None:
        st.warning("Load a dataset on the **Data** tab first.")
        return

    with st.form("hyperparameters"):
        c1, c2, c3, c4 = st.columns(4)
        sequence_length = c1.number_input("Input window (hours)", 6, 168, 24, step=1,
                                          help="24 = one full daily cycle.")
        units_1 = c2.number_input("LSTM layer 1 units", 8, 256, 64, step=8)
        units_2 = c3.number_input("LSTM layer 2 units", 4, 128, 32, step=4)
        dropout = c4.slider("Dropout", 0.0, 0.6, 0.2, 0.05,
                            help="Share of connections switched off during training.")
        c5, c6, c7, c8 = st.columns(4)
        epochs = c5.number_input("Max epochs", 1, 200, 30, step=1)
        batch_size = c6.selectbox("Batch size", [16, 32, 64, 128], index=1)
        patience = c7.number_input("EarlyStopping patience", 1, 30, 5, step=1)
        train_ratio = c8.slider("Train share", 0.5, 0.95, 0.8, 0.05)
        horizons = st.multiselect("Forecast horizons (hours ahead)", [1, 2, 3, 6, 12, 24],
                                  default=[1])
        submitted = st.form_submit_button("Train the network", type="primary")

    if not submitted:
        if state("outcome"):
            st.success("A trained model is already loaded — see **Results**.")
        return
    if not horizons:
        st.error("Pick at least one horizon.")
        return

    cfg = TrainingConfig(
        datetime_column=state("datetime_column"), target_column=state("target_column"),
        sequence_length=int(sequence_length), horizons=tuple(sorted(horizons)),
        lstm_units=(int(units_1), int(units_2)), dropout=float(dropout),
        epochs=int(epochs), batch_size=int(batch_size), patience=int(patience),
        train_ratio=float(train_ratio), run_name="streamlit_run")

    from traffic_lstm.train import train

    with st.spinner("Preparing sequences…"):
        bundle = build_datasets(cfg, df=state("raw"))
    st.caption("{:,} training sequences · {:,} test sequences · input shape ({}, 1)".format(
        len(bundle.X_train), len(bundle.X_test), cfg.sequence_length))

    progress = StreamlitProgress(cfg.epochs)
    import traffic_lstm.model as model_module

    original = model_module.default_callbacks
    model_module.default_callbacks = lambda c: original(c) + [progress.callback]
    try:
        outcome = train(cfg, bundle=bundle, verbose=0)
    finally:
        model_module.default_callbacks = original

    progress.bar.progress(1.0, text="Done — {} epochs".format(
        outcome["artifacts"]["epochs_run"]))
    st.session_state.update(cfg=cfg, outcome=outcome)
    st.success("Trained in {}s. Open **Results**.".format(
        outcome["artifacts"]["training_seconds"]))


def page_results():
    st.header("3 · Results")
    outcome = state("outcome")
    if not outcome:
        st.warning("Train a model first.")
        return
    cfg, artifacts = state("cfg"), outcome["artifacts"]

    horizon = st.selectbox("Horizon", list(cfg.horizons),
                           format_func=lambda h: "+{}h".format(h))
    k = list(cfg.horizons).index(horizon)
    block = artifacts["results"]["h{}".format(horizon)]
    lstm, seasonal, persistence = (block["lstm"], block["naive_same_hour_yesterday"],
                                   block["naive_persistence"])

    metric_row([
        ("MAE", "{:,.0f}".format(lstm["mae"]), "Average error, in units of the target"),
        ("RMSE", "{:,.0f}".format(lstm["rmse"]), "Penalises large misses harder"),
        ("MAPE", "{:.1f}%".format(lstm["mape"]), "Average error as a percentage"),
        ("vs best baseline", "{:+.1f}%".format(block["improvement_over_best_naive_pct"]),
         "Negative means a naive rule beats the network"),
    ])

    st.markdown(
        "> The model is wrong by about **{:,.0f}** on average. "
        "Predicting *the same hour yesterday* gives **{:,.0f}**, and *the last hour* "
        "gives **{:,.0f}**. Without those two numbers the MAE means nothing."
        .format(lstm["mae"], seasonal["mae"], persistence["mae"]))

    fig = go.Figure()
    fig.add_bar(x=["LSTM", "Same hour yesterday", "Last hour"],
                y=[lstm["mae"], seasonal["mae"], persistence["mae"]],
                marker_color=[ACCENT, GREY, "#cccccc"],
                text=["{:,.0f}".format(v) for v in
                      (lstm["mae"], seasonal["mae"], persistence["mae"])],
                textposition="outside")
    fig.update_layout(height=330, margin=dict(t=40, b=10), yaxis_title="MAE (lower is better)",
                      title="Is the network better than a naive rule?")
    st.plotly_chart(fig, width="stretch")

    hours = st.slider("Hours of the test set to display", 50, 1000, 250, 50)
    actual = outcome["actual"][:hours, k]
    predicted = outcome["predicted"][:hours, k]
    timestamps = outcome["bundle"].test_timestamps.to_numpy()[:hours]

    fig2 = go.Figure()
    fig2.add_scatter(x=timestamps, y=actual, name="Actual", line=dict(color=BLUE, width=1.8))
    fig2.add_scatter(x=timestamps, y=predicted, name="Predicted (+{}h)".format(horizon),
                     line=dict(color=ACCENT, width=1.8, dash="dash"))
    fig2.update_layout(height=430, margin=dict(t=40, b=10), title="Actual vs predicted",
                       yaxis_title=cfg.target_column)
    st.plotly_chart(fig2, width="stretch")

    errors = outcome["predicted"][:, k] - outcome["actual"][:, k]
    fig3 = go.Figure()
    fig3.add_histogram(x=errors, nbinsx=70, marker_color=ACCENT)
    fig3.update_layout(height=320, margin=dict(t=40, b=10),
                       title="Error distribution — mean {:+,.0f}, std {:,.0f}".format(
                           errors.mean(), errors.std()),
                       xaxis_title="Prediction error")
    st.plotly_chart(fig3, width="stretch")


def page_neurons():
    st.header("4 · Inside the network")
    outcome = state("outcome")
    if not outcome:
        st.warning("Train a model first.")
        return
    cfg, bundle, model = state("cfg"), outcome["bundle"], outcome["model"]

    actual = outcome["actual"][:, 0]
    presets = {
        "Busiest hour of the test set": int(np.argmax(actual)),
        "Quietest hour of the test set": int(np.argmin(actual)),
        "Largest model error": int(np.argmax(np.abs(outcome["predicted"][:, 0] - actual))),
    }
    choice = st.selectbox("Which window to look inside", list(presets) + ["Pick manually"])
    index = (st.slider("Test window index", 0, len(bundle.X_test) - 1, 0)
             if choice == "Pick manually" else presets[choice])

    window = bundle.X_test[index]
    window_values = inverse_target(bundle.scaler, window[:, 0]).ravel()
    truth = float(inverse_target(bundle.scaler, bundle.y_test[index])[0])
    prediction = float(inverse_target(
        bundle.scaler, model.predict(window[None, ...], verbose=0)[0])[0])
    level, comment = traffic_level(prediction, cfg.level_low, cfg.level_high)

    with st.spinner("Replaying the LSTM cell step by step…"):
        trace = trace_network(model, window)

    metric_row([
        ("Window ends", str(bundle.test_timestamps.iloc[index])[:16], "Last hour fed in"),
        ("Predicted", "{:,.0f}".format(prediction), level + " — " + comment),
        ("Actual", "{:,.0f}".format(truth), "Ground truth"),
        ("Replay error", "{:.1e}".format(trace.max_abs_error or 0.0),
         "Difference between the NumPy replay and Keras — proof these numbers are real"),
    ])

    layer_names = [layer.name for layer in trace.layers]
    layer = trace.layer(st.selectbox("Layer", layer_names))

    tab_heat, tab_step, tab_gates, tab_units = st.tabs(
        ["Activation map", "Hour by hour", "Gates", "Unit detail"])

    with tab_heat:
        st.caption("Every unit (rows) at every hour of the window (columns). "
                   "Red = positive activation, blue = negative, white = silent.")
        fig = go.Figure(go.Heatmap(
            z=layer.h.T, colorscale="RdBu", zmid=0, reversescale=True,
            colorbar=dict(title="h")))
        fig.update_layout(height=520, margin=dict(t=30, b=10),
                          xaxis_title="Hour of the window (0 = oldest)",
                          yaxis_title="{} unit".format(layer.name))
        st.plotly_chart(fig, width="stretch")

    with tab_step:
        t = st.slider("Hour of the window", 0, layer.timesteps - 1, layer.timesteps - 1,
                      help="Drag to watch the memory build up.")
        st.caption("Traffic fed in at this step: **{:,.0f}** (t-{}h)".format(
            window_values[t], layer.timesteps - t))

        # Fit the colour scale to this layer rather than to the tanh bounds —
        # hidden states usually sit well inside [-1, 1] and a fixed scale
        # washes the grid out completely.
        limit = max(float(np.percentile(np.abs(layer.h), 99)), 1e-3)
        side = int(np.ceil(np.sqrt(layer.units)))
        padded = np.full(side * side, np.nan)
        padded[: layer.units] = layer.h[t]
        labels = np.full(side * side, "", dtype=object)
        labels[: layer.units] = ["u{:02d}<br>{:+.2f}".format(u, layer.h[t, u])
                                 for u in range(layer.units)]
        grid = go.Figure(go.Heatmap(
            z=padded.reshape(side, side), text=labels.reshape(side, side),
            texttemplate="%{text}", textfont=dict(size=9),
            colorscale="RdBu", zmid=0, zmin=-limit, zmax=limit, reversescale=True,
            hoverinfo="skip", showscale=True))
        grid.update_layout(height=560, margin=dict(t=30, b=10),
                           title="{} at hour t-{}h (colour scale ±{:.2f})".format(
                               layer.name, layer.timesteps - t, limit),
                           xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"))
        st.plotly_chart(grid, width="stretch")

        gates = go.Figure()
        for key, label, colour in (("f", "Forget", "#8c564b"), ("i", "Input", "#2ca02c"),
                                   ("o", "Output", "#9467bd")):
            gates.add_bar(x=[label], y=[float(layer.gate(key)[t].mean())], marker_color=colour,
                          name=label)
        gates.update_layout(height=260, margin=dict(t=40, b=10), yaxis_range=[0, 1],
                            title="Mean gate opening at this step (0 = closed, 1 = open)",
                            showlegend=False)
        st.plotly_chart(gates, width="stretch")

    with tab_gates:
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Forget gate — how much memory survives", "Input gate — how much is written",
            "Output gate — how much is exposed", "Cell state — the memory itself"))
        spec = [("f", 1, 1, "#8c564b"), ("i", 1, 2, "#2ca02c"),
                ("o", 2, 1, "#9467bd"), ("c", 2, 2, "#1f77b4")]
        for key, row, col, colour in spec:
            values = layer.gate(key)
            mean = values.mean(axis=1) if key != "c" else np.abs(values).mean(axis=1)
            fig.add_scatter(y=mean, row=row, col=col, line=dict(color=colour, width=2.5),
                            showlegend=False)
        fig.update_layout(height=560, margin=dict(t=60, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Averaged over all {} units. A forget gate that stays near 1 means the "
                   "layer is carrying information across the whole window.".format(layer.units))

    with tab_units:
        profiles = neuron_profiles(layer)
        table = pd.DataFrame(most_active(profiles, 20))[
            ["unit", "final_h", "peak_activation", "peak_timestep",
             "mean_abs_activation", "mean_forget", "role"]]
        st.dataframe(table, width="stretch", hide_index=True)

        unit = st.number_input("Inspect unit", 0, layer.units - 1, int(table.iloc[0]["unit"]))
        detail = go.Figure()
        for key, label, colour in (("h", "hidden h", ACCENT), ("c", "cell c", BLUE),
                                   ("f", "forget f", "#8c564b"), ("i", "input i", "#2ca02c"),
                                   ("o", "output o", "#9467bd")):
            detail.add_scatter(y=layer.gate(key)[:, int(unit)], name=label,
                               line=dict(color=colour, width=2))
        detail.update_layout(height=400, margin=dict(t=40, b=10),
                             title="{} u{:02d} — every value, hour by hour".format(
                                 layer.name, int(unit)),
                             xaxis_title="Hour of the window")
        st.plotly_chart(detail, width="stretch")

    with st.expander("Which hours actually moved the prediction?"):
        with st.spinner("Perturbing each timestep…"):
            scores = input_sensitivity(model, window)
        fig = go.Figure(go.Bar(x=["t-{}h".format(layer.timesteps - t)
                                  for t in range(len(scores))],
                               y=scores, marker_color=ACCENT))
        fig.update_layout(height=330, margin=dict(t=40, b=10),
                          title="Share of the prediction each past hour is responsible for")
        st.plotly_chart(fig, width="stretch")


def page_predict():
    st.header("5 · Forecast")
    outcome = state("outcome")
    if not outcome:
        st.warning("Train a model first.")
        return
    cfg, bundle, model = state("cfg"), outcome["bundle"], outcome["model"]
    series, target = state("series"), state("target_column")

    mode = st.radio("Input", ["Last {} hours of the dataset".format(cfg.sequence_length),
                              "Type the values myself"], horizontal=True)
    if mode.startswith("Last"):
        window = series[target].to_numpy()[-cfg.sequence_length:]
        ends_at = str(series[state("datetime_column")].iloc[-1])
    else:
        default = ", ".join("{:.0f}".format(v) for v in
                            series[target].to_numpy()[-cfg.sequence_length:])
        text = st.text_area("{} comma-separated values".format(cfg.sequence_length), default,
                            height=110)
        try:
            window = np.array([float(v) for v in text.replace("\n", ",").split(",")
                               if v.strip()])
        except ValueError:
            st.error("Could not parse those numbers.")
            return
        if len(window) != cfg.sequence_length:
            st.error("Need exactly {} values, got {}.".format(cfg.sequence_length, len(window)))
            return
        ends_at = "manual input"

    scaled = bundle.scaler.transform(np.asarray(window, dtype="float64").reshape(-1, 1))
    predicted = bundle.scaler.inverse_transform(
        model.predict(scaled.reshape(1, cfg.sequence_length, 1), verbose=0).reshape(-1, 1)
    ).ravel()

    st.caption("Window ends: {}".format(ends_at))
    cols = st.columns(len(cfg.horizons))
    for col, horizon, value in zip(cols, cfg.horizons, predicted):
        level, comment = traffic_level(value, cfg.level_low, cfg.level_high)
        badge = {"LOW": "🟢", "MODERATE": "🟠", "HIGH": "🔴"}[level]
        col.metric("+{}h".format(horizon), "{:,.0f}".format(value),
                   "{} {} — {}".format(badge, level, comment))

    fig = go.Figure()
    fig.add_scatter(x=list(range(-cfg.sequence_length, 0)), y=window,
                    name="History", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=[h for h in cfg.horizons], y=predicted, name="Forecast",
                    mode="markers+lines", line=dict(color=ACCENT, width=2, dash="dot"),
                    marker=dict(size=12))
    fig.add_vline(x=0, line_dash="dash", line_color=GREY)
    fig.update_layout(height=380, margin=dict(t=40, b=10),
                      title="Input window and forecast", xaxis_title="Hours from now",
                      yaxis_title=target)
    st.plotly_chart(fig, width="stretch")


def page_vault():
    st.header("6 · Obsidian vault")
    outcome = state("outcome")
    st.markdown(
        "Regenerate the vault from the **currently trained model**: canvases with the real "
        "activations, one note per neuron, the gate notes, and the hour-by-hour timeline.")
    if not outcome:
        st.warning("Train a model first.")
        return
    include_timeline = st.checkbox("Include the hour-by-hour timeline canvases", value=True)
    if st.button("Regenerate vault", type="primary"):
        from traffic_lstm.obsidian_export import export_vault
        from traffic_lstm.train import make_figures

        with st.spinner("Rendering figures and replaying the network…"):
            make_figures(state("cfg"), outcome)
            path = export_vault(state("cfg"), outcome["model"], outcome["bundle"],
                                outcome["artifacts"], timeline=include_timeline, verbose=False)
        st.success("Vault written to `{}`".format(path))
        st.caption("In Obsidian: **Open folder as vault** → select that folder → "
                   "start from `00 Start Here`.")


# ------------------------------------------------------------------- main ---
st.title("🚦 Traffic LSTM")
st.caption("Forecast the next hour of traffic from the last 24 — and look inside the network "
           "while it does it.")

PAGES = {
    "1 · Data": page_data,
    "2 · Train": page_train,
    "3 · Results": page_results,
    "4 · Inside the network": page_neurons,
    "5 · Forecast": page_predict,
    "6 · Obsidian vault": page_vault,
}
with st.sidebar:
    st.subheader("Steps")
    selection = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    if state("outcome"):
        cfg = state("cfg")
        st.success("Model trained\n\n{} → LSTM {} → LSTM {} → Dense {}".format(
            cfg.sequence_length, cfg.lstm_units[0], cfg.lstm_units[-1], cfg.dense_units))
    else:
        st.info("No model trained yet.")
    st.caption("Every activation shown in this app is replayed from the trained weights and "
               "checked against Keras.")

PAGES[selection]()
