"""NumPy replay of baseline_univariate.keras — no TensorFlow required.

The .keras zip stores LSTM/Dense weights. The cell math is the same as
``traffic_lstm.introspect.replay_lstm`` (verified against the committed
rush/quiet traces to ~1e-7). Demo inference uses this path because the VM
does not ship TensorFlow.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KERAS_PATH = ROOT / "models" / "baseline_univariate.keras"
ARTIFACT_PATH = ROOT / "models" / "baseline_univariate_artifacts.json"
CSV_PATH = ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
LOOKUP_PATH = ROOT / "presentation" / "public" / "hour_lookup.json"

# Documented contrasting test hours (neuron_traces.json / vault).
RUSH_INDEX = 4009
QUIET_INDEX = 4068
QUIET_ACTUAL = 151.0
RUSH_ACTUAL = 7213.0
SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def replay_lstm(x_seq, kernel, recurrent_kernel, bias) -> dict:
    units = kernel.shape[1] // 4
    timesteps = x_seq.shape[0]
    h = np.zeros(units, dtype="float64")
    c = np.zeros(units, dtype="float64")
    rec = {k: np.zeros((timesteps, units), dtype="float64") for k in ("i", "f", "g", "o", "c", "h")}
    for t in range(timesteps):
        z = x_seq[t] @ kernel + h @ recurrent_kernel + bias
        i = _sigmoid(z[0 * units : 1 * units])
        f = _sigmoid(z[1 * units : 2 * units])
        g = np.tanh(z[2 * units : 3 * units])
        o = _sigmoid(z[3 * units : 4 * units])
        c = f * c + i * g
        h = o * np.tanh(c)
        rec["i"][t], rec["f"][t], rec["g"][t], rec["o"][t] = i, f, g, o
        rec["c"][t], rec["h"][t] = c, h
    return rec


def load_scaler() -> tuple[float, float]:
    artifacts = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    data_min = float(artifacts["scaler"]["data_min"][0])
    data_max = float(artifacts["scaler"]["data_max"][0])
    return data_min, data_max


def load_weights(keras_path: Path = KERAS_PATH) -> dict:
    import h5py

    with zipfile.ZipFile(keras_path) as zipped:
        raw = zipped.read("model.weights.h5")
    with h5py.File(io.BytesIO(raw), "r") as handle:
        def ds(path: str) -> np.ndarray:
            return np.array(handle[path], dtype="float64")

        return {
            "lstm1": (
                ds("layers/lstm/cell/vars/0"),
                ds("layers/lstm/cell/vars/1"),
                ds("layers/lstm/cell/vars/2"),
            ),
            "lstm2": (
                ds("layers/lstm_1/cell/vars/0"),
                ds("layers/lstm_1/cell/vars/1"),
                ds("layers/lstm_1/cell/vars/2"),
            ),
            "dense": (ds("layers/dense/vars/0"), ds("layers/dense/vars/1")),
            "out": (ds("layers/dense_1/vars/0"), ds("layers/dense_1/vars/1")),
        }


def forward(window_scaled: np.ndarray, weights: dict) -> dict:
    """window_scaled shape (24,) or (24, 1) in MinMax units."""
    x = np.asarray(window_scaled, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    layer1 = replay_lstm(x, *weights["lstm1"])
    layer2 = replay_lstm(layer1["h"], *weights["lstm2"])
    hidden = layer2["h"][-1]
    dense_w, dense_b = weights["dense"]
    dense = np.maximum(hidden @ dense_w + dense_b, 0.0)
    out_w, out_b = weights["out"]
    output_scaled = float((dense @ out_w + out_b).ravel()[0])
    return {"lstm1": layer1, "lstm2": layer2, "dense": dense, "output_scaled": output_scaled}


def _describe_role(mean_forget: float, mean_abs_h: float) -> str:
    if mean_abs_h < 0.05:
        return "Dormant - contributes almost nothing on this sample"
    if mean_forget > 0.75:
        return "Long memory - carries information across the whole day"
    if mean_forget < 0.35:
        return "Short memory - reacts mostly to the last few hours"
    return "Mixed - balances recent hours against the daily pattern"


def top_neurons(layer: dict, k: int = 6) -> list[dict]:
    hidden = layer["h"]
    forget = layer["f"]
    units = hidden.shape[1]
    ranked = []
    for unit in range(units):
        mean_abs = float(np.mean(np.abs(hidden[:, unit])))
        mean_forget = float(np.mean(forget[:, unit]))
        ranked.append(
            {
                "unit": unit,
                "mean_abs": mean_abs,
                "final": float(hidden[-1, unit]),
                "role": _describe_role(mean_forget, mean_abs),
                "mean_forget": mean_forget,
            }
        )
    ranked.sort(key=lambda row: -row["mean_abs"])
    return ranked[:k]


def pack_row(
    index: int,
    timestamp: str,
    actual: float,
    predicted: float,
    trace: dict,
    scaled_window: np.ndarray,
) -> dict:
    layer1 = trace["lstm1"]
    layer2 = trace["lstm2"]
    return {
        "index": int(index),
        "timestamp": timestamp,
        "actual_vehicles": float(actual),
        "predicted_vehicles": float(predicted),
        "gates": {key: float(layer1[key][-1].mean()) for key in ("i", "f", "g", "o")},
        "neurons": [float(v) for v in layer2["h"][-1]],
        "dense": [float(v) for v in trace["dense"]],
        "top_neurons": top_neurons(layer2),
        "scaled_window": [float(v) for v in np.asarray(scaled_window).ravel()],
    }


def load_cleaned_series(path: Path = CSV_PATH) -> list[tuple[datetime, float]]:
    buckets: dict[datetime, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            moment = datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S")
            buckets[moment].append(float(row["traffic_volume"]))
    return sorted((moment, sum(values) / len(values)) for moment, values in buckets.items())


def build_test_windows() -> dict:
    series = load_cleaned_series()
    values = np.array([volume for _, volume in series], dtype="float64")
    timestamps = [moment for moment, _ in series]
    data_min, data_max = load_scaler()
    train_size = int(len(values) * TRAIN_RATIO)
    train_raw, test_raw = values[:train_size], values[train_size:]
    bridged = np.concatenate([train_raw[-SEQUENCE_LENGTH:], test_raw])
    scaled = (bridged - data_min) / (data_max - data_min)
    windows, actuals, stamps = [], [], []
    for i in range(SEQUENCE_LENGTH, len(bridged)):
        k = i - SEQUENCE_LENGTH
        windows.append(scaled[i - SEQUENCE_LENGTH : i])
        actuals.append(float(bridged[i]))
        stamps.append(timestamps[train_size + k].strftime("%Y-%m-%d %H:%M:%S"))
    return {
        "windows": np.asarray(windows, dtype="float64"),
        "actuals": np.asarray(actuals, dtype="float64"),
        "timestamps": stamps,
        "data_min": data_min,
        "data_max": data_max,
        "train_size": train_size,
    }


def demand_from_intensity(intensity: float) -> float:
    t = max(0.0, min(1.0, float(intensity)))
    return QUIET_ACTUAL + t * (RUSH_ACTUAL - QUIET_ACTUAL)


def nearest_index(demand: float, actuals: np.ndarray) -> int:
    return int(np.argmin(np.abs(actuals - float(demand))))


def sample_indices(actuals: np.ndarray, n: int = 96) -> list[int]:
    order = np.argsort(actuals)
    ranks = np.linspace(0, len(order) - 1, n).astype(int)
    chosen = {int(order[rank]) for rank in ranks}
    chosen.add(RUSH_INDEX)
    chosen.add(QUIET_INDEX)
    return sorted(chosen)


def build_lookup(n: int = 96) -> dict:
    weights = load_weights()
    bundle = build_test_windows()
    data_min, data_max = bundle["data_min"], bundle["data_max"]
    hours = []
    for index in sample_indices(bundle["actuals"], n=n):
        window = bundle["windows"][index]
        trace = forward(window, weights)
        predicted = trace["output_scaled"] * (data_max - data_min) + data_min
        hours.append(
            pack_row(
                index=index,
                timestamp=bundle["timestamps"][index],
                actual=float(bundle["actuals"][index]),
                predicted=predicted,
                trace=trace,
                scaled_window=window,
            )
        )
    return {
        "model": "baseline_univariate",
        "keras_path": "models/baseline_univariate.keras",
        "dataset": "Metro_Interstate_Traffic_Volume",
        "claim": (
            "Each row is a real Metro Interstate test-hour forward pass of "
            "baseline_univariate.keras (NumPy replay of the trained weights, "
            "same cell as introspect.py). Not a blend of quiet/rush traces."
        ),
        "source": "keras-numpy-replay",
        "scaler": {"data_min": data_min, "data_max": data_max},
        "quiet_actual": QUIET_ACTUAL,
        "rush_actual": RUSH_ACTUAL,
        "hours": hours,
    }


def write_lookup(path: Path = LOOKUP_PATH, n: int = 96) -> Path:
    payload = build_lookup(n=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class LiveEngine:
    """Load weights + lookup once; each forecast is a real forward pass."""

    def __init__(self) -> None:
        self.lookup_error = None
        self.weights = None
        self.data_min, self.data_max = 0.0, 7280.0
        try:
            if ARTIFACT_PATH.exists():
                self.data_min, self.data_max = load_scaler()
            self.weights = load_weights()
        except Exception as exc:  # noqa: BLE001 — sidecar must stay up
            self.lookup_error = f"keras weights unavailable: {exc}"
        if LOOKUP_PATH.exists():
            self.lookup = json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))
        else:
            try:
                self.lookup = build_lookup()
                write_lookup()
            except Exception as exc:  # noqa: BLE001
                self.lookup = {"hours": [], "claim": "lookup missing"}
                self.lookup_error = self.lookup_error or str(exc)
        self.hours = list(self.lookup.get("hours") or [])
        self.actuals = np.array(
            [row["actual_vehicles"] for row in self.hours], dtype="float64"
        ) if self.hours else np.array([], dtype="float64")

    def health(self) -> dict:
        return {
            "ok": bool(self.hours) and self.weights is not None,
            "model": "baseline_univariate",
            "hours": len(self.hours),
            "weights_loaded": self.weights is not None,
            "error": self.lookup_error,
            "label": "replayed Keras output for the closest real hour",
        }

    def forecast(self, intensity: float, queue: float | None = None) -> dict:
        if not self.hours:
            return {
                "ok": False,
                "source": "unavailable",
                "error": self.lookup_error or "no hour lookup",
            }
        demand = demand_from_intensity(intensity)
        nearest = nearest_index(demand, self.actuals)
        row = dict(self.hours[nearest])
        source = "lookup"
        if self.weights is not None and row.get("scaled_window"):
            trace = forward(row["scaled_window"], self.weights)
            predicted = trace["output_scaled"] * (self.data_max - self.data_min) + self.data_min
            packed = pack_row(
                index=row["index"],
                timestamp=row["timestamp"],
                actual=row["actual_vehicles"],
                predicted=predicted,
                trace=trace,
                scaled_window=row["scaled_window"],
            )
            packed.pop("scaled_window", None)
            row = packed
            source = "sidecar"
        else:
            row.pop("scaled_window", None)
        return {
            "ok": True,
            "source": source,
            "label": "Replayed Keras output for the closest real hour",
            "model": "baseline_univariate",
            "demand_query": demand,
            "intensity": float(intensity),
            "queue": queue,
            "closed_loop": False,
            **row,
        }
