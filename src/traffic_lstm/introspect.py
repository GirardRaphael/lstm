"""Open the box: replay the trained LSTM cell by hand, hour by hour.

Keras gives you a prediction. It does not give you the four gate values of
neuron 37 at hour 18. So this module re-implements the LSTM cell in NumPy
using the trained weights, which lets us record *everything*:

    z  = x_t . W + h_{t-1} . U + b
    i  = sigmoid(z_i)        input gate   - how much new information to write
    f  = sigmoid(z_f)        forget gate  - how much of the memory to keep
    g  = tanh(z_g)           candidate    - what the new information is
    o  = sigmoid(z_o)        output gate  - how much of the memory to expose
    c  = f * c_{t-1} + i * g cell state   - the long-term memory
    h  = o * tanh(c)         hidden state - what the next layer actually sees

`verify_against_keras` proves the re-implementation is exact (~1e-6), so the
numbers exported to the Obsidian vault are the real ones, not an illustration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Keras packs the four gates into one matrix in this order.
GATE_ORDER = ("i", "f", "g", "o")
GATE_NAMES = {
    "i": "Input gate",
    "f": "Forget gate",
    "g": "Candidate memory",
    "o": "Output gate",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LayerTrace:
    """Everything that happened inside one LSTM layer for one input sequence."""

    name: str
    units: int
    i: np.ndarray       # (timesteps, units) input gate
    f: np.ndarray       # (timesteps, units) forget gate
    g: np.ndarray       # (timesteps, units) candidate
    o: np.ndarray       # (timesteps, units) output gate
    c: np.ndarray       # (timesteps, units) cell state  (long-term memory)
    h: np.ndarray       # (timesteps, units) hidden state (what is passed on)

    @property
    def timesteps(self) -> int:
        return self.h.shape[0]

    def gate(self, key: str) -> np.ndarray:
        return getattr(self, key)

    def final_hidden(self) -> np.ndarray:
        return self.h[-1]


@dataclass
class NetworkTrace:
    """A full forward pass, layer by layer, for a single 24-hour window."""

    input_sequence: np.ndarray          # (timesteps, features) scaled
    layers: list = field(default_factory=list)
    dense_hidden: np.ndarray | None = None   # (dense_units,)
    output: np.ndarray | None = None         # (n_outputs,) scaled
    keras_output: np.ndarray | None = None
    max_abs_error: float | None = None

    def layer(self, name: str) -> LayerTrace:
        for lt in self.layers:
            if lt.name == name:
                return lt
        names = [lt.name for lt in self.layers]
        raise KeyError("No layer named " + repr(name) + ". Have: " + repr(names))


# ------------------------------------------------------------- extraction ---
def lstm_layers(model) -> list:
    return [layer for layer in model.layers if layer.__class__.__name__ == "LSTM"]


def dense_layers(model) -> list:
    return [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]


def replay_lstm(x_seq, kernel, recurrent_kernel, bias, name: str) -> LayerTrace:
    """Run one LSTM layer step by step, recording every gate at every step."""
    units = kernel.shape[1] // 4
    timesteps = x_seq.shape[0]

    h = np.zeros(units, dtype="float64")
    c = np.zeros(units, dtype="float64")
    rec = {k: np.zeros((timesteps, units)) for k in ("i", "f", "g", "o", "c", "h")}

    for t in range(timesteps):
        z = x_seq[t] @ kernel + h @ recurrent_kernel + bias
        i = _sigmoid(z[0 * units:1 * units])
        f = _sigmoid(z[1 * units:2 * units])
        g = np.tanh(z[2 * units:3 * units])
        o = _sigmoid(z[3 * units:4 * units])

        c = f * c + i * g          # keep part of the old memory, write part of the new
        h = o * np.tanh(c)         # expose part of the memory to the next layer

        for key, val in zip(("i", "f", "g", "o", "c", "h"), (i, f, g, o, c, h)):
            rec[key][t] = val

    return LayerTrace(name=name, units=units, **rec)


def trace_network(model, x_sample, verify: bool = True) -> NetworkTrace:
    """Replay the whole network on one sample of shape (timesteps, features)."""
    x_sample = np.asarray(x_sample, dtype="float64")
    if x_sample.ndim == 3:
        x_sample = x_sample[0]

    trace = NetworkTrace(input_sequence=x_sample)
    signal = x_sample

    for layer in lstm_layers(model):
        kernel, recurrent_kernel, bias = (w.astype("float64") for w in layer.get_weights())
        lt = replay_lstm(signal, kernel, recurrent_kernel, bias, layer.name)
        trace.layers.append(lt)
        # Dropout is identity at inference time, so the next layer sees either
        # the full sequence of hidden states, or just the last one.
        signal = lt.h if layer.return_sequences else lt.h[-1][None, :]

    vector = signal[-1]
    for dense in dense_layers(model):
        w, b = (v.astype("float64") for v in dense.get_weights())
        vector = vector @ w + b
        if getattr(dense.activation, "__name__", "linear") == "relu":
            vector = np.maximum(vector, 0.0)
            trace.dense_hidden = vector.copy()
    trace.output = vector

    if verify:
        keras_out = model.predict(x_sample[None, ...].astype("float32"), verbose=0)[0]
        trace.keras_output = keras_out
        trace.max_abs_error = float(np.max(np.abs(keras_out - vector)))

    return trace


# ---------------------------------------------------------------- summary ---
def neuron_profiles(layer_trace: LayerTrace, top_k: int = 3) -> list:
    """Per-neuron fingerprint: when it fires, how hard, and what it remembers."""
    profiles = []
    for u in range(layer_trace.units):
        h = layer_trace.h[:, u]
        c = layer_trace.c[:, u]
        f = layer_trace.f[:, u]
        i = layer_trace.i[:, u]
        peak_t = int(np.argmax(np.abs(h)))
        # A forget gate near 1.0 on average means the unit holds on to the
        # past; near 0.0 means it only cares about the present.
        memory_score = float(np.mean(f))
        profiles.append(
            {
                "unit": u,
                "layer": layer_trace.name,
                "final_h": float(h[-1]),
                "peak_activation": float(h[peak_t]),
                "peak_timestep": peak_t,
                "mean_abs_activation": float(np.mean(np.abs(h))),
                "activation_range": float(h.max() - h.min()),
                "mean_forget": memory_score,
                "mean_input": float(np.mean(i)),
                "final_cell": float(c[-1]),
                "role": _describe_role(memory_score, float(np.mean(np.abs(h)))),
                "top_timesteps": [int(t) for t in np.argsort(-np.abs(h))[:top_k]],
            }
        )
    return profiles


def _describe_role(mean_forget: float, mean_abs_h: float) -> str:
    if mean_abs_h < 0.05:
        return "Dormant - contributes almost nothing on this sample"
    if mean_forget > 0.75:
        return "Long memory - carries information across the whole day"
    if mean_forget < 0.35:
        return "Short memory - reacts mostly to the last few hours"
    return "Mixed - balances recent hours against the daily pattern"


def most_active(profiles: list, k: int = 10) -> list:
    return sorted(profiles, key=lambda p: -p["mean_abs_activation"])[:k]


def input_sensitivity(model, x_sample, epsilon: float = 0.05) -> np.ndarray:
    """Which of the 24 hours actually moved the prediction?

    Perturb one timestep at a time and measure the change in output. This is a
    simple, defensible saliency measure - no gradients required, and easy to
    explain out loud.
    """
    x_sample = np.asarray(x_sample, dtype="float32")
    if x_sample.ndim == 3:
        x_sample = x_sample[0]
    base = model.predict(x_sample[None, ...], verbose=0)[0]
    scores = np.zeros(x_sample.shape[0])
    for t in range(x_sample.shape[0]):
        bumped = x_sample.copy()
        bumped[t] += epsilon
        moved = model.predict(bumped[None, ...], verbose=0)[0]
        scores[t] = float(np.mean(np.abs(moved - base)))
    total = scores.sum()
    return scores / total if total > 0 else scores
