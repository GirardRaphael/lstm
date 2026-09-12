"""Builders for Obsidian's JSON Canvas format.

A canvas is just JSON: a list of positioned nodes and a list of edges. That
makes it the perfect target for "draw me the network as it actually behaved" -
we place one node per neuron and colour it with the activation we replayed
from the trained weights.

Three canvases are produced:

* `build_architecture_canvas`  - the static shape of the model
* `build_neuron_canvas`        - every unit, coloured by what it did on one window
* `build_timeline_canvas`      - the same layer, hour by hour, 24 of them
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluate import traffic_level
from .introspect import GATE_NAMES, most_active, neuron_profiles

SPARK_CHARS = "▁▂▃▄▅▆▇█"


# ------------------------------------------------------------- formatting ---
def sparkline(values) -> str:
    """A tiny inline chart that renders everywhere, no plugin required."""
    values = np.asarray(values, dtype="float64").ravel()
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return SPARK_CHARS[0] * len(values)
    idx = ((values - lo) / (hi - lo) * (len(SPARK_CHARS) - 1)).round().astype(int)
    return "".join(SPARK_CHARS[i] for i in idx)


def _mix(a, b, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    rgb = tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


NEUTRAL = (246, 246, 246)
WARM = (214, 39, 40)     # positive activation
COOL = (31, 119, 180)    # negative activation


def activation_colour(value: float, scale: float = 1.0) -> str:
    """Diverging colour: blue = negative, grey = silent, red = positive."""
    t = min(abs(float(value)) / max(scale, 1e-9), 1.0)
    return _mix(NEUTRAL, WARM if value >= 0 else COOL, t)


def gate_colour(value: float) -> str:
    """Sequential colour for a gate in [0, 1]: pale = closed, green = wide open."""
    return _mix((248, 248, 248), (44, 160, 44), float(value))


# ------------------------------------------------------------ canvas core ---
class Canvas:
    """Minimal builder for Obsidian's JSON Canvas format."""

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self._n = 0

    def text(self, text, x, y, w=160, h=48, color=None) -> str:
        self._n += 1
        node_id = "n{}".format(self._n)
        node = {"id": node_id, "type": "text", "text": text,
                "x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        if color:
            node["color"] = color
        self.nodes.append(node)
        return node_id

    def link(self, a, b, color=None, label=None, from_side="right", to_side="left") -> None:
        self._n += 1
        edge = {"id": "e{}".format(self._n), "fromNode": a, "fromSide": from_side,
                "toNode": b, "toSide": to_side}
        if color:
            edge["color"] = color
        if label:
            edge["label"] = label
        self.edges.append(edge)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": self.nodes, "edges": self.edges}
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return path


# --------------------------------------------------------- canvas layouts ---
LAYOUT = {
    "input": {"x": 0, "w": 180, "h": 44, "gap_y": 52},
    "lstm_1": {"x": 380, "w": 104, "h": 56, "gap_x": 118, "gap_y": 66, "cols": 8, "y": 240},
    "lstm_2": {"x": 1420, "w": 104, "h": 56, "gap_x": 118, "gap_y": 66, "cols": 4, "y": 240},
    "dense": {"x": 2000, "w": 104, "h": 56, "gap_x": 118, "gap_y": 66, "cols": 4, "y": 380},
    "output": {"x": 2520, "w": 250, "h": 120, "y": 500},
}
HEADER_Y = 60


def _grid_position(layer: str, index: int):
    spec = LAYOUT[layer]
    row, col = divmod(index, spec["cols"])
    return spec["x"] + col * spec["gap_x"], spec["y"] + row * spec["gap_y"]


def _header(canvas: Canvas, x: int, label: str, width: int = 340) -> str:
    return canvas.text("### " + label, x, HEADER_Y, width, 60, color="6")


# ------------------------------------------------------------ 1. structure ---
def build_architecture_canvas(cfg, artifacts: dict) -> Canvas:
    """A clean, static picture of the model - the one for the slide deck."""
    c = Canvas()
    quality = artifacts["data_quality"]
    title = c.text(
        "# Traffic LSTM\n**{}**\n\n{:,} hourly rows\n{} to {}".format(
            cfg.run_name, quality["rows"], quality["start"][:10], quality["end"][:10]),
        -40, -280, 430, 200, color="6")

    blocks = [
        ("Input\n`(24, 1)`\n24 hours of traffic volume", "1"),
        ("LSTM {}\n`return_sequences=True`\none hidden state per hour".format(cfg.lstm_units[0]), "2"),
        ("Dropout {}\nregularisation".format(cfg.dropout), "0"),
        ("LSTM {}\n`return_sequences=False`\none summary vector for the day".format(cfg.lstm_units[-1]), "3"),
        ("Dropout {}\nregularisation".format(cfg.dropout), "0"),
        ("Dense {}\n`relu`\nnon-linear recombination".format(cfg.dense_units), "4"),
        ("Dense {}\n`linear`\nvehicles per hour".format(cfg.n_outputs), "5"),
    ]
    ids = [c.text(text, 0, i * 180, 430, 130, color=colour)
           for i, (text, colour) in enumerate(blocks)]

    c.link(title, ids[0], from_side="bottom", to_side="top")
    for a, b in zip(ids, ids[1:]):
        c.link(a, b, from_side="bottom", to_side="top")

    guide = c.text(
        "## How to read this vault\n\n"
        "- **Neurons - Rush Hour** - every unit coloured by what it actually did\n"
        "- **Neurons - Quiet Night** - the same units on a near-empty road\n"
        "- **Timeline/** - one canvas per hour: watch the memory build up\n"
        "- **Neurons/** - one note per unit, with its own trace\n\n"
        "Every number here was replayed from the trained weights.",
        540, 180, 480, 340, color="6")
    c.link(ids[3], guide, from_side="right", to_side="left")
    return c


# ------------------------------------------------------------- 2. neurons ---
def build_neuron_canvas(cfg, trace, sensitivity, label: str, context: dict) -> Canvas:
    """Every neuron of the network, coloured by its activation on one window."""
    c = Canvas()
    layer_1, layer_2 = trace.layers[0], trace.layers[1]
    x_in = trace.input_sequence[:, 0]
    n_steps = len(x_in)

    level, comment = traffic_level(context["prediction"], cfg.level_low, cfg.level_high)
    c.text(
        "# {}\n\n**Window ends:** {}\n**Predicted:** {:,.0f} vehicles - **{}** ({})\n"
        "**Actual:** {:,.0f} vehicles\n\n"
        "Red = positive activation, blue = negative, grey = silent. "
        "Only the strongest connections are drawn.".format(
            label, context["timestamp"], context["prediction"], level, comment,
            context["target"]),
        -40, -300, 920, 230, color="6")

    # --- input column ---
    _header(c, LAYOUT["input"]["x"], "Input - last {} hours".format(n_steps), 300)
    input_ids = []
    lo, hi = float(x_in.min()), float(x_in.max())
    span = max(hi - lo, 1e-9)
    for t in range(n_steps):
        y = 160 + t * LAYOUT["input"]["gap_y"]
        shade = _mix((248, 248, 248), (255, 127, 14), (float(x_in[t]) - lo) / span)
        input_ids.append(c.text(
            "**t-{}h** - {:,.0f}".format(n_steps - t, context["window_values"][t]),
            LAYOUT["input"]["x"], y, LAYOUT["input"]["w"], LAYOUT["input"]["h"], color=shade))

    # --- LSTM layers ---
    ids = {"lstm_1": [], "lstm_2": []}
    for key, layer in (("lstm_1", layer_1), ("lstm_2", layer_2)):
        _header(c, LAYOUT[key]["x"], "{} - {} units".format(layer.name, layer.units), 380)
        scale = max(float(np.abs(layer.h[-1]).max()), 0.2)
        for u in range(layer.units):
            x, y = _grid_position(key, u)
            value = float(layer.h[-1, u])
            node_text = "[[{} u{:02d}|u{:02d}]]\n`{:+.2f}`".format(layer.name, u, u, value)
            ids[key].append(c.text(node_text, x, y, LAYOUT[key]["w"], LAYOUT[key]["h"],
                                   color=activation_colour(value, scale)))

    # --- dense + output ---
    dense_ids = []
    if trace.dense_hidden is not None:
        _header(c, LAYOUT["dense"]["x"], "Dense {} (relu)".format(len(trace.dense_hidden)), 380)
        dense_scale = max(float(np.abs(trace.dense_hidden).max()), 1e-6)
        for u, value in enumerate(trace.dense_hidden):
            x, y = _grid_position("dense", u)
            dense_ids.append(c.text(
                "**d{:02d}**\n`{:+.2f}`".format(u, float(value)), x, y,
                LAYOUT["dense"]["w"], LAYOUT["dense"]["h"],
                color=activation_colour(float(value), dense_scale)))

    output_id = c.text(
        "# {:,.0f}\nvehicles / hour\n\n**{}**".format(context["prediction"], level),
        LAYOUT["output"]["x"], LAYOUT["output"]["y"],
        LAYOUT["output"]["w"], LAYOUT["output"]["h"], color="1")

    # --- representative edges: hours that mattered -> units that fired ---
    top_hours = [int(t) for t in np.argsort(-sensitivity)[:4]]
    top_1 = [p["unit"] for p in most_active(neuron_profiles(layer_1), 6)]
    top_2 = [p["unit"] for p in most_active(neuron_profiles(layer_2), 4)]

    for t in top_hours:
        for u in top_1[:3]:
            c.link(input_ids[t], ids["lstm_1"][u], color="2")
    for u in top_1:
        for v in top_2[:2]:
            c.link(ids["lstm_1"][u], ids["lstm_2"][v], color="3")
    for v in top_2:
        for d in range(min(3, len(dense_ids))):
            c.link(ids["lstm_2"][v], dense_ids[d], color="4")
    for node in dense_ids[:4]:
        c.link(node, output_id, color="1")

    bullets = "\n".join(
        "- **t-{}h** - {:.1f}% of the movement".format(n_steps - t, sensitivity[t] * 100)
        for t in top_hours)
    c.text("## Hours that moved the prediction\n\n" + bullets,
           LAYOUT["output"]["x"], LAYOUT["output"]["y"] + 220, 380, 240, color="6")
    return c


# ------------------------------------------------------------ 3. timeline ---
def build_timeline_canvas(cfg, trace, t: int, context: dict) -> Canvas:
    """One hour of the window: what the memory looks like at this exact step."""
    c = Canvas()
    layer = trace.layers[0]
    x_in = trace.input_sequence[:, 0]
    n_steps = len(x_in)

    previous = "[[t{:02d}.canvas|<- previous hour]]".format(t - 1) if t > 0 else "_start of window_"
    following = "[[t{:02d}.canvas|next hour ->]]".format(t + 1) if t < n_steps - 1 else "_end of window_"

    c.text(
        "# Hour t-{}h   (step {} of {})\n\n"
        "**Traffic fed in:** {:,.0f} vehicles\n**Window ends:** {}\n\n"
        "{} - {}\n\n"
        "Colour = hidden state `h` of each unit **at this exact step**.".format(
            n_steps - t, t + 1, n_steps, context["window_values"][t],
            context["timestamp"], previous, following),
        -40, -300, 900, 240, color="6")

    for k, key in enumerate(("f", "i", "o")):
        value = float(layer.gate(key)[t].mean())
        c.text("**{}**\n`{:.2f}` open\n\n[[{}]]".format(GATE_NAMES[key], value, GATE_NAMES[key]),
               -40, 160 + k * 150, 280, 130, color=gate_colour(value))
    c.text("**Cell state**\nmean `|c|` = `{:.2f}`\n\n[[Cell state]]".format(
        float(np.abs(layer.c[t]).mean())), -40, 160 + 3 * 150, 280, 130, color="5")

    _header(c, LAYOUT["lstm_1"]["x"],
            "{} hidden states at t-{}h".format(layer.name, n_steps - t), 480)
    scale = max(float(np.abs(layer.h).max()), 0.2)
    for u in range(layer.units):
        x, y = _grid_position("lstm_1", u)
        value = float(layer.h[t, u])
        c.text("**u{:02d}**\n`{:+.2f}`".format(u, value), x, y,
               LAYOUT["lstm_1"]["w"], LAYOUT["lstm_1"]["h"],
               color=activation_colour(value, scale))

    rows = -(-layer.units // LAYOUT["lstm_1"]["cols"])
    c.text("## Window consumed so far\n\n`{}`\n\n{} of {} hours.".format(
        sparkline(x_in[: t + 1]), t + 1, n_steps),
        LAYOUT["lstm_1"]["x"], LAYOUT["lstm_1"]["y"] + rows * LAYOUT["lstm_1"]["gap_y"] + 40,
        480, 170, color="6")
    return c
