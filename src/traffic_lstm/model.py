"""The network itself.

    (24 hours, 1 feature)
            |
        LSTM 64   return_sequences=True   -> one hidden state per hour
            |
        Dropout 0.2
            |
        LSTM 32   return_sequences=False  -> one summary vector for the day
            |
        Dropout 0.2
            |
        Dense 16  relu                    -> non-linear recombination
            |
        Dense  N  linear                  -> traffic volume, one per horizon

The last layer is linear (no activation) because this is a regression: the
output is a number of vehicles, not a probability.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from .config import TrainingConfig


def set_seeds(seed: int = 42) -> None:
    """Make a run reproducible enough to be defended in a presentation."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(cfg: TrainingConfig, n_features: int = 1) -> Sequential:
    """Assemble the stacked LSTM regressor described in the module docstring."""
    layers: list = [Input(shape=(cfg.sequence_length, n_features), name="input_sequence")]

    for depth, units in enumerate(cfg.lstm_units):
        is_last_lstm = depth == len(cfg.lstm_units) - 1
        layers.append(
            LSTM(
                units,
                # Every LSTM but the last one must hand a full sequence to the
                # next one; the last one collapses the sequence into a vector.
                return_sequences=not is_last_lstm,
                name=f"lstm_{depth + 1}",
            )
        )
        layers.append(Dropout(cfg.dropout, name=f"dropout_{depth + 1}"))

    layers.append(Dense(cfg.dense_units, activation="relu", name="dense_hidden"))
    layers.append(Dense(cfg.n_outputs, name="traffic_output"))

    model = Sequential(layers, name=cfg.run_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",   # penalises large misses hard - we care about peaks
        metrics=["mae"],  # interpretable: "wrong by N vehicles on average"
    )
    return model


def default_callbacks(cfg: TrainingConfig) -> list:
    """Stop before overfitting, and slow down when progress stalls."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, cfg.patience // 2),
            min_lr=1e-5,
            verbose=0,
        ),
    ]


def summarise(model: Sequential) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)
