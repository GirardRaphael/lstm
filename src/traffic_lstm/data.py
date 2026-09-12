"""Raw CSV -> supervised sequences, without leaking the future into the past.

The three rules this module enforces:

1. Rows are sorted chronologically and duplicate timestamps are collapsed.
2. The train/test split is chronological - never shuffled.
3. The scaler is fitted on the training slice only. Anything else is data
   leakage: the model would indirectly learn the range of the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import TrainingConfig
from .features import build_feature_frame, feature_columns


# ---------------------------------------------------------------- loading ---
def load_raw(path: str | Path, datetime_column: str = "date_time") -> pd.DataFrame:
    """Read a CSV and parse its datetime column."""
    df = pd.read_csv(path)
    if datetime_column not in df.columns:
        raise KeyError(
            f"Column '{datetime_column}' not found. Available: {list(df.columns)}"
        )
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    return df


def describe_quality(df: pd.DataFrame, datetime_column: str, target_column: str) -> dict:
    """Facts about the dataset that belong in the report, not in a comment."""
    ts = df[datetime_column]
    full_range = pd.date_range(ts.min(), ts.max(), freq="h")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "duplicated_timestamps": int(ts.duplicated().sum()),
        "missing_values": int(df[target_column].isna().sum()),
        "expected_hourly_rows": int(len(full_range)),
        "missing_hours": int(len(full_range) - ts.nunique()),
        "target_min": float(df[target_column].min()),
        "target_max": float(df[target_column].max()),
        "target_mean": float(df[target_column].mean()),
    }


def clean_series(
    df: pd.DataFrame,
    datetime_column: str = "date_time",
    target_column: str = "traffic_volume",
) -> pd.DataFrame:
    """Sort chronologically, collapse duplicate hours, drop missing targets.

    The Metro Interstate dataset contains a few hundred duplicated timestamps
    (the same hour logged twice with different weather descriptions). We keep
    one row per hour by averaging the target, which is the honest choice: we
    neither invent data nor silently keep two conflicting observations.
    """
    out = (
        df[[datetime_column, target_column]]
        .dropna(subset=[target_column])
        .sort_values(datetime_column)
        .groupby(datetime_column, as_index=False)[target_column]
        .mean()
        .reset_index(drop=True)
    )
    return out


# -------------------------------------------------------------- sequences ---
def contiguous_mask(timestamps, freq_hours: int = 1) -> np.ndarray:
    """True where a row follows the previous one by exactly one period.

    Used to detect windows that span a gap in the series. The first element is
    True by convention - there is nothing before it to be discontinuous with.
    """
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    deltas = ts.diff().dt.total_seconds().div(3600)
    return np.concatenate([[True], np.isclose(deltas.to_numpy()[1:], freq_hours)])


def create_sequences(
    data: np.ndarray, sequence_length: int, horizons: tuple[int, ...] = (1,),
    contiguous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over the series to build (X, y) pairs.

    ``data`` has shape ``(rows, features)``. **Column 0 is the target.** Every
    column feeds the input window; only column 0 is predicted.

    For a window ending at index ``i`` (exclusive):

        X = data[i - sequence_length : i, :]        the last `sequence_length` hours
        y = [data[i + h - 1, 0] for h in horizons]  the target h hours later

    With ``horizons=(1,)`` this is the classic "predict the next hour".
    Returns ``(X, y, end_index)`` where ``end_index[k]`` is the index in
    ``data`` of the first predicted step, so predictions can be traced back to
    real timestamps.

    If ``contiguous`` is given (see :func:`contiguous_mask`), any window that
    spans a break in the series is skipped rather than silently pretending
    that 24 readings taken across a 300-day outage are "the last 24 hours".
    """
    max_h = max(horizons)
    xs, ys, idx = [], [], []
    for i in range(sequence_length, len(data) - max_h + 1):
        if contiguous is not None and not contiguous[i - sequence_length + 1 : i + max_h].all():
            continue
        xs.append(data[i - sequence_length : i, :])
        ys.append([data[i + h - 1, 0] for h in horizons])
        idx.append(i)
    if not xs:
        raise ValueError(
            f"Not enough data: need at least {sequence_length + max_h} rows, got {len(data)}"
        )
    # Already (samples, timesteps, features), which is what an LSTM expects.
    X = np.asarray(xs, dtype="float32")
    return X, np.asarray(ys, dtype="float32"), np.asarray(idx, dtype="int64")


# ---------------------------------------------------------------- bundles ---
@dataclass
class DataBundle:
    """Everything downstream code needs, already split and scaled."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler          # target column only - used to read predictions
    feature_scaler: MinMaxScaler  # every input column
    feature_names: list           # column order of the input matrix, target first
    series: pd.DataFrame          # cleaned, full, unscaled
    train_size: int               # number of raw rows used for training
    test_timestamps: pd.Series    # timestamp of the first predicted step
    test_indices: np.ndarray      # window end-index of each test sample, in
                                  # the bridged series - baselines align on it
    quality: dict

    @property
    def n_features(self) -> int:
        return self.X_train.shape[2]


def build_datasets(cfg: TrainingConfig, df: pd.DataFrame | None = None) -> DataBundle:
    """Full pipeline: load -> clean -> split -> scale -> sequence."""
    if df is None:
        df = load_raw(cfg.data_path, cfg.datetime_column)
    quality = describe_quality(df, cfg.datetime_column, cfg.target_column)

    # With no extras this produces exactly the same two columns as
    # `clean_series`, so the univariate baseline stays reproducible.
    series = build_feature_frame(
        df, cfg.datetime_column, cfg.target_column,
        exogenous=cfg.exogenous_columns, calendar=cfg.calendar_features)
    names = feature_columns(series, cfg.datetime_column)

    values = series[names].to_numpy(dtype="float64")
    train_size = int(len(values) * cfg.train_ratio)
    train_raw, test_raw = values[:train_size], values[train_size:]

    # Fit on train only -> no leakage.
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = feature_scaler.fit_transform(train_raw)

    # MinMax scales each column independently, so column 0 of the scaled matrix
    # is the target scaled by its own min/max. That lets a one-column scaler
    # invert predictions no matter how many inputs the model takes.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw[:, :1])

    timestamps = series[cfg.datetime_column]
    contiguous = contiguous_mask(timestamps) if cfg.drop_gapped_windows else None

    X_train, y_train, _ = create_sequences(
        train_scaled, cfg.sequence_length, cfg.horizons,
        contiguous=None if contiguous is None else contiguous[:train_size])

    # The first test window needs the last `sequence_length` hours of training
    # data as its history - otherwise we would throw away the first day of the
    # test set. This is history, not leakage: the model only ever looks back.
    bridged_raw = np.concatenate([train_raw[-cfg.sequence_length :], test_raw])
    bridged_scaled = feature_scaler.transform(bridged_raw)
    bridged_contiguous = (None if contiguous is None
                          else contiguous[train_size - cfg.sequence_length:])
    X_test, y_test, end_idx = create_sequences(
        bridged_scaled, cfg.sequence_length, cfg.horizons, contiguous=bridged_contiguous
    )

    # Map each test sample back to its real timestamp.
    offset = train_size - cfg.sequence_length
    ts = timestamps.to_numpy()
    test_timestamps = pd.Series(ts[end_idx + offset])

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_scaler=feature_scaler,
        feature_names=names,
        series=series,
        train_size=train_size,
        test_timestamps=test_timestamps,
        test_indices=end_idx,
        quality=quality,
    )


def inverse_target(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Undo the scaling for an array of shape (n,) or (n, k)."""
    values = np.asarray(values, dtype="float64")
    flat = values.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(values.shape)
