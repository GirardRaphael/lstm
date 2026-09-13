"""Causal training pipeline, version 2 - a fresh, self-contained implementation.

v1 (``data.py``/``train.py``) proved the modelling point but kept the three
leaks documented in ROAD_PRODUCT_PLAN.md ("Critical findings STILL OPEN"):

1. **Look-ahead in preprocessing** - weather was clipped and filled with
   statistics computed over the whole frame, so a missing early value could
   change when only later observations changed.
2. **Validation contamination** - the scaler was fitted on the whole pre-test
   slice, which includes the segment Keras later used for validation.
3. **Silent gap crossing** - windows were allowed to span missing hours, and
   the seasonal baseline used row offsets, which are not wall-clock
   "yesterday" on an irregular series.

v2 enforces a strict causal contract:

* **Split first.** Raw timestamps are partitioned into fit / validation /
  test *before* any learned transform is fitted. Both scalers (feature and
  target) are fitted on fit rows only - validation rows are never seen by a
  transform either.
* **Complete windows or nothing.** A window containing a missing value or a
  non-cadence step is EXCLUDED - never imputed, never filled forward or
  backward - and every exclusion is counted by reason.
* **Boundary purge.** An example whose target horizon crosses a partition
  boundary is dropped entirely. Rows from an earlier partition MAY still
  serve as input context: that is history, not leakage.
* **Declared units.** Every modelled column carries an explicit units
  declaration. A column named ``temp`` is *not* assumed to be kelvin; the
  pipeline applies no unit conversion - it records and checks declarations.
* **Known-future calendar values are allowed.** Hour-of-day and day-of-week
  sine/cosine features are deterministic functions of the timestamp, so they
  are legitimately known at forecast time. They are the only "future"
  inputs permitted, and they are derived, never observed.
* **Timestamp-true baselines.** The seasonal baseline looks up the same
  clock hour one day (optionally one week) earlier by *timestamp*, falling
  back to persistence when that observation is absent.
* **Versioned, verifiable packages.** A run writes a directory holding the
  model file(s), both scalers, the ordered feature list, the preprocessing
  version, units, cadence, site scope, the cleaned-series SHA256 and a
  per-file SHA256 that is verified on load - corrupted or mismatched
  artifacts are rejected. Split/coverage metadata, dependency versions, the
  seed and the evaluation protocol travel with the package. Undefined
  percentage metrics (e.g. MAPE when no actual exceeds the threshold)
  serialise as JSON ``null``. A run name that already exists is refused,
  and a forecast is refused when the recent history contains gaps.

Archived v1 artifacts (``models/*_artifacts.json`` without a
``pipeline_version``) keep their original semantics and stay readable via
:func:`read_legacy_artifact`. Records whose weights were not retained
(``window_012``, ``window_024``) are metrics-only: they parse, but
:func:`load_legacy_model` refuses to load them as models.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import DEFAULT_DATASET, MODEL_DIR

PIPELINE_VERSION = "v2"
PREPROCESSING_VERSION = "v2-causal-1"
# Archived artifacts carry no pipeline_version key; they are reported as "v1".
PIPELINE_VERSION_LEGACY = "v1"

FIT, VALIDATION, TEST = "fit", "validation", "test"
PARTITIONS = (FIT, VALIDATION, TEST)

# Exclusion reasons. Every candidate window that is dropped is counted exactly
# once, by the first check it fails, in the order they are listed here.
REASON_NON_CADENCE = "non_cadence_step"
REASON_MISSING_INPUT = "missing_input"
REASON_MISSING_TARGET = "missing_target"
REASON_BOUNDARY_PURGE = "boundary_purge"
EXCLUSION_REASONS = (
    REASON_NON_CADENCE,
    REASON_MISSING_INPUT,
    REASON_MISSING_TARGET,
    REASON_BOUNDARY_PURGE,
)

# Calendar features are derived from the timestamp alone - known-future by
# design, documented in every manifest under schema.known_future_inputs.
CALENDAR_COLUMNS = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos")


# ---------------------------------------------------------------- errors ----
class PipelineV2Error(Exception):
    """Base class for every error raised by the v2 pipeline."""


class UnitsDeclarationError(ValueError, PipelineV2Error):
    """A modelled column has no units declaration (or an unknown one has)."""


class SchemaError(ValueError, PipelineV2Error):
    """A frame does not match the declared schema."""


class PartitionError(ValueError, PipelineV2Error):
    """The chronological split is impossible or produced an empty partition."""


class ArtifactExistsError(FileExistsError, PipelineV2Error):
    """A package with this run name already exists; overwriting is refused."""


class ArtifactCorruptionError(RuntimeError, PipelineV2Error):
    """A packaged file is missing or its SHA256 does not match the manifest."""


class GapHistoryError(ValueError, PipelineV2Error):
    """Recent history has a gap, a duplicate, missing values or is too short."""


class MetricsOnlyArtifactError(RuntimeError, PipelineV2Error):
    """The record kept metrics but its weights were not retained."""


# ---------------------------------------------------------------- config ----
@dataclass
class V2Config:
    """Data contract and hyper-parameters for one causal v2 run.

    ``units`` is REQUIRED for the target and every exogenous column. The
    pipeline never converts units; the declaration is recorded in the
    package and checked again at inference time.
    """

    pipeline_version: str = PIPELINE_VERSION

    # --- data contract ---
    data_path: Path = DEFAULT_DATASET
    datetime_column: str = "date_time"
    target_column: str = "traffic_volume"
    exogenous_columns: Sequence[str] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    cadence: str = "1h"                 # expected spacing between rows
    site_scope: str = ""                # e.g. "metro-interstate-mn"; required
    calendar_features: bool = False

    # --- sequence construction ---
    sequence_length: int = 24
    horizons: Sequence[int] = (1,)

    # --- chronological split (ratios over cleaned rows, or explicit) ---
    fit_ratio: float = 0.70
    validation_ratio: float = 0.15
    fit_end: str | None = None          # rows < fit_end are fit
    validation_end: str | None = None   # rows >= validation_end are test

    # --- cleaning policy ---
    duplicate_policy: str = "mean"      # "mean" | "first" | "error"
    seasonal_period: str = "24h"        # seasonal baseline lag ("168h" = week)
    mape_min_actual: float = 1.0        # MAPE undefined below this; -> null

    # --- bookkeeping ---
    seed: int = 42
    run_name: str = "v2_run"
    output_dir: Path = MODEL_DIR / "v2"

    # --- LSTM ---
    lstm_units: Sequence[int] = (64, 32)
    dense_units: int = 16
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    patience: int = 5

    # --- XGBoost comparator ---
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    xgb_early_stopping: int = 30

    def __post_init__(self) -> None:
        if self.pipeline_version != PIPELINE_VERSION:
            raise ValueError(
                f"V2Config only builds pipeline_version={PIPELINE_VERSION!r}, "
                f"got {self.pipeline_version!r}")
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        self.horizons = tuple(int(h) for h in self.horizons)
        self.lstm_units = tuple(int(u) for u in self.lstm_units)
        self.exogenous_columns = tuple(self.exogenous_columns)
        self.units = dict(self.units)

        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be >= 1")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if self.datetime_column == self.target_column:
            raise ValueError("datetime and target columns must differ")
        if self.target_column in self.exogenous_columns:
            raise ValueError("target column must not also be exogenous")
        if self.datetime_column in self.exogenous_columns:
            raise ValueError("datetime column must not also be exogenous")

        # --- units: required for every modelled column, rejected otherwise ---
        declared = [self.target_column, *self.exogenous_columns]
        missing = [c for c in declared if not str(self.units.get(c, "")).strip()]
        unknown = [c for c in self.units if c not in declared]
        if missing:
            raise UnitsDeclarationError(
                "Missing units declaration for column(s) "
                f"{missing}. Units are required: a column named 'temp' is not "
                "assumed to be kelvin - declare e.g. units={'temp': 'celsius'}.")
        if unknown:
            raise UnitsDeclarationError(
                f"Units declared for column(s) {unknown} that are not modelled "
                f"(target + exogenous = {declared}). Fix the typo or declare "
                "the column as exogenous.")

        # --- cadence ---
        try:
            seconds = self.cadence_seconds
        except ValueError as exc:
            raise ValueError(f"cadence {self.cadence!r} is not parseable") from exc
        if seconds <= 0:
            raise ValueError("cadence must be positive")
        if self.seasonal_timedelta.total_seconds() <= 0:
            raise ValueError("seasonal_period must be positive")

        # --- split ---
        explicit = (self.fit_end is not None, self.validation_end is not None)
        if any(explicit) and not all(explicit):
            raise ValueError("fit_end and validation_end must be set together")
        if all(explicit):
            if pd.Timestamp(self.fit_end) >= pd.Timestamp(self.validation_end):
                raise ValueError("fit_end must be earlier than validation_end")
        else:
            if not 0.0 < self.fit_ratio < 1.0:
                raise ValueError("fit_ratio must be in (0, 1)")
            if not 0.0 < self.validation_ratio < 1.0:
                raise ValueError("validation_ratio must be in (0, 1)")
            if self.fit_ratio + self.validation_ratio >= 1.0:
                raise ValueError("fit_ratio + validation_ratio must be < 1 "
                                 "(the test partition needs the remainder)")

        if self.duplicate_policy not in ("mean", "first", "error"):
            raise ValueError("duplicate_policy must be 'mean', 'first' or 'error'")
        if not str(self.site_scope).strip():
            raise ValueError("site_scope is required - name the site/corridor "
                             "this model is allowed to serve.")
        if (not self.run_name or self.run_name in (".", "..")
                or "/" in self.run_name or "\\" in self.run_name):
            raise ValueError(f"run_name {self.run_name!r} is not a safe directory name")

    # -- convenience ---------------------------------------------------------
    @property
    def cadence_seconds(self) -> float:
        return pd.Timedelta(self.cadence).total_seconds()

    @property
    def seasonal_timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(self.seasonal_period)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_path"] = str(self.data_path)
        d["output_dir"] = str(self.output_dir)
        d["horizons"] = list(self.horizons)
        d["lstm_units"] = list(self.lstm_units)
        d["exogenous_columns"] = list(self.exogenous_columns)
        d["units"] = dict(self.units)
        return d


# ---------------------------------------------------------------- scalers ---
def scaler_to_dict(scaler: MinMaxScaler) -> dict:
    """Plain-JSON form of a fitted MinMaxScaler - no pickle, no version drift."""
    return {
        "data_min": [float(v) for v in scaler.data_min_],
        "data_max": [float(v) for v in scaler.data_max_],
        "feature_range": [float(v) for v in scaler.feature_range],
    }


def scaler_from_dict(payload: dict) -> MinMaxScaler:
    """Rebuild a MinMaxScaler from :func:`scaler_to_dict` output."""
    scaler = MinMaxScaler(feature_range=tuple(payload["feature_range"]))
    data_min = np.asarray(payload["data_min"], dtype="float64")
    data_max = np.asarray(payload["data_max"], dtype="float64")
    scaler.fit(np.vstack([data_min, data_max]))
    return scaler


def _inverse_target(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Undo target scaling for an array of shape (n,) or (n, k)."""
    values = np.asarray(values, dtype="float64")
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


# ---------------------------------------------------------------- hashes ----
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _series_sha256(frame: pd.DataFrame, datetime_column: str,
                   columns: Sequence[str]) -> str:
    """Hash the cleaned observed series (timestamps + declared data columns).

    Serialisation is canonical (ISO timestamps, 17-significant-digit floats,
    LF endings, NaN as empty), so the hash is stable across platforms.
    """
    out = frame[[datetime_column, *columns]].copy()
    out[datetime_column] = out[datetime_column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    text = out.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_versions() -> dict:
    """Dependency/runtime versions recorded in every package.

    Distribution names vary by platform (this Windows environment installs
    TensorFlow as ``tensorflow_cpu``), so each canonical key tries aliases.
    """
    aliases = {
        "numpy": ("numpy",),
        "pandas": ("pandas",),
        "scikit-learn": ("scikit-learn",),
        "tensorflow": ("tensorflow", "tensorflow_cpu", "tensorflow-intel"),
        "keras": ("keras", "keras-cpu"),
        "xgboost": ("xgboost",),
    }
    versions = {"python": platform.python_version()}
    for key, candidates in aliases.items():
        found = None
        for dist in candidates:
            try:
                found = importlib_metadata.version(dist)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
        versions[key] = found
    return versions


def _json_sanitized(obj):
    """Make a structure JSON-safe: numpy scalars -> Python, non-finite -> null.

    A non-finite float is never a meaningful metric; writing it as JSON
    ``null`` keeps the manifest valid JSON and makes "undefined" explicit.
    """
    if isinstance(obj, dict):
        return {key: _json_sanitized(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitized(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _json_sanitized(obj.item())
    return obj


# ------------------------------------------------------------- cleaning -----
def _add_calendar_features(frame: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Append hour/weekday sin-cos pairs - deterministic functions of time."""
    out = frame.copy()
    stamps = out[datetime_column]
    hour = stamps.dt.hour.to_numpy(dtype="float64")
    weekday = stamps.dt.dayofweek.to_numpy(dtype="float64")
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return out


def _load_and_clean(cfg: V2Config, df: pd.DataFrame | None):
    """Sort, collapse duplicates per policy, coerce numerics. Never fill.

    Rows with missing values are KEPT: a missing observation is missing data,
    not a gap in time. Windows containing them are excluded later, by reason.
    Duplicate timestamps are collapsed per ``duplicate_policy`` ("mean"
    averages the available observations of each column, NaN-skipping; "first"
    keeps the first record; "error" refuses). The collapsed count is reported.
    """
    if df is None:
        df = pd.read_csv(cfg.data_path)
    if cfg.datetime_column not in df.columns:
        raise SchemaError(
            f"Column {cfg.datetime_column!r} not found. "
            f"Available: {list(df.columns)}")
    model_columns = [cfg.target_column, *cfg.exogenous_columns]
    absent = [c for c in model_columns if c not in df.columns]
    if absent:
        raise SchemaError(
            f"Declared column(s) {absent} not found. Available: {list(df.columns)}")

    working = df[[cfg.datetime_column, *model_columns]].copy()
    working[cfg.datetime_column] = pd.to_datetime(working[cfg.datetime_column])
    for column in model_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.sort_values(cfg.datetime_column).reset_index(drop=True)

    rows_raw = len(working)
    duplicated = int(working[cfg.datetime_column].duplicated().sum())
    if duplicated and cfg.duplicate_policy == "error":
        raise SchemaError(
            f"{duplicated} duplicate timestamp(s) found and "
            "duplicate_policy='error'.")
    if duplicated and cfg.duplicate_policy == "mean":
        working = (working
                   .groupby(cfg.datetime_column, as_index=False)[model_columns]
                   .mean())
    elif duplicated:  # "first"
        working = working.drop_duplicates(subset=[cfg.datetime_column],
                                          keep="first")
    working = working.sort_values(cfg.datetime_column).reset_index(drop=True)

    stamps = working[cfg.datetime_column]
    expected = len(pd.date_range(stamps.min(), stamps.max(), freq=cfg.cadence))
    quality = {
        "rows_raw": int(rows_raw),
        "rows_clean": int(len(working)),
        "duplicate_rows_collapsed": int(rows_raw - len(working)),
        "start": stamps.min().isoformat(),
        "end": stamps.max().isoformat(),
        "cadence": cfg.cadence,
        "expected_intervals": int(expected),
        "missing_intervals": int(expected - len(working)),
        "missing_values_by_column": {
            c: int(working[c].isna().sum()) for c in model_columns},
    }
    return working, quality


def _partition_rows(cfg: V2Config, stamps: pd.Series):
    """Assign every cleaned row to fit(0) / validation(1) / test(2).

    Ratios count cleaned rows; explicit timestamps cut at
    ``ts < fit_end`` / ``fit_end <= ts < validation_end`` / ``ts >=
    validation_end``. Either way the split is chronological and happens
    before any transform is fitted.
    """
    n = len(stamps)
    if cfg.fit_end is not None:
        fit_count = int(stamps.searchsorted(pd.Timestamp(cfg.fit_end), side="left"))
        val_count = int(stamps.searchsorted(pd.Timestamp(cfg.validation_end),
                                            side="left")) - fit_count
        rule = "explicit_timestamps"
    else:
        fit_count = int(n * cfg.fit_ratio)
        val_count = int(n * cfg.validation_ratio)
        rule = "ratios_over_cleaned_rows"
    test_count = n - fit_count - val_count
    if min(fit_count, val_count, test_count) < 1:
        raise PartitionError(
            f"Chronological split needs at least one row per partition; got "
            f"fit={fit_count}, validation={val_count}, test={test_count} "
            f"from {n} cleaned rows.")

    partition_of_row = np.concatenate([
        np.zeros(fit_count, dtype="int8"),
        np.ones(val_count, dtype="int8"),
        np.full(test_count, 2, dtype="int8"),
    ])
    boundaries = {
        "rule": rule,
        "fit": {"start": stamps.iloc[0].isoformat(),
                "end": stamps.iloc[fit_count - 1].isoformat(),
                "rows": int(fit_count)},
        "validation": {"start": stamps.iloc[fit_count].isoformat(),
                       "end": stamps.iloc[fit_count + val_count - 1].isoformat(),
                       "rows": int(val_count)},
        "test": {"start": stamps.iloc[fit_count + val_count].isoformat(),
                 "end": stamps.iloc[-1].isoformat(),
                 "rows": int(test_count)},
        "boundary_purge_rule": (
            "an example whose target horizon crosses a partition boundary is "
            "dropped entirely; rows from an earlier partition may still serve "
            "as input context"),
    }
    return partition_of_row, boundaries


# ------------------------------------------------------------- partitions ---
@dataclass
class PartitionData:
    """The eligible windows of one partition, already scaled."""

    name: str
    X: np.ndarray                # (n, sequence_length, n_features), float32
    y: np.ndarray                # (n, n_horizons), float32, scaled target
    context_end_stamps: np.ndarray   # (n,) datetime64 - last input step
    target_stamps: np.ndarray        # (n, n_horizons) datetime64
    end_index: np.ndarray            # (n,) row of first predicted step

    def __len__(self) -> int:
        return len(self.end_index)


@dataclass
class V2Data:
    """Everything downstream code needs, prepared causally."""

    cfg: V2Config
    series: pd.DataFrame             # cleaned, full, unscaled
    feature_names: list              # matrix column order, target first
    partitions: dict                 # name -> PartitionData
    feature_scaler: MinMaxScaler     # every input column, fit rows only
    target_scaler: MinMaxScaler      # target column only, fit rows only
    boundaries: dict
    coverage: dict                   # candidate/eligible/excluded counts
    quality: dict
    series_sha256: str


def prepare_v2(cfg: V2Config, df: pd.DataFrame | None = None) -> V2Data:
    """Causal preparation: split first, fit transforms on fit rows only.

    A candidate window ends at row ``i``: inputs are rows
    ``[i - sequence_length, i)`` and targets are rows ``i + h - 1`` for each
    horizon ``h``. A candidate is eligible only when

    * every step inside its span is exactly one cadence (no gap crossing),
    * every input row is complete across all model columns,
    * every target row has an observed target (a missing exogenous value on
      a *target* row does not matter - future weather is not an input), and
    * all its targets live in one partition (else it is purged).

    Nothing is imputed anywhere in this function.
    """
    frame, quality = _load_and_clean(cfg, df)
    if cfg.calendar_features:
        frame = _add_calendar_features(frame, cfg.datetime_column)
    feature_names = [cfg.target_column, *cfg.exogenous_columns]
    if cfg.calendar_features:
        feature_names += list(CALENDAR_COLUMNS)

    stamps = frame[cfg.datetime_column]
    partition_of_row, boundaries = _partition_rows(cfg, stamps)

    values = frame[feature_names].to_numpy(dtype="float64")
    fit_rows = partition_of_row == 0
    # MinMaxScaler disregards NaN in fit and preserves it in transform, so a
    # missing fit-period value neither shifts the scaling nor gets invented.
    feature_scaler = MinMaxScaler(feature_range=(0, 1)).fit(values[fit_rows])
    target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(values[fit_rows][:, :1])
    scaled = feature_scaler.transform(values)

    deltas = stamps.diff().dt.total_seconds().to_numpy()
    step_ok = np.isclose(deltas, cfg.cadence_seconds)
    step_ok[0] = True  # convention: nothing before row 0 to be discontinuous with
    input_complete = frame[feature_names].notna().all(axis=1).to_numpy()
    target_complete = frame[cfg.target_column].notna().to_numpy()

    length = cfg.sequence_length
    horizons = tuple(cfg.horizons)
    max_h = max(horizons)
    n = len(frame)
    ts_np = stamps.to_numpy()

    ends_by_partition: dict[str, list[int]] = {name: [] for name in PARTITIONS}
    exclusions: Counter = Counter()
    candidates = 0
    for i in range(length, n - max_h + 1):
        candidates += 1
        if not step_ok[i - length + 1: i + max_h].all():
            exclusions[REASON_NON_CADENCE] += 1
            continue
        if not input_complete[i - length: i].all():
            exclusions[REASON_MISSING_INPUT] += 1
            continue
        target_rows = [i + h - 1 for h in horizons]
        if not all(target_complete[r] for r in target_rows):
            exclusions[REASON_MISSING_TARGET] += 1
            continue
        parts = partition_of_row[target_rows]
        if int(parts.min()) != int(parts.max()):
            exclusions[REASON_BOUNDARY_PURGE] += 1
            continue
        ends_by_partition[PARTITIONS[int(parts[0])]].append(i)

    partitions = {}
    n_features = len(feature_names)
    for name in PARTITIONS:
        ends = ends_by_partition[name]
        if ends:
            X = np.stack([scaled[i - length: i] for i in ends]).astype("float32")
            y = np.asarray([[scaled[i + h - 1, 0] for h in horizons]
                            for i in ends], dtype="float32")
            ctx = ts_np[[i - 1 for i in ends]]
            tgt = ts_np[np.asarray([[i + h - 1 for h in horizons]
                                    for i in ends])]
            end_index = np.asarray(ends, dtype="int64")
        else:
            X = np.empty((0, length, n_features), dtype="float32")
            y = np.empty((0, len(horizons)), dtype="float32")
            ctx = np.empty((0,), dtype="datetime64[ns]")
            tgt = np.empty((0, len(horizons)), dtype="datetime64[ns]")
            end_index = np.empty((0,), dtype="int64")
        partitions[name] = PartitionData(name, X, y, ctx, tgt, end_index)

    coverage = {
        "candidate_windows": int(candidates),
        "eligible": {name: int(len(ends_by_partition[name])) for name in PARTITIONS},
        "excluded": {reason: int(exclusions.get(reason, 0))
                     for reason in EXCLUSION_REASONS},
        "policy": ("complete windows only; excluded windows are never imputed "
                   "or filled; each exclusion is counted once, by the first "
                   f"failed check in {list(EXCLUSION_REASONS)}"),
    }
    observed_columns = [cfg.target_column, *cfg.exogenous_columns]
    return V2Data(
        cfg=cfg,
        series=frame,
        feature_names=feature_names,
        partitions=partitions,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        boundaries=boundaries,
        coverage=coverage,
        quality=quality,
        series_sha256=_series_sha256(frame, cfg.datetime_column, observed_columns),
    )


# --------------------------------------------------------------- baselines --
def build_value_lookup(series: pd.DataFrame, datetime_column: str,
                       target_column: str) -> dict:
    """Timestamp -> observed target value (missing observations absent)."""
    return {t: float(v) for t, v in zip(series[datetime_column],
                                        series[target_column])
            if pd.notna(v)}


def persistence_baseline(series: pd.DataFrame, datetime_column: str,
                         target_column: str, context_end_stamps) -> np.ndarray:
    """"The future looks like the last hour we saw" - by timestamp."""
    lookup = build_value_lookup(series, datetime_column, target_column)
    out = np.empty(len(context_end_stamps), dtype="float64")
    for k, ctx in enumerate(context_end_stamps):
        out[k] = lookup.get(pd.Timestamp(ctx), np.nan)
    return out


def seasonal_baseline(series: pd.DataFrame, datetime_column: str,
                      target_column: str, target_stamps, context_end_stamps,
                      period: str = "24h") -> np.ndarray:
    """"The same clock hour one period ago", looked up by real timestamp.

    Row offsets are not wall-clock "yesterday" on an irregular series, so the
    lookup is ``target_timestamp - period``. When that observation is absent
    (a gap, or before the series start) the baseline falls back to
    persistence: the value at the end of the input context.
    """
    lookup = build_value_lookup(series, datetime_column, target_column)
    delta = pd.Timedelta(period)
    out = np.empty(len(target_stamps), dtype="float64")
    for k, (target, ctx) in enumerate(zip(target_stamps, context_end_stamps)):
        value = lookup.get(pd.Timestamp(target) - delta)
        if value is None:
            value = lookup.get(pd.Timestamp(ctx), np.nan)
        out[k] = value
    return out


# ---------------------------------------------------------------- metrics ---
def regression_metrics_v2(actual: np.ndarray, predicted: np.ndarray,
                          min_actual_for_mape: float = 1.0) -> dict:
    """MAE/RMSE/MAPE with an explicit undefined-value policy.

    MAPE is undefined when no actual exceeds ``min_actual_for_mape`` and is
    returned as ``None`` (JSON ``null``) rather than as a misleading number.
    An empty slice scores all-null instead of crashing report generation.
    """
    actual = np.asarray(actual, dtype="float64").ravel()
    predicted = np.asarray(predicted, dtype="float64").ravel()
    if actual.size == 0:
        return {"mae": None, "rmse": None, "mape": None, "n": 0}
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mask = np.abs(actual) > min_actual_for_mape
    mape = (float(np.mean(np.abs((actual[mask] - predicted[mask])
                                 / actual[mask])) * 100)
            if mask.any() else None)
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": int(actual.size)}


def improvement_pct(model_mae: float | None,
                    best_baseline_mae: float | None) -> float | None:
    """Percent improvement over a baseline; undefined over a zero-error one."""
    if model_mae is None or not best_baseline_mae:
        return None
    return round((best_baseline_mae - model_mae) / best_baseline_mae * 100, 2)


def _evaluate_partition(cfg: V2Config, data: V2Data, part: PartitionData,
                        model_predictions: dict) -> dict:
    """Score every model and both baselines on the same eligible windows."""
    block = {}
    for k, horizon in enumerate(cfg.horizons):
        actual = _inverse_target(data.target_scaler, part.y[:, k])
        persistence = persistence_baseline(
            data.series, cfg.datetime_column, cfg.target_column,
            part.context_end_stamps)
        seasonal = seasonal_baseline(
            data.series, cfg.datetime_column, cfg.target_column,
            part.target_stamps[:, k], part.context_end_stamps,
            period=cfg.seasonal_period)

        entry = {}
        for model_name, predicted_scaled in model_predictions.items():
            entry[model_name] = regression_metrics_v2(
                actual, _inverse_target(data.target_scaler,
                                        predicted_scaled[:, k]),
                cfg.mape_min_actual)
        entry["naive_persistence"] = regression_metrics_v2(
            actual, persistence, cfg.mape_min_actual)
        entry["naive_seasonal"] = regression_metrics_v2(
            actual, seasonal, cfg.mape_min_actual)

        naive_maes = [entry[name]["mae"]
                      for name in ("naive_persistence", "naive_seasonal")]
        naive_maes = [m for m in naive_maes if m is not None]
        best_naive = min(naive_maes) if naive_maes else None
        entry["improvement_over_best_naive_pct"] = {
            model_name: improvement_pct(entry[model_name]["mae"], best_naive)
            for model_name in model_predictions
        }
        block[f"h{horizon}"] = entry
    return block


# --------------------------------------------------------------- training ---
def _xgb_predict(estimator, X: np.ndarray) -> np.ndarray:
    """Predict honouring the early-stopped tree count, before and after reload."""
    best = getattr(estimator, "best_iteration", None)
    if best is not None:
        return estimator.predict(X, iteration_range=(0, int(best) + 1))
    return estimator.predict(X)


def train_v2(cfg: V2Config, df: pd.DataFrame | None = None,
             verbose: int = 1) -> dict:
    """Train the stacked LSTM and the XGBoost comparator on the same windows.

    Both models see exactly the same eligible fit/validation/test windows of
    the same causally prepared data. The LSTM validates on the explicit
    chronological validation partition (never Keras's ``validation_split``);
    XGBoost early-stops on the same partition. The package is written only
    under a run name that does not exist yet.
    """
    package_dir = Path(cfg.output_dir) / cfg.run_name
    if package_dir.exists():
        raise ArtifactExistsError(
            f"Run name {cfg.run_name!r} already exists at {package_dir}. "
            "Refusing to overwrite a versioned package; choose a new run_name.")

    data = prepare_v2(cfg, df)
    for name in PARTITIONS:
        if len(data.partitions[name]) == 0:
            raise PartitionError(
                f"Partition {name!r} has no eligible windows "
                f"(coverage: {data.coverage}). Training would be meaningless.")

    from .config import TrainingConfig  # local import: keeps TF out of module import
    from .model import build_model, default_callbacks, set_seeds

    set_seeds(cfg.seed)
    shim = TrainingConfig(
        sequence_length=cfg.sequence_length,
        horizons=tuple(cfg.horizons),
        lstm_units=tuple(cfg.lstm_units),
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        seed=cfg.seed,
        run_name=cfg.run_name,
    )
    fit, val, test = (data.partitions[name] for name in PARTITIONS)

    model = build_model(shim, n_features=len(data.feature_names))
    started = time.time()
    history = model.fit(
        fit.X, fit.y,
        validation_data=(val.X, val.y),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,               # temporal order is kept, always
        callbacks=default_callbacks(shim),
        verbose=verbose,
    )
    lstm_seconds = time.time() - started

    from xgboost import XGBRegressor

    xgb_models = {}
    xgb_seconds = 0.0
    for k, horizon in enumerate(cfg.horizons):
        estimator = XGBRegressor(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_learning_rate,
            max_depth=cfg.xgb_max_depth,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample,
            reg_lambda=1.0,
            early_stopping_rounds=cfg.xgb_early_stopping,
            eval_metric="mae",
            random_state=cfg.seed,
            n_jobs=1,                # deterministic
        )
        started = time.time()
        estimator.fit(fit.X.reshape(len(fit.X), -1), fit.y[:, k],
                      eval_set=[(val.X.reshape(len(val.X), -1), val.y[:, k])],
                      verbose=False)
        xgb_seconds += time.time() - started
        xgb_models[horizon] = estimator

    def predict_all(part: PartitionData) -> dict:
        flat = part.X.reshape(len(part.X), -1)
        return {
            "lstm": model.predict(part.X, verbose=0),
            "xgboost": np.column_stack([
                _xgb_predict(xgb_models[h], flat) for h in cfg.horizons]),
        }

    evaluation = {
        "validation": _evaluate_partition(cfg, data, val, predict_all(val)),
        "test": _evaluate_partition(cfg, data, test, predict_all(test)),
    }
    package = save_package(
        cfg, data, model, xgb_models,
        history={k: [float(v) for v in vals]
                 for k, vals in history.history.items()},
        evaluation=evaluation,
        training_seconds={"lstm": round(lstm_seconds, 1),
                          "xgboost": round(xgb_seconds, 1)},
    )
    if verbose:
        _print_summary(cfg, package["manifest"])
    return {
        "cfg": cfg,
        "data": data,
        "model": model,
        "xgb_models": xgb_models,
        "history": history.history,
        "evaluation": evaluation,
        "package_dir": package["package_dir"],
        "manifest": package["manifest"],
    }


def _print_summary(cfg: V2Config, manifest: dict) -> None:
    print("\n" + "=" * 62)
    print(f"  V2 RESULTS - {cfg.run_name}  (site: {cfg.site_scope})")
    print("=" * 62)
    for horizon in cfg.horizons:
        block = manifest["evaluation"]["test"][f"h{horizon}"]
        print(f"\n  Horizon +{horizon}")
        for name in ("lstm", "xgboost", "naive_persistence", "naive_seasonal"):
            m = block[name]
            mape = "  null" if m["mape"] is None else f"{m['mape']:5.1f}%"
            print(f"    {name:<18} MAE {m['mae']:8.1f}   RMSE {m['rmse']:8.1f}"
                  f"   MAPE {mape}")
    cov = manifest["coverage"]
    print(f"\n  Eligible windows: {cov['eligible']}; excluded: {cov['excluded']}")
    print(f"  Package: {Path(cfg.output_dir) / cfg.run_name}")
    print("=" * 62 + "\n")


# --------------------------------------------------------------- packaging --
def save_package(cfg: V2Config, data: V2Data, lstm_model, xgb_models: dict,
                 *, history: dict, evaluation: dict,
                 training_seconds: dict) -> dict:
    """Write the versioned artifact package. Existing run names are refused."""
    package_dir = Path(cfg.output_dir) / cfg.run_name
    if package_dir.exists():
        raise ArtifactExistsError(
            f"Run name {cfg.run_name!r} already exists at {package_dir}. "
            "Refusing to overwrite a versioned package; choose a new run_name.")
    package_dir.mkdir(parents=True)

    weights_path = package_dir / "model.keras"
    lstm_model.save(weights_path)
    models_meta = {
        "lstm": {"path": weights_path.name,
                 "kind": "keras-stacked-lstm",
                 "sha256": _sha256_file(weights_path)},
        "xgboost": {},
    }
    for horizon, estimator in xgb_models.items():
        path = package_dir / f"xgb_h{horizon}.json"
        estimator.save_model(path)
        best = getattr(estimator, "best_iteration", None)
        models_meta["xgboost"][f"h{horizon}"] = {
            "path": path.name,
            "kind": "xgboost-regressor",
            "trees_used": int(best) + 1 if best is not None else cfg.xgb_n_estimators,
            "sha256": _sha256_file(path),
        }

    derived = [c for c in data.feature_names
               if c not in [cfg.target_column, *cfg.exogenous_columns]]
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "run_name": cfg.run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site_scope": cfg.site_scope,
        "cadence": cfg.cadence,
        "cadence_seconds": cfg.cadence_seconds,
        "timestamp_convention": ("interval-end timestamps exactly as provided "
                                 "by the dataset; compared naively, no "
                                 "timezone conversion"),
        "units": dict(cfg.units),
        "schema": {
            "datetime_column": cfg.datetime_column,
            "target_column": cfg.target_column,
            "exogenous_columns": list(cfg.exogenous_columns),
            "feature_names": list(data.feature_names),
            "calendar_features": bool(cfg.calendar_features),
            "derived_features": derived,
            "known_future_inputs": ("calendar sin/cos features are functions "
                                    "of the timestamp only and are "
                                    "legitimately known at forecast time"),
        },
        "sequence_length": cfg.sequence_length,
        "horizons": list(cfg.horizons),
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "duplicate_policy": cfg.duplicate_policy,
        "splits": data.boundaries,
        "coverage": data.coverage,
        "data_quality": data.quality,
        "series_sha256": data.series_sha256,
        "scalers": {
            "target": scaler_to_dict(data.target_scaler),
            "features": scaler_to_dict(data.feature_scaler),
        },
        "models": models_meta,
        "weights_path": models_meta["lstm"]["path"],
        "dependencies": _runtime_versions(),
        "training": {
            "epochs_run": len(history.get("loss", [])),
            "training_seconds": training_seconds,
            "history": history,
            "lstm": {"lstm_units": list(cfg.lstm_units),
                     "dense_units": cfg.dense_units,
                     "dropout": cfg.dropout,
                     "learning_rate": cfg.learning_rate,
                     "batch_size": cfg.batch_size,
                     "patience": cfg.patience},
            "xgboost": {"n_estimators": cfg.xgb_n_estimators,
                        "max_depth": cfg.xgb_max_depth,
                        "learning_rate": cfg.xgb_learning_rate,
                        "subsample": cfg.xgb_subsample,
                        "colsample_bytree": cfg.xgb_colsample,
                        "early_stopping": cfg.xgb_early_stopping},
        },
        "evaluation": {
            "protocol": {
                "metrics": ["mae", "rmse", "mape"],
                "undefined_policy": ("percentage metrics with no actual above "
                                     "mape_min_actual, and improvements over a "
                                     "zero-error baseline, are undefined and "
                                     "serialise as JSON null"),
                "mape_min_actual": cfg.mape_min_actual,
                "baselines": ["naive_persistence",
                              f"naive_seasonal ({cfg.seasonal_period}, "
                              "timestamp-based, persistence fallback)"],
                "partition_rule": ("chronological fit/validation/test; scalers "
                                   "fitted on fit rows only; windows with "
                                   "missing or non-cadence steps excluded; "
                                   "examples whose targets cross a partition "
                                   "boundary purged"),
            },
            "validation": evaluation["validation"],
            "test": evaluation["test"],
        },
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitized(manifest), indent=2),
                             encoding="utf-8")
    return {"package_dir": package_dir, "manifest": manifest,
            "manifest_path": manifest_path}


class V2Package:
    """A loaded, hash-verified v2 package: scalers, schema and models."""

    def __init__(self, package_dir: Path, manifest: dict):
        self.package_dir = Path(package_dir)
        self.manifest = manifest
        self.cfg = V2Config(**manifest["config"])
        self.feature_names = list(manifest["schema"]["feature_names"])
        self.units = dict(manifest["units"])
        self.target_scaler = scaler_from_dict(manifest["scalers"]["target"])
        self.feature_scaler = scaler_from_dict(manifest["scalers"]["features"])
        self._lstm = None
        self._xgboost: dict = {}

    # -- lazy model loading ---------------------------------------------------
    def load_lstm(self):
        if self._lstm is None:
            from tensorflow.keras.models import load_model
            self._lstm = load_model(
                self.package_dir / self.manifest["models"]["lstm"]["path"])
        return self._lstm

    def load_xgboost(self, horizon: int):
        key = f"h{horizon}"
        if key not in self.manifest["models"]["xgboost"]:
            raise KeyError(f"No XGBoost model stored for horizon {horizon}; "
                           f"available: {list(self.manifest['models']['xgboost'])}")
        if key not in self._xgboost:
            from xgboost import XGBRegressor
            estimator = XGBRegressor()
            estimator.load_model(
                self.package_dir / self.manifest["models"]["xgboost"][key]["path"])
            self._xgboost[key] = estimator
        return self._xgboost[key]

    # -- inference -------------------------------------------------------------
    def build_input_window(self, recent_history: pd.DataFrame) -> np.ndarray:
        """Validate recent history and build the scaled (1, L, F) window.

        Refuses (GapHistoryError) when the used window contains a duplicate
        timestamp, a non-cadence step or a missing value - a final forecast
        is never produced from gapped history.
        """
        cfg = self.cfg
        frame = recent_history.copy()
        if cfg.datetime_column not in frame.columns:
            raise SchemaError(
                f"Column {cfg.datetime_column!r} not found in recent history.")
        required = [cfg.target_column, *cfg.exogenous_columns]
        absent = [c for c in required if c not in frame.columns]
        if absent:
            raise SchemaError(
                f"Recent history is missing declared column(s) {absent}; the "
                f"package schema requires {required} in units {self.units}.")
        frame[cfg.datetime_column] = pd.to_datetime(frame[cfg.datetime_column])
        frame = frame.sort_values(cfg.datetime_column).reset_index(drop=True)
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if frame[cfg.datetime_column].duplicated().any():
            raise GapHistoryError(
                "Recent history contains duplicate timestamps - ambiguous.")
        if len(frame) < cfg.sequence_length:
            raise GapHistoryError(
                f"Recent history has {len(frame)} rows; the model needs the "
                f"last {cfg.sequence_length}.")
        window = frame.iloc[-cfg.sequence_length:].reset_index(drop=True)

        deltas = window[cfg.datetime_column].diff().dt.total_seconds().to_numpy()[1:]
        if not np.isclose(deltas, cfg.cadence_seconds).all():
            raise GapHistoryError(
                f"Recent history is not {cfg.cadence}-consecutive over the "
                f"last {cfg.sequence_length} steps - refusing to forecast "
                "across a gap.")
        if window[required].isna().any().any():
            raise GapHistoryError(
                "Recent history contains missing values in modelled columns - "
                "refusing to forecast; v2 never imputes at serving time.")

        if cfg.calendar_features:
            window = _add_calendar_features(window, cfg.datetime_column)
        values = window[self.feature_names].to_numpy(dtype="float64")
        scaled = self.feature_scaler.transform(values)
        return scaled.reshape(1, cfg.sequence_length,
                            len(self.feature_names)).astype("float32")

    def forecast(self, recent_history: pd.DataFrame, model: str = "lstm",
                 declared_units: Mapping[str, str] | None = None) -> dict:
        """Multivariate-capable forecast with correctly timestamped horizons."""
        if declared_units is not None and dict(declared_units) != self.units:
            raise SchemaError(
                f"Declared units {dict(declared_units)} do not match the "
                f"package contract {self.units}.")
        window = self.build_input_window(recent_history)
        if model == "lstm":
            predicted_scaled = np.asarray(
                self.load_lstm().predict(window, verbose=0)[0], dtype="float64")
        elif model == "xgboost":
            flat = window.reshape(1, -1)
            predicted_scaled = np.asarray([
                _xgb_predict(self.load_xgboost(h), flat)[0]
                for h in self.cfg.horizons], dtype="float64")
        else:
            raise ValueError(f"unknown model {model!r}; use 'lstm' or 'xgboost'")
        predicted = _inverse_target(self.target_scaler, predicted_scaled)

        history = recent_history.copy()
        history[self.cfg.datetime_column] = pd.to_datetime(
            history[self.cfg.datetime_column])
        window_end = history[self.cfg.datetime_column].max()
        step = pd.Timedelta(self.cfg.cadence)
        forecasts = [
            {"horizon": int(h),
             "timestamp": (window_end + int(h) * step).isoformat(),
             "value": float(value),
             "units": self.units[self.cfg.target_column]}
            for h, value in zip(self.cfg.horizons, predicted)
        ]
        return {
            "pipeline_version": PIPELINE_VERSION,
            "run_name": self.cfg.run_name,
            "site_scope": self.cfg.site_scope,
            "model": model,
            "cadence": self.cfg.cadence,
            "window_end": window_end.isoformat(),
            "forecasts": forecasts,
        }


def load_package(package_dir: str | Path, verify: bool = True) -> V2Package:
    """Load a v2 package, verifying every model file's SHA256 first."""
    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {package_dir}. A v2 package is a directory "
            "written by save_package/train_v2; archived v1 artifacts are read "
            "with read_legacy_artifact.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("pipeline_version")
    if version != PIPELINE_VERSION:
        raise ValueError(
            f"{manifest_path} declares pipeline_version={version!r}, not "
            f"{PIPELINE_VERSION!r}. Refusing to load it as a v2 package.")
    if verify:
        entries = [manifest["models"]["lstm"],
                   *manifest["models"].get("xgboost", {}).values()]
        for entry in entries:
            path = package_dir / entry["path"]
            if not path.exists():
                raise ArtifactCorruptionError(
                    f"{path} is listed in the manifest but missing on disk.")
            actual = _sha256_file(path)
            if actual != entry["sha256"]:
                raise ArtifactCorruptionError(
                    f"SHA256 mismatch for {path.name}: manifest says "
                    f"{entry['sha256'][:12]}..., file hashes to "
                    f"{actual[:12]}... The artifact is corrupt or was "
                    "substituted; refusing to load.")
    return V2Package(package_dir, manifest)


def forecast(package_dir: str | Path, recent_history: pd.DataFrame,
             model: str = "lstm",
             declared_units: Mapping[str, str] | None = None) -> dict:
    """One-call inference: verify, load and forecast from recent history."""
    return load_package(package_dir).forecast(
        recent_history, model=model, declared_units=declared_units)


# ----------------------------------------------------------- legacy (v1) ----
def read_legacy_artifact(path: str | Path) -> dict:
    """Parse an archived v1 artifact with its original semantics (read-only).

    v1 artifacts carry no ``pipeline_version``; they are reported as "v1".
    ``metrics_only`` is True when the record references weights that were not
    retained on disk (the archived ``window_012``/``window_024`` runs) - such
    records still parse, but must never be loaded as models.
    """
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    version = record.get("pipeline_version", PIPELINE_VERSION_LEGACY)
    if version != PIPELINE_VERSION_LEGACY:
        raise ValueError(
            f"{path.name} declares pipeline_version={version!r}; use "
            "load_package for versioned packages.")
    weights = None
    model_path = record.get("model_path")
    if model_path:
        # The artifact's own directory first: archived absolute paths point at
        # the training machine and must never be silently substituted.
        candidates = [path.parent / Path(model_path).name, Path(model_path)]
        weights = next((c for c in candidates if c.exists()), None)
    return {
        "pipeline_version": PIPELINE_VERSION_LEGACY,
        "path": path,
        "record": record,
        "run_name": record.get("run_name"),
        "metrics_only": weights is None,
        "weights_path": weights,
    }


def load_legacy_model(path: str | Path):
    """Load a legacy v1 model; refuse metrics-only records.

    The nine archived runs with retained weights load normally. The
    metrics-only records (``window_012``, ``window_024``) raise
    :class:`MetricsOnlyArtifactError` instead of failing obscurely inside the
    model loader or, worse, being silently substituted.
    """
    info = read_legacy_artifact(path)
    if info["metrics_only"]:
        raise MetricsOnlyArtifactError(
            f"{info['run_name']!r} is a metrics-only record: its artifact "
            "references weights that were not retained on disk. Its metrics "
            "remain readable; it must not be loaded as a model.")
    from tensorflow.keras.models import load_model
    return load_model(info["weights_path"])
