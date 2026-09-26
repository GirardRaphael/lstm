"""Evaluation helpers: fit-only climatology and time-aware uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_mean(fit_frame, datetime_column, target_column, target_stamps):
    """Fit hour-of-week means on fit rows; unseen slots use the fit mean."""
    stamps = pd.to_datetime(fit_frame[datetime_column])
    values = fit_frame[target_column]
    means = values.groupby(stamps.dt.dayofweek * 24 + stamps.dt.hour).mean()
    target = pd.DatetimeIndex(target_stamps)
    fallback = values.mean()
    means = means.fillna(fallback)
    return np.array([
        means.get(int(day * 24 + hour), fallback)
        for day, hour in zip(target.dayofweek, target.hour)
    ], dtype=float)


def block_interval(values, stamps, *, seed=42, repeats=500, block_hours=24):
    """Percentile CI of a mean, resampling whole observed calendar-day blocks.

    Gaps are never interpreted as consecutive hours. This describes uncertainty
    conditional on one fitted model and this test period, not training variability.
    """
    values = np.asarray(values, dtype=float)
    stamps = pd.DatetimeIndex(stamps)
    if len(values) != len(stamps) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires aligned finite values and timestamps")
    groups = pd.DataFrame({"value": values, "block": stamps.floor(f"{block_hours}h")})
    blocks = groups.groupby("block")["value"].agg(["sum", "count"])
    if len(blocks) < 2:
        return None
    rng = np.random.default_rng(seed)
    sums, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    draws = [rng.integers(0, len(blocks), len(blocks)) for _ in range(repeats)]
    means = [sums[d].sum() / counts[d].sum() for d in draws]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"low": float(lo), "high": float(hi), "confidence": 0.95,
            "blocks": len(blocks), "block_hours": block_hours,
            "replicates": repeats, "seed": seed}


def diagnostic_metrics(actual, predicted, stamps, *, seed=42):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = predicted - actual
    stamps = pd.DatetimeIndex(stamps)
    peak = (stamps.dayofweek < 5) & np.isin(stamps.hour, [7, 8, 9, 16, 17, 18])
    return {"bias": float(error.mean()) if len(error) else None,
            "peak_hour_mae": float(np.abs(error[peak]).mean()) if peak.any() else None,
            "peak_hour_n": int(peak.sum()),
            "mae_ci": block_interval(np.abs(error), stamps, seed=seed)}


def direct_features(series, datetime_column, target_column, target_stamps, context_stamps):
    """Timestamp lags t-1/t-24/t-168 and target clock; unavailable lags stay NaN.

    A lag after the forecast origin is unavailable, including for longer horizons.
    XGBoost handles missing values; no observations are invented to fill outages.
    """
    lookup = series.set_index(datetime_column)[target_column]
    target = pd.DatetimeIndex(target_stamps)
    context = pd.DatetimeIndex(context_stamps)
    columns = []
    for hours in (1, 24, 168):
        source = target - pd.Timedelta(hours=hours)
        values = lookup.reindex(source).to_numpy(dtype=float, copy=True)
        values[source > context] = np.nan
        columns.append(values)
    for clock, period in ((target.hour, 24), (target.dayofweek, 7)):
        columns.extend([np.sin(2 * np.pi * clock / period),
                        np.cos(2 * np.pi * clock / period)])
    return np.column_stack(columns)
