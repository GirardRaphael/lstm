"""Turn the raw CSV into a feature matrix the LSTM can consume.

Version 1 of this project fed the network nothing but past traffic. That
isolates a clean question ("can traffic alone predict traffic?") but it also
guarantees the model is blind to the situations a traffic operator cares about
most: snow, rain, public holidays.

This module adds those columns back. Two families:

**Exogenous** - measured conditions that ship with the dataset:
`temp`, `rain_1h`, `snow_1h`, `clouds_all`.

**Calendar** - what the timestamp itself already tells us. Hour of day and day
of week are *cyclical*: hour 23 is adjacent to hour 0, and feeding the raw
integer would tell the network that 23 is maximally far from 0. Encoding each
as a (sin, cos) pair puts them on a circle, so midnight sits next to 23:00.

    hour_sin = sin(2*pi*hour/24)      hour_cos = cos(2*pi*hour/24)

The target always ends up in **column 0** of the matrix. That matters: a
MinMaxScaler scales every column independently, so column 0 of the scaled
matrix is exactly the target scaled by the target's own min/max, and the rest
of the pipeline can keep inverting it with a single-column scaler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the Metro Interstate dataset ships that are worth feeding in.
DEFAULT_EXOGENOUS = ("temp", "rain_1h", "snow_1h", "clouds_all")

# `rain_1h` contains one absurd outlier (9831.3 mm in an hour, a sensor fault).
# Left alone it dominates the MinMax scaling and squashes every real rain value
# to near zero, which would make the feature useless rather than helpful.
PHYSICAL_LIMITS = {
    "rain_1h": (0.0, 60.0),     # mm in one hour; >60 is not weather, it is a fault
    "snow_1h": (0.0, 30.0),     # mm water equivalent
    "temp": (200.0, 340.0),     # kelvin; the dataset uses 0 K for missing readings
    "clouds_all": (0.0, 100.0),  # percent
}


def _clip_to_physical(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible sensor readings with the nearest plausible value."""
    out = frame.copy()
    for column, (low, high) in PHYSICAL_LIMITS.items():
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            # A reading outside the physical range is a fault, not an extreme:
            # clamp it rather than deleting the whole hour.
            out[column] = values.clip(low, high).fillna(values.median())
    return out


def holiday_flag(frame: pd.DataFrame, datetime_column: str,
                 holiday_column: str = "holiday") -> pd.Series:
    """1.0 for every hour of a public holiday, 0.0 otherwise.

    The raw column only marks the *first* hour of a holiday; the other 23 hours
    read "None". Taken literally the feature would fire once a year per holiday
    and be useless, so we spread the flag across the whole calendar date.
    """
    if holiday_column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    marked = frame[holiday_column].astype("string").str.lower().ne("none") & \
        frame[holiday_column].notna()
    holiday_dates = set(frame.loc[marked, datetime_column].dt.date)
    return frame[datetime_column].dt.date.map(
        lambda d: 1.0 if d in holiday_dates else 0.0).astype("float64")


def build_feature_frame(
    df: pd.DataFrame,
    datetime_column: str = "date_time",
    target_column: str = "traffic_volume",
    exogenous: tuple = (),
    calendar: bool = False,
) -> pd.DataFrame:
    """Clean, deduplicate and assemble the feature matrix.

    Returns a frame whose columns are, in order:
    ``[datetime_column, target_column, *exogenous, *calendar features]``.
    """
    working = df.copy()
    working[datetime_column] = pd.to_datetime(working[datetime_column])

    available = [c for c in exogenous if c in working.columns]
    missing = [c for c in exogenous if c not in working.columns]
    if missing:
        raise KeyError(
            "Requested feature(s) not in the file: {}. Available: {}".format(
                missing, list(working.columns)))

    working = _clip_to_physical(working)

    if calendar:
        working["_is_holiday"] = holiday_flag(working, datetime_column)

    numeric = [target_column] + available + (["_is_holiday"] if calendar else [])
    for column in numeric:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    # One row per hour. Duplicated timestamps are averaged, exactly as in the
    # univariate path, so the two versions stay comparable.
    frame = (working[[datetime_column] + numeric]
             .dropna(subset=[target_column])
             .sort_values(datetime_column)
             .groupby(datetime_column, as_index=False)[numeric]
             .mean()
             .reset_index(drop=True))

    # Any remaining hole in a weather column is filled from the nearest reading
    # rather than dropped - losing an hour of traffic to a missing cloud
    # percentage would be a bad trade.
    for column in numeric:
        frame[column] = frame[column].ffill().bfill()

    if calendar:
        stamps = frame[datetime_column]
        hour = stamps.dt.hour.to_numpy(dtype="float64")
        weekday = stamps.dt.dayofweek.to_numpy(dtype="float64")
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        frame["is_weekend"] = (weekday >= 5).astype("float64")
        frame = frame.rename(columns={"_is_holiday": "is_holiday"})

    return frame


def feature_columns(frame: pd.DataFrame, datetime_column: str) -> list:
    """Every column of the matrix except the timestamp, target first."""
    return [c for c in frame.columns if c != datetime_column]


def describe_features(frame: pd.DataFrame, datetime_column: str) -> list:
    """Human-readable notes for the report and the vault."""
    descriptions = {
        "traffic_volume": "vehicles counted during the hour (the target)",
        "temp": "temperature in kelvin",
        "rain_1h": "rainfall in the hour, mm (clipped at 60 to kill a sensor fault)",
        "snow_1h": "snowfall in the hour, mm water equivalent",
        "clouds_all": "cloud cover, percent",
        "is_holiday": "1 for every hour of a public holiday",
        "hour_sin": "hour of day, sine component (so 23:00 sits next to 00:00)",
        "hour_cos": "hour of day, cosine component",
        "weekday_sin": "day of week, sine component",
        "weekday_cos": "day of week, cosine component",
        "is_weekend": "1 on Saturday and Sunday",
    }
    return [(c, descriptions.get(c, "")) for c in feature_columns(frame, datetime_column)]
