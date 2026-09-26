"""Strict interval-end count schema for the local observatory."""
from io import BytesIO
import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 200_000
REQUIRED = ["stream_id", "timestamp", "interval_seconds", "vehicle_count"]


def validate_csv(content: bytes, *, now=None):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV must be nonempty and at most 10 MiB")
    frame = pd.read_csv(BytesIO(content), dtype={"stream_id": "string"})
    if not set(REQUIRED).issubset(frame.columns) or not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError("Need 1–200000 rows and columns: " + ", ".join(REQUIRED))
    frame = frame[REQUIRED].copy()
    if frame.stream_id.isna().any() or frame.stream_id.nunique() != 1:
        raise ValueError("An import must contain exactly one nonmissing stream_id")
    stream = str(frame.stream_id.iloc[0])
    if not stream.strip() or len(stream) > 200:
        raise ValueError("stream_id must have 1–200 characters")
    # Require a timezone on EACH original timestamp. utc=True alone would silently
    # accept naive timestamps, which is not an acceptable source contract.
    parsed = []
    for value in frame.timestamp:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("Every timestamp needs an explicit UTC offset")
        parsed.append(stamp.tz_convert("UTC"))
    stamps = pd.DatetimeIndex(parsed)
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be strictly increasing, without duplicates")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if (stamps > current).any():
        raise ValueError("Future or unfinished intervals are not observations")
    for column in ("interval_seconds", "vehicle_count"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain finite nonnegative integers")
        if (values > 2**31-1).any():
            raise ValueError(f"{column} exceeds the supported integer range")
        frame[column] = values.astype("int64")
    if frame.interval_seconds.nunique() != 1 or frame.interval_seconds.iloc[0] <= 0:
        raise ValueError("One positive interval_seconds value is required")
    cadence = int(frame.interval_seconds.iloc[0])
    differences = stamps.to_series().diff().dropna().dt.total_seconds().to_numpy()
    if len(differences) and not np.isclose(differences % cadence, 0).all():
        raise ValueError("Timestamps are off the declared interval grid")
    missing = int(np.sum(differences / cadence - 1))
    frame["timestamp"] = stamps
    return frame, {"stream_id": stream, "rows": len(frame), "interval_seconds": cadence,
                   "start": stamps[0].isoformat(), "end": stamps[-1].isoformat(),
                   "missing_intervals": missing, "coverage": len(frame)/(len(frame)+missing),
                   "age_seconds": float((current-stamps[-1]).total_seconds()),
                   "timestamp_convention": "completed interval end, explicit offset converted to UTC",
                   "source_identity": "declared by importer; not independently certified"}


def demo_csv(hours=2400):
    """Synthetic, explicitly named demo. Not an authorized sensor feed."""
    stamps = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=hours, freq="h")
    rng = np.random.default_rng(42)
    counts = np.maximum(0, 200 + 130*np.sin(2*np.pi*stamps.hour/24) + rng.normal(0, 12, hours)).astype(int)
    return pd.DataFrame({"stream_id": "synthetic-demo/hourly", "timestamp": stamps,
                         "interval_seconds": 3600, "vehicle_count": counts}).to_csv(index=False).encode()
