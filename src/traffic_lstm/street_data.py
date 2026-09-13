"""Read-only preflight for one street observation stream, never a controller.

Timestamps label the END of completed intervals and must include a UTC offset.
Passing these checks means structurally usable for analysis, not that a sensor
is accurate or that any existing model is suitable for the street.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationContract:
    stream_id: str
    interval_seconds: int = 300
    minimum_samples: int = 24
    max_age_seconds: int = 600

    def __post_init__(self):
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must identify one site/approach/movement/sensor stream")
        for name in ("interval_seconds", "minimum_samples", "max_age_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _aware_timestamp(value):
    if not isinstance(value, (str, pd.Timestamp)):
        raise ValueError("Expected a timestamp with an explicit UTC offset")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamp must include an explicit UTC offset")
    return stamp.tz_convert("UTC")


def inspect_observations(frame: pd.DataFrame, contract: ObservationContract, *, now=None) -> dict:
    """Reject bad input; never silently sort, merge, interpolate or substitute.

    Validate all supplied rows. A caller building rolling windows must isolate
    one stream and define its cadence before calling this function.
    """
    clock = pd.Timestamp.now(tz="UTC") if now is None else _aware_timestamp(now)
    report = {
        "schema_version": 1, "mode": "read_only_preflight",
        "status": "blocked", "valid_for_analysis": False,
        "street_deployment_approved": False,
        "checked_at": clock.isoformat(), "contract": asdict(contract),
        "rows": len(frame), "latest_interval_end": None, "age_seconds": None,
        "issues": [],
        "limitations": [
            "No sensor accuracy or model suitability has been established.",
            "Hourly motorway models must not be relabelled as five-minute street models.",
            "No signal-control actions or congestion diagnoses are produced.",
        ],
    }
    issues = report["issues"]

    def issue(code, detail):
        issues.append({"code": code, "detail": detail})

    required = {"stream_id", "timestamp", "interval_seconds", "vehicle_count"}
    if not frame.columns.is_unique:
        issue("duplicate_columns", "Column names must be unique")
        return report
    missing = sorted(required - set(frame.columns))
    if missing:
        issue("missing_columns", ", ".join(missing))
        return report
    if len(frame) < contract.minimum_samples:
        issue("insufficient_history", f"Need at least {contract.minimum_samples} completed intervals")
    ids = frame["stream_id"]
    if ids.isna().any() or not ids.eq(contract.stream_id).all():
        issue("stream_mismatch", "Every row must belong to the declared stream")
    interval = pd.to_numeric(frame["interval_seconds"], errors="coerce")
    if interval.isna().any() or not interval.eq(contract.interval_seconds).all():
        issue("interval_mismatch", "Reported aggregation interval differs from the contract")
    counts = pd.to_numeric(frame["vehicle_count"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        issue("invalid_counts", "Raw vehicle counts must be finite non-negative integers")

    try:
        stamps = pd.DatetimeIndex([_aware_timestamp(value) for value in frame["timestamp"]])
    except (ValueError, TypeError, OverflowError):
        issue("invalid_timestamps", "Use ISO timestamps with UTC offsets; missing/naive dates are rejected")
        return report
    if len(stamps):
        if stamps.has_duplicates:
            issue("duplicate_timestamps", "Duplicate interval ends must be resolved at ingestion")
        if not stamps.is_monotonic_increasing:
            issue("out_of_order", "Intervals must arrive in strictly increasing time order")
        deltas = (stamps[1:] - stamps[:-1]).total_seconds()
        if len(deltas) and not np.equal(deltas, contract.interval_seconds).all():
            issue("non_contiguous", "Missing/overlapping intervals: do not treat row count as elapsed time")
        latest = stamps.max()
        age = float((clock - latest).total_seconds())
        report.update(latest_interval_end=latest.isoformat(), age_seconds=age)
        if age < 0:
            issue("future_observation", "An interval ending in the future is not a completed observation")
        elif age > contract.max_age_seconds:
            issue("stale_observations", "Latest completed interval is too old for this analysis window")
    report["valid_for_analysis"] = not issues
    report["status"] = "valid_for_analysis" if not issues else "blocked"
    return report


def demo_observations(now=None) -> pd.DataFrame:
    """Synthetic demonstration only; never represents a real street feed."""
    clock = pd.Timestamp.now(tz="UTC") if now is None else _aware_timestamp(now)
    return pd.DataFrame({
        "stream_id": "DEMO-NOT-A-REAL-STREET",
        "timestamp": pd.date_range(end=clock.floor("5min"), periods=24, freq="5min").astype(str),
        "interval_seconds": 300,
        "vehicle_count": [10 + (i % 8) * 2 for i in range(24)],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--stream-id")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--minimum-samples", type=int, default=24)
    parser.add_argument("--max-age-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.csv and not args.stream_id:
        parser.error("--csv requires --stream-id")
    try:
        contract = (ObservationContract("DEMO-NOT-A-REAL-STREET") if args.demo else
                    ObservationContract(args.stream_id, args.interval_seconds,
                                        args.minimum_samples, args.max_age_seconds))
        frame = demo_observations() if args.demo else pd.read_csv(args.csv)
        report = inspect_observations(frame, contract)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report["synthetic_demo"] = args.demo
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["valid_for_analysis"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
