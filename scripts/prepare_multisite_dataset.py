"""Build a multi-series dataset: 12 air-quality stations, one row per hour.

Everything measured so far used a single series, which is the case a tree
ensemble handles best. The standard argument for a recurrent model is that it
should pull ahead when **several correlated series share one model** - the
recurrence learns a representation across them, while a tree gets one flat
column per (series x timestep) and has to rediscover the structure.

The UCI *Beijing Multi-Site Air-Quality* data is a clean test of exactly that:
12 monitoring stations across one city, the same quantity at each, hourly for
four years. Pollution moves across the city, so the stations are genuinely
correlated rather than independent.

The output is deliberately shaped to need **no pipeline changes at all**: one
wide CSV whose columns are the stations, so `--target` picks the station to
predict and `--exogenous` hands the others in as extra input channels.

    python scripts/prepare_multisite_dataset.py --source path/to/PRSA_Data_...
    python scripts/prepare_multisite_dataset.py            # downloads from UCI
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "samples" / "beijing_pm25_multisite.csv"
SOURCE_URL = ("https://archive.ics.uci.edu/static/public/501/"
              "beijing+multi+site+air+quality+data.zip")
POLLUTANT = "PM2.5"


def fetch_frames() -> dict:
    """Download the nested archive and return {station: DataFrame}."""
    print("Downloading the Beijing Multi-Site Air-Quality data from UCI ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=300) as response:
        outer_bytes = response.read()

    frames = {}
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        inner_name = next(n for n in outer.namelist() if n.lower().endswith(".zip"))
        with outer.open(inner_name) as handle:
            inner_bytes = handle.read()
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for name in inner.namelist():
            if not name.endswith(".csv"):
                continue
            with inner.open(name) as handle:
                frame = pd.read_csv(handle)
            frames[str(frame["station"].iloc[0])] = frame
    return frames


def read_local(folder: Path) -> dict:
    frames = {}
    for path in sorted(folder.glob("PRSA_Data_*.csv")):
        frame = pd.read_csv(path)
        frames[str(frame["station"].iloc[0])] = frame
    if not frames:
        raise SystemExit("No PRSA_Data_*.csv found in {}".format(folder))
    return frames


def build_wide(frames: dict, pollutant: str = POLLUTANT) -> pd.DataFrame:
    """One column per station, one row per hour."""
    columns = {}
    for station, frame in sorted(frames.items()):
        stamps = pd.to_datetime(dict(year=frame["year"], month=frame["month"],
                                     day=frame["day"], hour=frame["hour"]))
        series = pd.Series(pd.to_numeric(frame[pollutant], errors="coerce").to_numpy(),
                           index=stamps, name=station)
        columns[station] = series[~series.index.duplicated(keep="first")]

    wide = pd.DataFrame(columns).sort_index()
    wide.index.name = "date_time"

    # Each station has scattered missing readings. Interpolating a *feature*
    # over short gaps is reasonable - it is a real measurement that failed, not
    # an invented event - but only over short gaps, and the target column is
    # left with its holes so no fabricated value is ever scored.
    limit = 3
    filled = wide.interpolate(method="time", limit=limit, limit_direction="both")
    return filled.reset_index()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="Folder holding the 12 PRSA_Data_*.csv files.")
    parser.add_argument("--out", type=Path, default=TARGET)
    parser.add_argument("--pollutant", default=POLLUTANT)
    args = parser.parse_args(argv)

    frames = read_local(args.source) if args.source else fetch_frames()
    wide = build_wide(frames, args.pollutant)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out, index=False)

    stations = [c for c in wide.columns if c != "date_time"]
    stamps = wide["date_time"]
    expected = pd.date_range(stamps.min(), stamps.max(), freq="h")
    correlations = wide[stations].corr().to_numpy()
    off_diagonal = correlations[~np.eye(len(stations), dtype=bool)]

    print("\nWrote {}".format(args.out))
    print("  stations      : {}".format(len(stations)))
    print("  rows          : {:,}".format(len(wide)))
    print("  period        : {} -> {}".format(stamps.min(), stamps.max()))
    print("  missing hours : {:,} of {:,}".format(
        len(expected) - stamps.nunique(), len(expected)))
    print("  remaining NaNs: {:,} ({:.2%} of cells)".format(
        int(wide[stations].isna().sum().sum()),
        wide[stations].isna().sum().sum() / (len(wide) * len(stations))))
    print("  cross-station correlation: min {:.2f}, mean {:.2f}, max {:.2f}".format(
        off_diagonal.min(), off_diagonal.mean(), off_diagonal.max()))
    print("\n  {} columns: {}".format(args.pollutant, ", ".join(stations)))

    first, rest = stations[0], stations[1:]
    print("\nSingle-series run (one station only):")
    print("  python -m traffic_lstm.train --run-name air_single \\")
    print("      --data {} \\".format(args.out.as_posix()))
    print("      --target {} --calendar".format(first))
    print("\nMulti-series run (all 12 stations as input channels):")
    print("  python -m traffic_lstm.train --run-name air_multi \\")
    print("      --data {} \\".format(args.out.as_posix()))
    print("      --target {} --calendar \\".format(first))
    print("      --exogenous {}".format(" ".join(rest)))


if __name__ == "__main__":
    main()
