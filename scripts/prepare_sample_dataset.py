"""Build a second, unrelated dataset so the pipeline can be tested on it.

The whole project claims to work on "any CSV with a timestamp column and a
numeric column". A claim like that is worth nothing until it has been run on
data it was not designed around, so this prepares the UCI *Bike Sharing*
dataset: hourly bike rentals in Washington DC, 2011-2012.

Different city, different quantity, different scale (rentals peak around 1,000
an hour, motorway traffic around 7,000), and a different failure mode - bike
demand collapses in winter, which motorway traffic does not.

The raw file splits the timestamp across `dteday` and `hr`, so this rebuilds a
single `date_time` column and renames the columns to something readable.

    python scripts/prepare_sample_dataset.py
    python scripts/prepare_sample_dataset.py --source path/to/hour.csv
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "samples" / "bike_sharing_hourly.csv"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

WEATHER = {1: "clear", 2: "misty", 3: "light_rain_snow", 4: "heavy_rain_snow"}


def fetch_raw() -> pd.DataFrame:
    print("Downloading the Bike Sharing dataset from UCI ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        with archive.open("hour.csv") as handle:
            return pd.read_csv(handle)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame["date_time"] = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    frame["rentals"] = raw["cnt"].astype("int64")
    frame["casual"] = raw["casual"].astype("int64")
    frame["registered"] = raw["registered"].astype("int64")
    # The raw weather columns are normalised to 0-1; convert back to real units
    # so the file is readable and the feature clipping in features.py is not
    # tripped by values that look like faults.
    frame["temp_c"] = (raw["temp"] * 47 - 8).round(2)          # documented scaling
    frame["feels_like_c"] = (raw["atemp"] * 66 - 16).round(2)
    frame["humidity"] = (raw["hum"] * 100).round(1)
    frame["windspeed"] = (raw["windspeed"] * 67).round(2)
    frame["weather"] = raw["weathersit"].map(WEATHER).fillna("unknown")
    frame["is_holiday"] = raw["holiday"].astype("int64")
    frame["is_workingday"] = raw["workingday"].astype("int64")
    return frame.sort_values("date_time").reset_index(drop=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="A local hour.csv, if you already have it.")
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args(argv)

    raw = pd.read_csv(args.source) if args.source else fetch_raw()
    frame = prepare(raw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    stamps = frame["date_time"]
    expected = pd.date_range(stamps.min(), stamps.max(), freq="h")
    print("\nWrote {}".format(args.out))
    print("  rows          : {:,}".format(len(frame)))
    print("  period        : {} -> {}".format(stamps.min(), stamps.max()))
    print("  target        : rentals  ({:,} to {:,}, mean {:,.0f})".format(
        frame["rentals"].min(), frame["rentals"].max(), frame["rentals"].mean()))
    print("  missing hours : {:,} of {:,}".format(
        len(expected) - stamps.nunique(), len(expected)))
    print("  columns       : {}".format(", ".join(frame.columns)))
    print("\nTrain on it with:")
    print("  python -m traffic_lstm.train --run-name bike_sharing \\")
    print("      --data {} \\".format(args.out.as_posix()))
    print("      --target rentals --epochs 40")


if __name__ == "__main__":
    main()
