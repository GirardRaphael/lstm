"""Reproducible v2 comparison. Run python -m traffic_lstm.research --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT as ROOT
from .pipeline_v2 import V2Config, train_v2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["motorway", "bikes"], default="motorway")
    parser.add_argument("--run-name", required=True, help="Unique immutable run directory name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tree-layout", choices=["flattened", "direct_lags"], default="direct_lags")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    bike = args.dataset == "bikes"
    target = "rentals" if bike else "traffic_volume"
    cfg = V2Config(
        data_path=ROOT / ("data/samples/bike_sharing_hourly.csv" if bike else
                          "data/raw/Metro_Interstate_Traffic_Volume.csv"),
        target_column=target, units={target: "rentals/hour" if bike else "vehicles/hour"},
        site_scope="washington-dc-bike-aggregate" if bike else "i94-westbound-uci",
        run_name=args.run_name, calendar_features=True,
        fit_ratio=0.64, validation_ratio=0.16, sequence_length=24,
        epochs=args.epochs, patience=8, seed=args.seed, batch_size=128,
        xgb_n_estimators=1000, xgb_early_stopping=40, xgb_layout=args.tree_layout)
    # The experiment contract is saved BEFORE training or test scoring.
    protocol_dir = ROOT / "reports" / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = protocol_dir / f"{cfg.run_name}.json"
    with protocol_path.open("x", encoding="utf-8") as stream:
        json.dump({"config": cfg.to_dict(), "selection": "lowest validation MAE including baselines",
                   "status": "retrospective reanalysis; historical test period previously inspected",
                   "uncertainty": "paired calendar-day bootstrap; one training seed",
                   "scope": "hourly counts with calendar; no weather ablation; no traffic control"}, stream, indent=2)
    result = train_v2(cfg, verbose=2)
    print(f"Validation-selected candidate: {result['manifest']['selected_on_validation']}")
    print(f"Evidence: {result['package_dir'] / 'manifest.json'}")


if __name__ == "__main__":
    main()
