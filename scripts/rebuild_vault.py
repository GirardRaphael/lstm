"""Regenerate the Obsidian vault from an already-trained model.

Training takes ~18 minutes; rebuilding the vault takes seconds. Use this after
editing anything under `obsidian_export.py` or `obsidian_canvas.py`.

    python scripts/rebuild_vault.py
    python scripts/rebuild_vault.py --run-name multi_horizon --no-timeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorflow.keras.models import load_model  # noqa: E402

from traffic_lstm.config import TrainingConfig  # noqa: E402
from traffic_lstm.data import build_datasets  # noqa: E402
from traffic_lstm.obsidian_export import export_vault  # noqa: E402
from traffic_lstm.plots import (  # noqa: E402
    plot_baseline_comparison,
    plot_daily_profile,
    plot_error_distribution,
    plot_history,
    plot_predictions,
    plot_series,
)
from traffic_lstm.data import inverse_target  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline_univariate")
    parser.add_argument("--no-timeline", action="store_true",
                        help="Skip the 24 hour-by-hour canvases.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Reuse the figures already in reports/figures/.")
    args = parser.parse_args(argv)

    probe = TrainingConfig(run_name=args.run_name)
    if not probe.artifact_path.exists():
        raise SystemExit(
            "No artifacts for run '{}'. Train first:\n"
            "    python -m traffic_lstm.train --run-name {}".format(
                args.run_name, args.run_name))

    artifacts = json.loads(probe.artifact_path.read_text(encoding="utf-8"))
    cfg = TrainingConfig(**artifacts["config"])
    print("Loading {} …".format(cfg.model_path))
    model = load_model(cfg.model_path)
    bundle = build_datasets(cfg)

    if not args.no_figures:
        predicted = inverse_target(bundle.scaler, model.predict(bundle.X_test, verbose=0))
        actual = inverse_target(bundle.scaler, bundle.y_test)
        plot_series(bundle.series[cfg.datetime_column].to_numpy(),
                    bundle.series[cfg.target_column].to_numpy())
        plot_daily_profile(bundle.series, cfg.datetime_column, cfg.target_column)
        plot_history(artifacts["history"])
        plot_predictions(actual[:, 0], predicted[:, 0], horizon=cfg.horizons[0],
                         timestamps=bundle.test_timestamps.to_numpy())
        plot_error_distribution(actual[:, 0], predicted[:, 0])
        plot_baseline_comparison(artifacts["results"]["h{}".format(cfg.horizons[0])])

    export_vault(cfg, model, bundle, artifacts, timeline=not args.no_timeline)


if __name__ == "__main__":
    main()
