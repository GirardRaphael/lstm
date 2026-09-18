"""Load a trained model and forecast the next hour.

    python -m traffic_lstm.predict --last-hours
    python -m traffic_lstm.predict --values 2150 2430 3020 ... (24 numbers)
    python -m traffic_lstm.predict --run-name v2/my_run --last-hours   (v2 package)

Routing: an artifact's ``pipeline_version`` decides the loader (archived
artifacts carry no such key and are v1). v1 forecasts from bare values; a v2
package forecasts from timestamped, cadence-checked recent history only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import MODEL_DIR, TrainingConfig
from .evaluate import traffic_level


class TrafficForecaster:
    """A trained model plus the scaler it was trained with."""

    def __init__(self, model, scaler, cfg: TrainingConfig):
        self.model = model
        self.scaler = scaler
        self.cfg = cfg

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, run_name: str = "baseline_univariate") -> "TrafficForecaster":
        from tensorflow.keras.models import load_model

        from .train import scaler_from_dict

        cfg = TrainingConfig(run_name=run_name)
        if not cfg.artifact_path.exists():
            raise FileNotFoundError(
                "No artifacts for run '{}'. Train first: python -m traffic_lstm.train"
                .format(run_name))
        artifacts = json.loads(cfg.artifact_path.read_text(encoding="utf-8"))
        cfg = TrainingConfig(**artifacts["config"])
        return cls(load_model(cfg.model_path), scaler_from_dict(artifacts["scaler"]), cfg)

    # -- prediction ----------------------------------------------------------
    def predict(self, last_hours) -> dict:
        """Forecast from the last `sequence_length` traffic volumes."""
        expected_features = self.model.input_shape[-1]
        if expected_features != 1:
            raise ValueError(
                "Run '{}' was trained on {} input features ({}). This CLI only "
                "feeds past traffic, so it cannot drive that model. Multivariate "
                "inference requires the original feature order and feature scaler.".format(
                    self.cfg.run_name, expected_features,
                    ", ".join(self.cfg.exogenous_columns) or "see config"))

        values = np.asarray(last_hours, dtype="float64")
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Supply a one-dimensional sequence of finite numeric values.")
        if len(values) != self.cfg.sequence_length:
            raise ValueError(
                "Expected exactly {} values, got {}.".format(
                    self.cfg.sequence_length, len(values)))

        scaled = self.scaler.transform(values.reshape(-1, 1))
        window = scaled.reshape(1, self.cfg.sequence_length, 1)
        predicted_scaled = self.model.predict(window, verbose=0)[0]
        predicted = self.scaler.inverse_transform(
            np.asarray(predicted_scaled).reshape(-1, 1)).ravel()
        if len(predicted) != self.cfg.n_outputs or not np.isfinite(predicted).all():
            raise ValueError("Model produced invalid predictions; no forecast is available.")

        forecasts = []
        for horizon, value in zip(self.cfg.horizons, predicted):
            level, comment = traffic_level(value, self.cfg.level_low, self.cfg.level_high)
            forecasts.append({"horizon": int(horizon), "volume": float(value),
                              "level": level, "comment": comment})
        return {"input": values.tolist(), "forecasts": forecasts}

    def render(self, result: dict, ends_at: str | None = None) -> str:
        """The block to show on screen during the demo."""
        lines = ["=" * 46, "   TRAFFIC FORECAST".center(46), "=" * 46, ""]
        lines.append("  History analysed : last {} hours".format(self.cfg.sequence_length))
        if ends_at:
            lines.append("  Window ends      : {}".format(ends_at))
        recent = result["input"][-6:]
        lines.append("  Last 6 hours     : " + "  ".join("{:,.0f}".format(v) for v in recent))
        lines.append("")
        for f in result["forecasts"]:
            lines.append("  +{}h  ->  {:>7,.0f} vehicles   [{}]".format(
                f["horizon"], f["volume"], f["level"]))
            lines.append("          {}".format(f["comment"]))
        lines += ["", "=" * 46]
        return "\n".join(lines)


class V2Forecaster:
    """A hash-verified v2 package behind a forecast-and-render interface."""

    def __init__(self, package):
        self.package = package
        self.cfg = package.cfg          # the V2Config recorded in the manifest

    @classmethod
    def load(cls, run_name_or_path: str) -> "V2Forecaster":
        from .pipeline_v2 import load_package

        path = Path(str(run_name_or_path))
        if path.is_dir():
            package_dir = path
        elif path.name == "manifest.json" and path.exists():
            package_dir = path.parent
        else:
            package_dir = MODEL_DIR / "v2" / str(run_name_or_path)
        return cls(load_package(package_dir))

    def predict(self, recent_history, model: str = "lstm") -> dict:
        """Forecast from a timestamped DataFrame of recent observations."""
        return self.package.forecast(recent_history, model=model)

    def render(self, result: dict, ends_at: str | None = None) -> str:
        lines = ["=" * 46, "   TRAFFIC FORECAST".center(46), "=" * 46, ""]
        lines.append("  Pipeline         : v2 (causal, hash-verified package)")
        lines.append("  Run              : {}  (site: {})".format(
            result["run_name"], result["site_scope"]))
        lines.append("  History analysed : last {} {} steps".format(
            self.cfg.sequence_length, result["cadence"]))
        lines.append("  Window ends      : {}".format(result["window_end"]))
        lines.append("  Model            : {}".format(result["model"]))
        lines.append("")
        for f in result["forecasts"]:
            lines.append("  +{}h  ->  {:>7,.0f} {}   (at {})".format(
                f["horizon"], f["value"], f["units"], f["timestamp"]))
        lines += ["", "=" * 46]
        return "\n".join(lines)


def load_forecaster(run_name: str = "baseline_univariate"):
    """Route to the v1 or v2 loader by the artifact's pipeline_version.

    Precedence: an explicit path to a v2 package (or its manifest.json), then
    ``models/<name>_artifacts.json`` (no pipeline_version key means v1, exactly
    like the archived runs), then ``models/v2/<name>/``.
    """
    path = Path(str(run_name))
    if path.is_dir() or path.name == "manifest.json":
        return V2Forecaster.load(run_name)

    artifact_path = MODEL_DIR / "{}_artifacts.json".format(run_name)
    if artifact_path.exists():
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        version = (payload.get("pipeline_version")
                   or payload.get("config", {}).get("pipeline_version")
                   or "v1")
        if version == "v1":
            return TrafficForecaster.load(run_name)
        if version == "v2":
            return V2Forecaster.load(run_name)
        raise ValueError(
            "Unknown pipeline_version {!r} in {}; known versions: v1, v2. "
            "Archived artifacts without the field are v1.".format(
                version, artifact_path.name))

    package_dir = MODEL_DIR / "v2" / str(run_name)
    if (package_dir / "manifest.json").exists():
        return V2Forecaster.load(run_name)
    raise FileNotFoundError(
        "No artifacts for run '{}'. Looked for {} and {}. Train first: "
        "python -m traffic_lstm.train".format(
            run_name, artifact_path.name, package_dir / "manifest.json"))


# ------------------------------------------------------------------- cli ----
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Forecast the next hour of traffic.")
    parser.add_argument("--run-name", default="baseline_univariate",
                        help="A v1 run name, a v2 package name under models/v2, "
                             "or a path to a v2 package directory.")
    parser.add_argument("--values", type=float, nargs="+",
                        help="The last N hourly traffic volumes (v1 models only).")
    parser.add_argument("--last-hours", action="store_true",
                        help="Use the final window of the dataset.")
    parser.add_argument("--model", choices=["lstm", "xgboost"], default="lstm",
                        help="v2 packages only: which packaged model to forecast with.")
    args = parser.parse_args(argv)

    forecaster = load_forecaster(args.run_name)
    ends_at = None

    if isinstance(forecaster, V2Forecaster):
        if args.values:
            parser.error("--values carries no timestamps; a v2 package forecasts "
                         "from cadence-checked, timestamped history only. "
                         "Use --last-hours.")
        if not args.last_hours:
            parser.error("Pass --last-hours for a v2 package.")
        import pandas as pd

        from .config import resolve_data_path
        from .pipeline_v2 import _load_and_clean

        cfg = forecaster.cfg
        raw = pd.read_csv(resolve_data_path(cfg.data_path))
        # Clean exactly as training did (duplicate collapse per the package's
        # own policy); the package then refuses the window if it still has a
        # gap, a duplicate or a missing value.
        series, _ = _load_and_clean(cfg, raw)
        window = series.tail(cfg.sequence_length)
        print(forecaster.render(forecaster.predict(window, model=args.model)))
        return

    if args.values:
        window = args.values
    elif args.last_hours:
        from .data import clean_series, load_raw

        raw = load_raw(forecaster.cfg.data_path, forecaster.cfg.datetime_column)
        series = clean_series(raw, forecaster.cfg.datetime_column,
                              forecaster.cfg.target_column)
        window = series[forecaster.cfg.target_column].to_numpy()[-forecaster.cfg.sequence_length:]
        ends_at = str(series[forecaster.cfg.datetime_column].iloc[-1])
    else:
        parser.error("Pass --values or --last-hours.")

    print(forecaster.render(forecaster.predict(window), ends_at))


if __name__ == "__main__":
    main()
