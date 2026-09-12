"""Load a trained model and forecast the next hour.

    python -m traffic_lstm.predict --last-hours
    python -m traffic_lstm.predict --values 2150 2430 3020 ... (24 numbers)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import TrainingConfig
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
                "feeds past traffic, so it cannot drive that model — use the "
                "Streamlit app, which has the full feature matrix.".format(
                    self.cfg.run_name, expected_features,
                    ", ".join(self.cfg.exogenous_columns) or "see config"))

        values = np.asarray(last_hours, dtype="float64").ravel()
        if len(values) != self.cfg.sequence_length:
            raise ValueError(
                "Expected exactly {} values, got {}.".format(
                    self.cfg.sequence_length, len(values)))

        scaled = self.scaler.transform(values.reshape(-1, 1))
        window = scaled.reshape(1, self.cfg.sequence_length, 1)
        predicted_scaled = self.model.predict(window, verbose=0)[0]
        predicted = self.scaler.inverse_transform(
            np.asarray(predicted_scaled).reshape(-1, 1)).ravel()

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


# ------------------------------------------------------------------- cli ----
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Forecast the next hour of traffic.")
    parser.add_argument("--run-name", default="baseline_univariate")
    parser.add_argument("--values", type=float, nargs="+",
                        help="The last N hourly traffic volumes.")
    parser.add_argument("--last-hours", action="store_true",
                        help="Use the final window of the dataset.")
    args = parser.parse_args(argv)

    forecaster = TrafficForecaster.load(args.run_name)
    ends_at = None

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
