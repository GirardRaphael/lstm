"""Smoke-test every model artifact committed to the repository."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorflow.keras.models import load_model

from traffic_lstm.config import TrainingConfig
from traffic_lstm.data import build_datasets


def main() -> None:
    artifacts = sorted((ROOT / "models").glob("*_artifacts.json"))
    if not artifacts:
        raise AssertionError("No saved model artifacts found")

    skipped = 0
    for artifact_path in artifacts:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        cfg = TrainingConfig(**payload["config"])
        if not cfg.model_path.exists():
            # Metrics-only records (e.g. the window sweep) archive results
            # without committed weights; there is no model to smoke-test.
            print("  SKIP  {}: no committed weights at {}".format(cfg.run_name, cfg.model_path.name))
            skipped += 1
            continue
        model = load_model(cfg.model_path)
        bundle = build_datasets(cfg)
        predicted = model.predict(bundle.X_test[:2], verbose=0)
        expected = (2, cfg.n_outputs)
        assert predicted.shape == expected, (cfg.run_name, predicted.shape, expected)
        print("  PASS  saved model: {} {}".format(cfg.run_name, predicted.shape))

    print("\n{} saved models passed, {} metrics-only records skipped\n".format(len(artifacts) - skipped, skipped))


if __name__ == "__main__":
    main()
