"""Checks for the M1-phase-2 wiring of pipeline v2 into the app contracts.

Runs without pytest:

    python tests/test_phase2_wiring.py

Covers: TrainingConfig.pipeline_version defaults to "v1" so archived artifact
JSONs keep their meaning; explicit "v2" round-trips; unknown versions are
rejected; the train CLI routes --pipeline both ways (including one real tiny
v2 train with 1 epoch); predict routes v1 and v2 artifacts to the right
loader. Heavy work is a single tiny v2 package trained once and cached.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from traffic_lstm import pipeline_v2 as v2
from traffic_lstm.config import TrainingConfig

PASSED = []
FAILED = []


def check(name: str):
    def wrapper(fn):
        try:
            fn()
            PASSED.append(name)
            print("  PASS  {}".format(name))
        except AssertionError as exc:
            frame = next((f for f in reversed(traceback.extract_tb(*sys.exc_info()[2:]))
                          if f.filename.endswith("test_phase2_wiring.py")), None)
            where = "line {}".format(frame.lineno) if frame else "?"
            FAILED.append((name, "{} ({})".format(exc, where)))
            print("  FAIL  {} -> {} ({})".format(name, exc, where))
        return fn
    return wrapper


class expect_raises:
    """Assert that the body raises (a subclass of) the given exception."""

    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is not None, "expected {} but nothing was raised".format(
            self.exc_type.__name__)
        assert issubclass(exc_type, self.exc_type), \
            "expected {} but got {}: {}".format(
                self.exc_type.__name__, exc_type.__name__, exc)
        return True


TMP = Path(tempfile.mkdtemp(prefix="traffic_phase2_tests_"))


# ---------------------------------------------------------------- fixtures --
def make_frame(hours=400, start="2021-01-01", seed=0) -> pd.DataFrame:
    """A clean daily cycle plus noise - hourly, gapless, no duplicates."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="h")
    cycle = 3000 + 1200 * np.sin(2 * np.pi * index.hour / 24)
    return pd.DataFrame({
        "date_time": index,
        "traffic_volume": cycle + rng.normal(0, 30, hours),
    })


CSV_PATH = TMP / "tiny.csv"
make_frame().to_csv(CSV_PATH, index=False)

CLI_V2_DIR = TMP / "cli_v2"
CLI_V2_ARGV = [
    "--pipeline", "v2", "--data", str(CSV_PATH),
    "--sequence-length", "8", "--horizons", "1",
    "--units", "4", "--dropout", "0.0",
    "--epochs", "1", "--batch-size", "64", "--patience", "2",
    "--run-name", "cli_tiny", "--site-scope", "test-corridor",
    "--output-dir", str(CLI_V2_DIR), "--no-figures",
]

_TRAINED = {}


def cli_trained() -> Path:
    """Train (once) a real tiny v2 package through the CLI path, 1 epoch."""
    if "package_dir" not in _TRAINED:
        from traffic_lstm import train as train_cli
        train_cli.main(CLI_V2_ARGV)
        _TRAINED["package_dir"] = CLI_V2_DIR / "cli_tiny"
    return _TRAINED["package_dir"]


# ------------------------------------------------------------------ config --
@check("config: archived artifacts without pipeline_version load as v1")
def _():
    artifacts = sorted((ROOT / "models").glob("*_artifacts.json"))
    assert len(artifacts) >= 9, "expected the archived run artifacts"
    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "pipeline_version" not in payload["config"], \
            "{} unexpectedly carries the field".format(path.name)
        cfg = TrainingConfig(**payload["config"])
        assert cfg.pipeline_version == "v1", path.name
        info = v2.read_legacy_artifact(path)
        assert info["pipeline_version"] == "v1", path.name


@check("config: explicit v2 round-trips through JSON")
def _():
    cfg = TrainingConfig(pipeline_version="v2", run_name="roundtrip")
    assert cfg.to_dict()["pipeline_version"] == "v2"
    restored = TrainingConfig(**json.loads(json.dumps(cfg.to_dict())))
    assert restored.pipeline_version == "v2"

    cfg2 = v2.V2Config(site_scope="test-corridor",
                       units={"traffic_volume": "vehicles/hour"},
                       run_name="roundtrip2", output_dir=TMP)
    restored2 = v2.V2Config(**json.loads(json.dumps(cfg2.to_dict())))
    assert restored2.pipeline_version == "v2"
    assert restored2.units == {"traffic_volume": "vehicles/hour"}


@check("config: unknown pipeline versions are rejected with a clear error")
def _():
    with expect_raises(ValueError):
        TrainingConfig(pipeline_version="v3")
    try:
        TrainingConfig(pipeline_version="v3")
    except ValueError as exc:
        assert "v1" in str(exc) and "v2" in str(exc), exc
    with expect_raises(ValueError):
        v2.V2Config(pipeline_version="v1", site_scope="x",
                    units={"traffic_volume": "vehicles/hour"})

    import traffic_lstm.config as config
    import traffic_lstm.predict as predict

    bogus_dir = TMP / "bogus_models"
    bogus_dir.mkdir(exist_ok=True)
    (bogus_dir / "mystery_artifacts.json").write_text(
        json.dumps({"pipeline_version": "v9", "config": {}}), encoding="utf-8")
    with patch.object(config, "MODEL_DIR", bogus_dir), \
            patch.object(predict, "MODEL_DIR", bogus_dir):
        with expect_raises(ValueError):
            predict.load_forecaster("mystery")

    bad_pkg = TMP / "bad_pkg"
    bad_pkg.mkdir(exist_ok=True)
    (bad_pkg / "manifest.json").write_text(
        json.dumps({"pipeline_version": "v3"}), encoding="utf-8")
    with expect_raises(ValueError):
        predict.load_forecaster(str(bad_pkg))


# --------------------------------------------------------------------- cli --
@check("cli: --pipeline defaults to v2 and v1 builds a v1 TrainingConfig")
def _():
    import traffic_lstm.train as train_cli

    args = train_cli.build_parser().parse_args([])
    assert args.pipeline == "v2", "new runs must default to the causal pipeline"

    captured = {}

    def fake_train(cfg, *a, **k):
        captured["cfg"] = cfg
        return {}

    with patch.object(train_cli, "train", fake_train):
        train_cli.main(["--pipeline", "v1", "--no-figures", "--run-name", "smoke_v1"])
    cfg = captured["cfg"]
    assert isinstance(cfg, TrainingConfig)
    assert cfg.pipeline_version == "v1"
    assert cfg.run_name == "smoke_v1"

    with patch.object(train_cli, "train", fake_train):
        train_cli.main(["--pipeline", "v1", "--no-figures"])
    assert captured["cfg"].run_name == "baseline_univariate", \
        "the v1 default run name must be preserved"


@check("cli: --pipeline v2 trains a real tiny package end-to-end (1 epoch)")
def _():
    package_dir = cli_trained()
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline_version"] == "v2"
    assert manifest["run_name"] == "cli_tiny"
    assert manifest["site_scope"] == "test-corridor"
    assert manifest["units"] == {"traffic_volume": "vehicles/hour"}
    assert manifest["training"]["epochs_run"] == 1
    assert (package_dir / "model.keras").exists()
    assert (package_dir / "xgb_h1.json").exists()

    # A second run under the same name is refused, not overwritten.
    from traffic_lstm import train as train_cli
    with expect_raises(v2.ArtifactExistsError):
        train_cli.main(CLI_V2_ARGV)


@check("cli: v2 requires declared units; --unit COLUMN=UNIT supplies them")
def _():
    import traffic_lstm.train as train_cli

    units = train_cli.resolve_units(["mystery=celsius"], ["traffic_volume", "mystery"])
    assert units == {"traffic_volume": "vehicles/hour", "mystery": "celsius"}, units
    with expect_raises(ValueError):
        train_cli.resolve_units(["broken"], ["traffic_volume"])

    frame = make_frame(200, seed=7)
    frame["mystery"] = 1.0
    csv2 = TMP / "tiny_exo.csv"
    frame.to_csv(csv2, index=False)
    with expect_raises(v2.UnitsDeclarationError):
        train_cli.main(["--pipeline", "v2", "--data", str(csv2),
                        "--exogenous", "mystery", "--run-name", "no_units",
                        "--site-scope", "test-corridor",
                        "--output-dir", str(CLI_V2_DIR), "--no-figures"])
    assert not (CLI_V2_DIR / "no_units").exists(), "nothing may be written on refusal"


# ----------------------------------------------------------------- predict --
@check("predict: v1 artifacts route to the legacy forecaster, unchanged")
def _():
    import traffic_lstm.predict as predict

    forecaster = predict.load_forecaster("baseline_univariate")
    assert isinstance(forecaster, predict.TrafficForecaster)
    assert forecaster.cfg.pipeline_version == "v1"
    result = forecaster.predict([2000 + 10 * k for k in range(24)])
    assert set(result["forecasts"][0]) == {"horizon", "volume", "level", "comment"}


@check("predict: v2 packages route to the package forecaster")
def _():
    import traffic_lstm.predict as predict

    package_dir = cli_trained()
    forecaster = predict.load_forecaster(str(package_dir))
    assert isinstance(forecaster, predict.V2Forecaster)

    frame = pd.read_csv(CSV_PATH)
    window = frame.tail(forecaster.cfg.sequence_length)
    result = forecaster.predict(window)
    assert result["pipeline_version"] == "v2"
    assert result["run_name"] == "cli_tiny"
    first = result["forecasts"][0]
    assert first["units"] == "vehicles/hour"
    assert np.isfinite(first["value"])
    assert first["timestamp"] > result["window_end"]

    # A gap in recent history is refused, never imputed.
    gapped = frame.tail(forecaster.cfg.sequence_length + 1).reset_index(drop=True)
    gapped = gapped.drop(index=1)
    with expect_raises(v2.GapHistoryError):
        forecaster.predict(gapped)


@check("predict: CLI refuses bare --values for v2, serves --last-hours")
def _():
    import traffic_lstm.predict as predict

    package_dir = cli_trained()
    with expect_raises(SystemExit):
        predict.main(["--run-name", str(package_dir),
                      "--values"] + ["3000"] * 8)

    out = io.StringIO()
    with redirect_stdout(out):
        predict.main(["--run-name", str(package_dir), "--last-hours"])
    text = out.getvalue()
    assert "TRAFFIC FORECAST" in text and "vehicles/hour" in text, text


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n{} passed, {} failed\n".format(len(PASSED), len(FAILED)))
    for name, reason in FAILED:
        print("  {}: {}".format(name, reason))
    sys.exit(1 if FAILED else 0)
