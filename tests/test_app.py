"""Render the first Streamlit page without starting a browser."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    app.run()
    assert not app.exception, [exception.value for exception in app.exception]
    assert app.title[0].value == "🚦 Traffic LSTM"
    assert len(app.metric) == 4
    print("  PASS  Streamlit app renders the bundled dataset")
    app.sidebar.radio[0].set_value("7 · Street data checks").run()
    assert not app.exception, [exception.value for exception in app.exception]
    assert any("analysis only" in item.value for item in app.success)
    app.number_input[0].set_value(3600).run()
    assert not app.exception
    assert any("blocked" in item.value for item in app.error)
    print("  PASS  Street data page accepts valid demo and blocks wrong cadence")

    # Forecast must use the trained bundle even if another CSV was selected.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd
    from traffic_lstm.data import build_datasets
    from traffic_lstm.predict import TrafficForecaster
    forecaster = TrafficForecaster.load()
    bundle = build_datasets(forecaster.cfg)
    app.session_state["cfg"] = forecaster.cfg
    app.session_state["outcome"] = {"bundle": bundle, "model": forecaster.model}
    app.session_state["series"] = pd.DataFrame({"unrelated_column": [1]})
    app.session_state["target_column"] = "unrelated_column"
    app.sidebar.radio[0].set_value("5 · Forecast").run()
    assert not app.exception, [exception.value for exception in app.exception]
    assert len(app.metric) == 1
    app.radio[0].set_value("Type the values myself").run()
    app.text_area[0].set_value(",".join(["nan"] * forecaster.cfg.sequence_length)).run()
    assert not app.exception
    assert any("finite" in item.value for item in app.error)
    print("  PASS  Forecast binds to trained data and rejects NaN input")


if __name__ == "__main__":
    main()
