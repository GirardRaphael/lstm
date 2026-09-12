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


if __name__ == "__main__":
    main()
