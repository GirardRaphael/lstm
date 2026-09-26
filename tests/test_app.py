from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_default_page_is_read_only_evidence():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.sidebar.radio[0].value == "0 · Forecast evidence"
    assert any(h.value == "Forecast evidence" for h in app.header)
    if app.selectbox:
        app.radio[0].set_value("test").run(timeout=30)
        assert not app.exception
        assert len(app.dataframe) == 1
