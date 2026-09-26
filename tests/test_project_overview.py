import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from traffic_lstm.project_overview import build_slides, briefing_markdown

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/observatory_app.py"


def test_briefing_results_come_from_saved_evidence_and_distinguish_protocols():
    slides = build_slides()
    for dataset, title in (("motorway", "Motorway: the tree earns its place"),
                           ("bikes", "Bikes: a small MAE gap needs restraint")):
        manifest = json.loads((ROOT / f"models/v2/{dataset}_v2_20260925/manifest.json").read_text())
        slide = next(s for s in slides if s.title == title)
        rows = {row["Candidate"]: row for row in slide.rows}
        assert rows["XGBoost"]["MAE"] == round(manifest["evaluation"]["test"]["h1"]["xgboost"]["mae"], 2)
        assert rows["LSTM"]["MAPE (%)"] == round(manifest["evaluation"]["test"]["h1"]["lstm"]["mape"], 2)
    bike = next(s for s in slides if s.title.startswith("Bikes:"))
    assert "crosses zero" in bike.takeaway
    exported = briefing_markdown(slides)
    assert "64% fit / 16% validation / 10% calibration / 10% evaluation" in exported
    assert "not an architecture-only ablation" in exported
    assert "Source" in exported and "undercoverage" in exported


def test_missing_evidence_does_not_fabricate_metrics(tmp_path):
    slides = build_slides(tmp_path)
    motorway = next(s for s in slides if s.title.startswith("Motorway:"))
    assert not motorway.rows and not motorway.facts
    assert "unavailable" in motorway.takeaway


def test_public_landing_slideshow_navigation_and_last_slide(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_ROOT", str(tmp_path / "not-initialized"))
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any(h.value == "The project in one minute" for h in app.header)
    assert next(b for b in app.button if b.label == "Previous").disabled
    next(b for b in app.button if b.label == "Next").click().run()
    assert any(h.value == "What the operator actually does" for h in app.header)
    for i, slide in enumerate(build_slides()):
        app.selectbox[0].set_value(i).run()
        assert not app.exception, slide.title
        assert any(h.value == slide.title for h in app.header)
    assert next(b for b in app.button if b.label == "Next").disabled
    next(b for b in app.button if b.label == "Previous").click().run()
    assert not next(b for b in app.button if b.label == "Next").disabled
    # Briefing is public, but no operational workspace action is exposed.
    assert not any(b.label == "Queue training" for b in app.button)
