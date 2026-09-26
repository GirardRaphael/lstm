from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from traffic_lstm.observatory import Store
from traffic_lstm.observations import demo_csv,validate_csv
from traffic_lstm.operations import worker_once

APP=Path(__file__).resolve().parents[1]/"app/observatory_app.py"


def test_login_expiry_and_viewer_permissions(tmp_path,monkeypatch):
    root=tmp_path/"workspace"
    admin=Store.initialize(root)
    store=Store(root,admin)
    viewer=store.add_user("viewer","viewer")
    monkeypatch.setenv("OBSERVATORY_ROOT",str(root))
    app=AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.text_input[0].label=="Access token"
    app.text_input[0].set_value("bad")
    next(b for b in app.button if b.label=="Sign in").click().run()
    assert app.error
    app.text_input[0].set_value(viewer)
    next(b for b in app.button if b.label=="Sign in").click().run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Training jobs").run()
    assert next(b for b in app.button if b.label=="Queue training").disabled
    store.revoke("viewer")
    app.run()
    assert app.text_input[0].label=="Access token"


def test_operator_job_activation_forecast_actual_review(tmp_path,monkeypatch):
    root=tmp_path/"workspace"
    token=Store.initialize(root)
    store=Store(root,token)
    frame,_=validate_csv(demo_csv())
    dataset=store.import_csv(frame.iloc[:-1].to_csv(index=False).encode(),"fixture","synthetic")
    monkeypatch.setenv("OBSERVATORY_ROOT",str(root))
    app=AppTest.from_file(str(APP))
    app.session_state.access_token=token
    app.run()
    app.sidebar.radio[0].set_value("Training jobs").run()
    next(b for b in app.button if b.label=="Queue training").click().run()
    assert len(store.list("jobs"))==1
    worker_once(store,timeout=60)
    app.sidebar.radio[0].set_value("Models").run()
    next(b for b in app.button if b.label=="Activate this version").click().run()
    assert store.list("active_models")
    app.sidebar.radio[0].set_value("Forecasts and errors").run()
    next(b for b in app.button if b.label=="Record next-hour forecast").click().run()
    assert store.list("forecasts")[0]["status"]=="available",app.error
    store.import_csv(frame.to_csv(index=False).encode(),"fixture actual","synthetic")
    app.run()
    assert any(m.label=="Observed MAE" for m in app.metric)
    assert not app.exception
