"""Authenticated local observation → job → forecast → actual/error workflow."""
import json
import os
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from traffic_lstm.observatory import Store, AccessDenied
from traffic_lstm.operations import DEFAULT_ROOT
from traffic_lstm.project_overview import render_project_slideshow

st.set_page_config(page_title="Traffic Observatory",page_icon="📊",layout="wide")
st.title("Traffic Observatory")
st.caption("Read-only traffic analytics · local research workspace · no signal control")
workspace = Path(os.environ.get("OBSERVATORY_ROOT",str(DEFAULT_ROOT)))
if not (workspace / "observatory.sqlite3").exists():
    render_project_slideshow()
    st.divider()
    st.info("Initialize your local workspace once, then sign in with its access token.")
    st.code("./observatory.ps1 init", language="powershell")
    st.stop()

if "access_token" not in st.session_state:
    render_project_slideshow()
    st.divider()
    with st.form("login"):
        token = st.text_input("Access token",type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            Store(workspace,token).identity()
            st.session_state.access_token=token
            st.rerun()
        except AccessDenied:
            st.error("Token is invalid, expired or revoked.")
    st.stop()

store = Store(workspace,st.session_state.access_token)
try:
    identity = store.identity()
except AccessDenied:
    del st.session_state.access_token
    st.rerun()

with st.sidebar:
    st.write(f"{identity['name']} · {identity['role']}")
    page=st.radio("Workspace",["Overview","Import observations","Training jobs","Models","Forecasts and errors","Audit and recovery"])
    if st.button("Sign out"):
        del st.session_state.access_token
        st.rerun()
    st.caption("Tokens expire after 30 days. This workspace is not a multi-tenant city service.")

def selector(label,rows,field="id"):
    return st.selectbox(label,[r[field] for r in rows]) if rows else None

try:
    if page == "Overview":
        render_project_slideshow()
        st.divider()
        st.header("Workspace status")
        datasets,jobs,models = [store.list(k) for k in ("datasets","jobs","models")]
        columns=st.columns(3)
        for column,label,items in zip(columns,["Imported snapshots","Training jobs","Model versions"],[datasets,jobs,models]):
            column.metric(label,len(items))
        st.info("Start with an import, queue training, run the worker, activate a model, then record a forecast. "
                "Import later observations to see forecast errors. Historical replay is labeled explicitly.")
        if datasets:
            st.dataframe(pd.DataFrame([{k:d[k] for k in ("id","stream","mode","source")} for d in datasets]),hide_index=True)
        failures=[j for j in jobs if j["status"]=="failed"]
        if failures:
            st.warning(f"{len(failures)} failed job(s). Review Training jobs before retrying.")
        st.markdown("Current research evidence: `reports/REPORT.md`. Generalization and interval diagnostics: `reports/BACKTEST_REPORT.md`.")
    elif page == "Import observations":
        st.header("Import completed observations")
        st.caption("CSV ≤ 10 MiB. Required: stream_id, timestamp with UTC offset, interval_seconds, vehicle_count. "
                   "Counts must be nonnegative integers; timestamps strictly increasing. One stream and cadence per import.")
        with st.form("import"):
            upload=st.file_uploader("Observation CSV",type=["csv"])
            source=st.text_input("Source and authorization/provenance reference")
            mode=st.selectbox("Observation mode",["historical","synthetic","live"])
            submit=st.form_submit_button("Validate and import",disabled=identity["role"]=="viewer")
        if submit:
            if upload is None:
                st.error("Choose a CSV first.")
            else:
                identifier=store.import_csv(upload.getvalue(),source,mode)
                meta,_=store.dataset(identifier)
                st.success(f"Imported snapshot {identifier}")
                st.json(json.loads(meta["quality"]))
        st.caption("Bundled historical replay can be imported with: ./observatory.ps1 bundled --dataset motorway")
    elif page == "Training jobs":
        st.header("Persistent training jobs")
        data=selector("Snapshot",store.list("datasets"))
        if st.button("Queue training",disabled=data is None or identity["role"]=="viewer"):
            st.success("Queued job " + store.enqueue(data))
        st.caption("Training runs outside this app. Start the bounded worker in a terminal:")
        st.code("./observatory.ps1 worker",language="powershell")
        jobs=store.list("jobs")
        if jobs:
            st.dataframe(pd.DataFrame(jobs).drop(columns=["lease"],errors="ignore"),hide_index=True)
            job=selector("Job to manage",jobs)
            cancel,retry,refresh=st.columns(3)
            if cancel.button("Cancel job",disabled=identity["role"]=="viewer"):
                store.cancel(job)
                st.rerun()
            if retry.button("Retry failed job",disabled=identity["role"]=="viewer"):
                st.success("New job " + store.retry(job))
            if refresh.button("Refresh status"):
                st.rerun()
    elif page == "Models":
        st.header("Model versions and activation")
        models=store.list("models")
        model=selector("Model version",models)
        if model:
            row,meta=store.model(model)
            st.write("Stream:",row["stream"]," · Candidate:",meta["selected"])
            st.json({"validation":meta["validation"],"evaluation":meta["evaluation"],"calibration":meta["calibration"]})
            st.caption("Activation records the local serving choice. It is not approval for field use. "
                       "Select an earlier version and activate it to roll back.")
            if st.button("Activate this version",disabled=identity["role"]!="admin"):
                store.activate(model)
                st.success("Active version updated and audited.")
            if st.button("Disable forecasts for this stream",disabled=identity["role"]!="admin"):
                store.deactivate(row["stream"])
                st.success("Forecasts disabled. No active model will be served.")
        else:
            st.info("No completed model jobs yet.")
        st.dataframe(pd.DataFrame(store.list("active_models")),hide_index=True)
    elif page == "Forecasts and errors":
        st.header("Record a forecast and review actuals")
        data=selector("Snapshot",store.list("datasets"))
        replay=st.checkbox("Historical replay (never a live forecast)",value=True)
        if st.button("Record next-hour forecast",disabled=data is None or identity["role"]=="viewer"):
            result=store.forecast(data,replay=replay)
            if result["status"]=="unavailable":
                st.warning("Forecast unavailable: " + ", ".join(result["reasons"]))
            else:
                st.success(f"{result['value']:.1f} vehicles; nominal 90% interval [{result['lower']:.1f}, {result['upper']:.1f}]")
            st.json(result)
        errors=store.errors()
        if errors:
            st.dataframe(pd.DataFrame(errors),hide_index=True)
            scored=[row for row in errors if row.get("actual") is not None and row["status"]=="available"]
            if scored:
                st.metric("Observed MAE",f"{sum(abs(row['error']) for row in scored)/len(scored):.2f}")
                st.metric("Observed interval coverage",f"{sum(row['covered'] for row in scored)/len(scored):.1%}")
            st.download_button("Download forecast/error records",json.dumps(errors,indent=2),file_name="forecast_errors.json")
        st.caption("Intervals are calibrated on a separate historical block. Temporal shift can invalidate nominal coverage. "
                   "Stale, gapped, mismatched or severely shifted inputs produce unavailable status.")
    else:
        st.header("Audit and recovery")
        if identity["role"]!="admin":
            st.warning("Administrator access required.")
        else:
            if st.button("Recover abandoned jobs (heartbeat older than 60 seconds)"):
                st.write(store.recover())
            st.code("./observatory.ps1 backup backup.sqlite3\n./observatory.ps1 --root var/restored restore backup.sqlite3",language="powershell")
            st.dataframe(pd.DataFrame(store.list("audit")),hide_index=True)
            st.caption("Backups include data, models and token hashes. Restrict filesystem access. "
                       "Restore to a new directory, verify it, and explicitly switch the workspace.")
except (ValueError,PermissionError,OSError) as exc:
    st.error(str(exc))
