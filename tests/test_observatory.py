from concurrent.futures import ThreadPoolExecutor
import json
import time
import numpy as np
import pandas as pd
import pytest
from traffic_lstm.observations import validate_csv, demo_csv, MAX_UPLOAD_BYTES
from traffic_lstm.observatory import Store, AccessDenied, restore_backup
from traffic_lstm.operations import worker_once
from traffic_lstm.uncertainty import calibrate, interval_metrics


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "workspace"
    token = Store.initialize(root)
    return Store(root,token)


def test_input_contract_and_dst():
    frame = pd.DataFrame({"stream_id":["one"]*3,
        "timestamp":["2024-11-03T00:00:00-04:00","2024-11-03T01:00:00-04:00","2024-11-03T01:00:00-05:00"],
        "interval_seconds":[3600]*3,"vehicle_count":[0,2,3]})
    clean,q = validate_csv(frame.to_csv(index=False).encode())
    assert q["missing_intervals"] == 0
    for column,value in [("stream_id","other"),("timestamp","2024-11-03 03:00"),
                         ("timestamp","2099-01-01T00:00:00Z"),("vehicle_count",-1),
                         ("vehicle_count",np.inf),("interval_seconds",300)]:
        bad=frame.astype(object).copy()
        bad.loc[1,column]=value
        with pytest.raises(ValueError):
            validate_csv(bad.to_csv(index=False).encode())
    with pytest.raises(ValueError):
        validate_csv(b"a"*(MAX_UPLOAD_BYTES+1))
    with pytest.raises(ValueError):
        validate_csv(frame.iloc[::-1].to_csv(index=False).encode())
    with pytest.raises(ValueError):
        validate_csv(pd.concat([frame,frame.tail(1)]).to_csv(index=False).encode())


def test_access_revocation_immutability_and_quota(store):
    viewer = Store(store.root,store.add_user("viewer","viewer"))
    with pytest.raises(AccessDenied):
        viewer.import_csv(demo_csv(),"synthetic","synthetic")
    with pytest.raises(AccessDenied):
        Store(store.root,"wrong").list("datasets")
    original=demo_csv()
    data=store.import_csv(original,"synthetic","synthetic")
    assert store.import_csv(original,"synthetic","synthetic") == data
    meta,frame=store.dataset(data)
    frame.loc[0,"vehicle_count"] += 1
    with pytest.raises(ValueError,match="Conflicting"):
        store.import_csv(frame.to_csv(index=False).encode(),"correction")
    for _ in range(5):
        store.enqueue(data)
    with pytest.raises(ValueError,match="five"):
        store.enqueue(data)
    store.revoke("viewer")
    with pytest.raises(AccessDenied):
        viewer.list("datasets")


def test_claim_cancel_retry_recover_and_restart(store):
    data=store.import_csv(demo_csv(),"synthetic","synthetic")
    job=store.enqueue(data)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed=list(executor.map(lambda _:Store(store.root,store.token).claim(),range(2)))
    assert sum(row is not None for row in claimed)==1
    lease=next(row["lease"] for row in claimed if row)
    store.cancel(job)
    assert not store.heartbeat(job,lease)
    store.finish(job,lease,error="cancelled")
    assert store.list("jobs")[0]["status"] == "cancelled"
    new=store.retry(job)
    claimed=store.claim()
    with store.connect(True) as db:
        db.execute("UPDATE jobs SET updated=? WHERE id=?",(time.time()-120,new))
    reopened=Store(store.root,store.token)
    assert reopened.recover()==[new]
    assert not store.finish(new,claimed["lease"],error="late worker")
    assert reopened.list("jobs")[0]["status"] == "failed"


def test_calibration_is_finite_sample_and_evaluation_is_separate():
    calibration=calibrate(np.arange(100.),np.zeros(100))
    assert calibration["radius"] == 90
    metrics=interval_metrics(np.array([0.,200.]),np.zeros(2),calibration)
    assert metrics["coverage"] == .5 and metrics["mean_width"] == 90
    with pytest.raises(ValueError):
        calibrate([1],[1])


def test_worker_to_forecast_actuals_backup_restore_and_corruption(store,tmp_path):
    content=demo_csv(2400)
    all_frame,q=validate_csv(content)
    # Keep the next hour outside the imported snapshot until after forecasting.
    data=store.import_csv(all_frame.iloc[:-1].to_csv(index=False).encode(),"synthetic fixture","synthetic")
    job=store.enqueue(data)
    assert worker_once(store,timeout=60)
    assert store.list("jobs")[0]["status"] == "succeeded", store.list("jobs")
    model=store.list("models")[0]["id"]
    viewer=Store(store.root,store.add_user("reader","viewer"))
    with pytest.raises(AccessDenied):
        viewer.activate(model)
    store.activate(model)
    prediction=store.forecast(data,replay=True)
    assert prediction["status"]=="available", prediction
    assert prediction["lower"] <= prediction["value"] <= prediction["upper"]
    assert store.forecast(data,replay=True)["id"] == prediction["id"]
    assert store.errors()[0]["actual"] is None
    store.import_csv(content,"synthetic fixture updated","synthetic")
    assert store.errors()[0]["actual"] == all_frame.vehicle_count.iloc[-1]
    assert "error" in store.errors()[0]
    stale=store.forecast(data,now=pd.Timestamp.now(tz="UTC")+pd.Timedelta(days=5))
    assert stale["status"] == "unavailable" and "stale_observations" in stale["reasons"]
    backup=tmp_path/"backup.sqlite3"
    store.backup(backup)
    restored=restore_backup(backup,tmp_path/"restored",store.token)
    assert restored.errors()==store.errors()
    assert restored.forecast(data,replay=True)["value"] == pytest.approx(prediction["value"])
    with pytest.raises(ValueError):
        restore_backup(backup,tmp_path/"restored",store.token)
    with store.connect(True) as db:
        db.execute("UPDATE models SET payload=? WHERE id=?",(b"corrupt",model))
    unavailable=store.forecast(data,replay=True)
    assert unavailable["status"]=="unavailable" and "model_unavailable" in unavailable["reasons"]


def test_timeout_records_failure(store,monkeypatch):
    import subprocess
    import sys
    import traffic_lstm.operations as operations
    original=subprocess.Popen
    def hung_worker(args,**kwargs):
        return original([sys.executable,"-c","import time; time.sleep(60)"],**kwargs)
    monkeypatch.setattr(operations.subprocess,"Popen",hung_worker)
    data=store.import_csv(demo_csv(),"synthetic fixture","synthetic")
    store.enqueue(data)
    worker_once(store,timeout=1)
    assert store.list("jobs")[0]["status"] == "failed"


def test_expiration_rotation_and_stream_contract(store):
    old=store.token
    replacement=store.rotate_token()
    with pytest.raises(AccessDenied):
        store.identity()
    store.token=replacement
    content=demo_csv()
    identifier=store.import_csv(content,"synthetic","synthetic")
    _,frame=store.dataset(identifier)
    frame.timestamp += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError,match="grid"):
        store.import_csv(frame.to_csv(index=False).encode(),"shifted","synthetic",now=frame.timestamp.iloc[-1])
    with store.connect(True) as db:
        db.execute("UPDATE users SET expires=?",(time.time()-1,))
    with pytest.raises(AccessDenied):
        store.identity()


def test_job_owner_boundaries_and_incomplete_history(store):
    data=store.import_csv(demo_csv(),"synthetic","synthetic")
    other=Store(store.root,store.add_user("operator","operator"))
    job=store.enqueue(data)
    with pytest.raises(AccessDenied):
        other.cancel(job)
    with pytest.raises(AccessDenied):
        other.claim()
    _,frame=store.dataset(data)
    gapped=frame.drop(frame.index[-2])
    gap_id=store.import_csv(gapped.to_csv(index=False).encode(),"gapped","synthetic")
    forecast=store.forecast(gap_id,replay=True)
    assert forecast["status"]=="unavailable" and "incomplete_recent_history" in forecast["reasons"]


def test_release_gate_shift_and_monitoring(store):
    data=store.import_csv(demo_csv(),"synthetic","synthetic")
    store.enqueue(data)
    worker_once(store,timeout=60)
    model=store.list("models")[0]["id"]
    record,meta=store.model(model)
    import hashlib
    original=json.dumps(meta,sort_keys=True)
    bad={**meta,"evaluation":{**meta["evaluation"],"interval":{"coverage":.50,"n":100}}}
    bad_json=json.dumps(bad,sort_keys=True)
    with store.connect(True) as db:
        db.execute("UPDATE models SET metadata=?,metadata_hash=? WHERE id=?",(bad_json,hashlib.sha256(bad_json.encode()).hexdigest(),model))
    with pytest.raises(ValueError,match="coverage"):
        store.activate(model)
    with store.connect(True) as db:
        db.execute("UPDATE models SET metadata=?,metadata_hash=? WHERE id=?",(original,hashlib.sha256(original.encode()).hexdigest(),model))
    store.activate(model)
    _,frame=store.dataset(data)
    future=frame.tail(24).copy()
    future.timestamp += pd.Timedelta(hours=24)
    future.vehicle_count=1000000
    shifted=store.import_csv(future.to_csv(index=False).encode(),"shifted","synthetic",now=future.timestamp.iloc[-1])
    blocked=store.forecast(shifted,replay=True,now=future.timestamp.iloc[-1])
    assert "distribution_shift_review_required" in blocked["reasons"]
    # Fault injection: thirty distinct scored forecasts with severe undercoverage.
    with store.connect(True) as db:
        for i,stamp in enumerate(frame.timestamp.iloc[-30:]):
            details=json.dumps({"value":100000.,"lower":99999.,"upper":100001.,"mode":"historical_replay"})
            db.execute("INSERT INTO forecasts VALUES(?,?,?,?,?,?,?,?,?,?)",(f"fault{i}","fixture",data,meta["stream_id"],model,time.time(),(stamp-pd.Timedelta(hours=1)).isoformat(),stamp.isoformat(),"available",details))
    blocked=store.forecast(data,replay=True)
    assert "interval_coverage_below_threshold" in blocked["reasons"]
    assert "forecast_error_deterioration" in blocked["reasons"]
