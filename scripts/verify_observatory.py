"""Real bundled-data acceptance run in an isolated, ignored workspace; no token output."""
import json
from pathlib import Path
import sys
from uuid import uuid4
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from traffic_lstm.observatory import Store,restore_backup
from traffic_lstm.operations import worker_once


def main():
    root=ROOT/"var"/f"acceptance-{uuid4().hex}"
    token=Store.initialize(root)
    store=Store(root,token)
    raw=pd.read_csv(ROOT/"data/raw/Metro_Interstate_Traffic_Volume.csv")
    if (raw.groupby("date_time").traffic_volume.nunique()>1).any():
        raise ValueError("Conflicting duplicate targets")
    raw=raw.drop_duplicates("date_time").sort_values("date_time")
    frame=pd.DataFrame({"stream_id":"historical-replay/i94-clock-as-utc",
        "timestamp":pd.to_datetime(raw.date_time).dt.tz_localize("UTC"),"interval_seconds":3600,
        "vehicle_count":raw.traffic_volume})
    dataset=store.import_csv(frame.iloc[:-24].to_csv(index=False).encode(),
        "UCI bundled I-94; clock labels assigned UTC for historical replay only", "historical")
    job=store.enqueue(dataset)
    worker_once(store,timeout=120)
    assert store.list("jobs")[0]["status"]=="succeeded",store.list("jobs")
    model=store.list("models")[0]["id"]
    record,meta=store.model(model)
    store.activate(model)
    forecast=store.forecast(dataset,replay=True)
    assert forecast["status"]=="available",forecast
    store.import_csv(frame.to_csv(index=False).encode(),"UCI bundled I-94; actuals added after forecast","historical")
    errors=store.errors()
    assert errors[0]["actual"] is not None
    backup=root/"verified-backup.sqlite3"
    store.backup(backup)
    restored=restore_backup(backup,root/"restored",token)
    assert restored.errors()==errors
    restored_value=restored.forecast(dataset,replay=True)["value"]
    assert restored_value==forecast["value"]
    store.deactivate(frame.stream_id.iloc[0])
    assert store.forecast(dataset,replay=True)["status"]=="unavailable"
    store.activate(model)
    summary={"status":"passed","workspace":str(root.relative_to(ROOT)),
             "scope":"bundled historical replay; clock-as-UTC is artificial, not source certification",
             "dataset":dataset,"job":job,"model":model,"selected":meta["selected"],
             "evaluation":meta["evaluation"],"forecast":forecast,"actual_review":errors[0],
             "restart_backup_restore_prediction_parity":True,"deactivation_and_reactivation":True,
             "credentials":"isolated test token not exported; initialize your own workspace for use"}
    (ROOT/"reports/observatory_acceptance.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"status":"passed","selected":meta["selected"],"evaluation":meta["evaluation"]},indent=2))


if __name__=="__main__":
    main()
