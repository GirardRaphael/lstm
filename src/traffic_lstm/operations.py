"""Local observatory CLI. Tokens come from OBSERVATORY_TOKEN, never command arguments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .config import PROJECT_ROOT
from .observatory import Store, restore_backup
from .observations import demo_csv

DEFAULT_ROOT = PROJECT_ROOT / "var" / "observatory"


def execute(store, job, lease):
    # Only a current administrator with a valid job lease can complete work.
    if not store.heartbeat(job,lease):
        return
    record = next((r for r in store.list("jobs") if r["id"] == job), None)
    if record is None:
        raise ValueError("Unknown job")
    try:
        meta, frame = store.dataset(record["dataset"])
        from .candidate import train_candidate
        model = train_candidate(frame, meta["stream"])
        store.finish(job,lease,model=model)
    except Exception as exc:
        store.finish(job,lease,error=f"{type(exc).__name__}: {exc}")
        raise


def worker_once(store, *, timeout=600):
    if not 1 <= timeout <= 3600:
        raise ValueError("Worker timeout must be 1–3600 seconds")
    job = store.claim()
    if job is None:
        return False
    logs = store.root / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["OBSERVATORY_TOKEN"] = store.token
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["OMP_NUM_THREADS"] = "1"
    try:
        with (logs / f"{job['id']}.log").open("wb") as log:
            proc = subprocess.Popen([sys.executable,"-m","traffic_lstm.operations","--root",str(store.root),
                                      "execute","--job",job["id"],"--lease",job["lease"]],
                                     stdout=log,stderr=subprocess.STDOUT,env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            start = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic()-start > timeout or not store.heartbeat(job["id"],job["lease"]):
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        store.finish(job["id"],job["lease"],error="Worker timed out or cancellation requested")
                        return True
                    time.sleep(.5)
            except BaseException:
                proc.terminate()
                proc.wait(timeout=5)
                store.finish(job["id"],job["lease"],error="Worker interrupted")
                raise
            if proc.returncode:
                store.finish(job["id"],job["lease"],error=f"Worker exited {proc.returncode}; inspect job log")
    except Exception as exc:
        store.finish(job["id"],job["lease"],error=f"Worker launch failed: {exc}")
        raise
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("rotate-token")
    user = sub.add_parser("add-user")
    user.add_argument("name")
    user.add_argument("--role", choices=["admin","operator","viewer"], default="viewer")
    revoke = sub.add_parser("revoke")
    revoke.add_argument("name")
    demo = sub.add_parser("demo")
    demo.add_argument("--hours", type=int, default=2400)
    bundled = sub.add_parser("bundled")
    bundled.add_argument("--dataset", choices=["motorway"], default="motorway")
    bundled.add_argument("--hold-back", type=int, default=24)
    imp = sub.add_parser("import")
    imp.add_argument("csv", type=Path)
    imp.add_argument("--source", required=True)
    imp.add_argument("--mode", choices=["historical","live","synthetic"], default="historical")
    for command in ("enqueue","cancel","retry","activate"):
        cmd = sub.add_parser(command)
        cmd.add_argument("id")
    deactivate = sub.add_parser("deactivate")
    deactivate.add_argument("stream")
    worker = sub.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--timeout", type=int, default=600)
    exe = sub.add_parser("execute", help=argparse.SUPPRESS)
    exe.add_argument("--job", required=True)
    exe.add_argument("--lease", required=True)
    sub.add_parser("recover")
    fc = sub.add_parser("forecast")
    fc.add_argument("dataset")
    fc.add_argument("--replay", action="store_true")
    sub.add_parser("errors")
    ls = sub.add_parser("list")
    ls.add_argument("collection", choices=["datasets","jobs","models","active_models","forecasts","audit","users"])
    backup = sub.add_parser("backup")
    backup.add_argument("destination", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("source", type=Path)
    args = p.parse_args(argv)
    token = os.environ.get("OBSERVATORY_TOKEN", "")
    if args.command == "init":
        print("Administrator token (save securely; expires in 30 days):")
        print(Store.initialize(args.root))
        return
    if args.command == "restore":
        restore_backup(args.source,args.root,token)
        print("Backup restored into a new workspace. Review active models and recover stale jobs.")
        return
    store = Store(args.root,token)
    if args.command == "add-user":
        print(store.add_user(args.name,args.role))
    elif args.command == "rotate-token":
        print(store.rotate_token())
    elif args.command == "deactivate":
        store.deactivate(args.stream)
    elif args.command == "revoke":
        store.revoke(args.name)
    elif args.command == "demo":
        if not 1000 <= args.hours <= 100000:
            p.error("demo hours must be 1000–100000")
        print(store.import_csv(demo_csv(args.hours), "generated synthetic count demo", "synthetic"))
    elif args.command == "import":
        if args.csv.stat().st_size > 10*1024*1024:
            p.error("CSV exceeds 10 MiB")
        print(store.import_csv(args.csv.read_bytes(),args.source,args.mode))
    elif args.command == "bundled":
        import pandas as pd
        frame = pd.read_csv(PROJECT_ROOT / "data/raw/Metro_Interstate_Traffic_Volume.csv")
        if (frame.groupby("date_time").traffic_volume.nunique()>1).any():
            raise ValueError("Conflicting duplicate count records in bundled data")
        frame=frame.drop_duplicates("date_time").sort_values("date_time")
        if not 0 <= args.hold_back < len(frame)-1000:
            p.error("hold-back must leave at least 1000 observations")
        if args.hold_back:
            frame=frame.iloc[:-args.hold_back]
        canonical=pd.DataFrame({"stream_id":"historical-replay/i94-clock-as-utc",
            "timestamp":pd.to_datetime(frame.date_time).dt.tz_localize("UTC"),
            "interval_seconds":3600,"vehicle_count":frame.traffic_volume})
        print(store.import_csv(canonical.to_csv(index=False).encode(),
            "Bundled UCI I-94. Clock labels assigned UTC for replay only; source timezone and interval-end semantics not certified.", "historical"))
    elif args.command in ("enqueue","cancel","retry","activate"):
        print(getattr(store,args.command)(args.id))
    elif args.command == "worker":
        while True:
            worker_once(store,timeout=args.timeout)
            if args.once:
                break
            time.sleep(2)
    elif args.command == "execute":
        execute(store,args.job,args.lease)
    elif args.command == "recover":
        print(json.dumps(store.recover()))
    elif args.command == "forecast":
        print(json.dumps(store.forecast(args.dataset,replay=args.replay),indent=2))
    elif args.command == "errors":
        print(json.dumps(store.errors(),indent=2))
    elif args.command == "list":
        print(json.dumps(store.list(args.collection),indent=2))
    elif args.command == "backup":
        store.backup(args.destination)
        print("Verified backup created")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, PermissionError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
