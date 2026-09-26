"""Single-workspace SQLite observatory. Local tokens, immutable evidence, no network I/O."""
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import secrets
import sqlite3
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from .observations import validate_csv

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,name TEXT UNIQUE NOT NULL,role TEXT NOT NULL,
 token_hash TEXT UNIQUE NOT NULL,expires REAL NOT NULL,revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,actor TEXT,created REAL,source TEXT,mode TEXT,
 stream TEXT,cadence INTEGER,checksum TEXT,quality TEXT,content BLOB);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,actor TEXT,dataset TEXT,status TEXT,created REAL,
 updated REAL,lease TEXT,attempt INTEGER DEFAULT 1,error TEXT,cancel INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY,job TEXT UNIQUE,dataset TEXT,stream TEXT,
 created REAL,metadata TEXT,payload BLOB,metadata_hash TEXT);
CREATE TABLE IF NOT EXISTS active_models(stream TEXT PRIMARY KEY,model TEXT);
CREATE TABLE IF NOT EXISTS forecasts(id TEXT PRIMARY KEY,actor TEXT,dataset TEXT,stream TEXT,model TEXT,
 created REAL,origin TEXT,target TEXT,status TEXT,details TEXT);
CREATE TABLE IF NOT EXISTS actuals(stream TEXT,timestamp TEXT,value INTEGER,
 PRIMARY KEY(stream,timestamp));
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT,created REAL,actor TEXT,event TEXT,details TEXT);
CREATE INDEX IF NOT EXISTS job_status ON jobs(status,created);
CREATE UNIQUE INDEX IF NOT EXISTS one_forecast_per_origin ON forecasts(stream,model,origin,json_extract(details,'$.mode')) WHERE status='available';
"""


class AccessDenied(PermissionError):
    pass


class Store:
    def __init__(self, root, token=None):
        self.root = Path(root).resolve()
        self.path = self.root / "observatory.sqlite3"
        self.token = token or ""
        if not self.path.exists():
            raise ValueError("Workspace not initialized. Run operations init first.")
        with sqlite3.connect(self.path) as db:
            if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Unsupported workspace schema version; restore with its matching application")

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @classmethod
    def initialize(cls, root, name="admin"):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / "observatory.sqlite3"
        # Exclusive creation prevents two initializers handing out admin credentials.
        with path.open("xb"):
            pass
        token = secrets.token_urlsafe(32)
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("INSERT INTO users VALUES(?,?,?,?,?,0)",
                       (uuid4().hex, name, "admin", sha256(token.encode()).hexdigest(), time.time()+30*86400))
            db.execute("PRAGMA user_version=1")
        return token

    def require(self, db, roles=("admin", "operator", "viewer")):
        user = db.execute("SELECT * FROM users WHERE token_hash=? AND revoked=0 AND expires>?",
                          (sha256(self.token.encode()).hexdigest(), time.time())).fetchone()
        if not user or user["role"] not in roles:
            raise AccessDenied("Invalid, expired, revoked or insufficiently privileged access token")
        return user

    @staticmethod
    def audit(db, actor, event, details):
        db.execute("INSERT INTO audit(created,actor,event,details) VALUES(?,?,?,?)",
                   (time.time(), actor, event, json.dumps(details)))

    def identity(self):
        with self.connect() as db:
            u = self.require(db)
            return {k:u[k] for k in ("id", "name", "role", "expires")}

    def add_user(self, name, role):
        if role not in ("admin", "operator", "viewer") or not 1 <= len(name) <= 100:
            raise ValueError("Invalid name or role")
        token = secrets.token_urlsafe(32)
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))["id"]
            if db.execute("SELECT count(*) FROM users WHERE revoked=0").fetchone()[0] >= 32:
                raise ValueError("Active-user limit reached")
            db.execute("INSERT INTO users VALUES(?,?,?,?,?,0)",
                       (uuid4().hex, name, role, sha256(token.encode()).hexdigest(), time.time()+30*86400))
            self.audit(db, actor, "user_created", {"name": name, "role":role})
        return token

    def revoke(self, name):
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))
            if name == actor["name"]:
                raise ValueError("Create a replacement administrator before revoking this token")
            db.execute("UPDATE users SET revoked=1 WHERE name=?", (name,))
            self.audit(db, actor["id"], "user_revoked", {"name":name})

    def import_csv(self, content, source, mode="historical", now=None):
        with self.connect() as db:
            self.require(db, ("admin", "operator"))
        if mode not in ("historical", "live", "synthetic") or not 1 <= len(source.strip()) <= 1000:
            raise ValueError("Declare source provenance and historical/live/synthetic mode")
        frame, quality = validate_csv(content, now=now)
        canonical = frame.to_csv(index=False).encode()
        checksum = sha256(canonical).hexdigest()
        identifier = uuid4().hex
        rows = [(quality["stream_id"], t.isoformat(), int(v)) for t,v in zip(frame.timestamp, frame.vehicle_count)]
        with self.connect(True) as db:
            actor = self.require(db, ("admin", "operator"))["id"]
            prior = db.execute("SELECT id FROM datasets WHERE checksum=? AND mode=? AND source=?", (checksum,mode,source)).fetchone()
            if prior:
                return prior[0]
            total = db.execute("SELECT coalesce(sum(length(content)),0),count(*) FROM datasets").fetchone()
            if total[0]+len(canonical) > 500*1024*1024 or total[1] >= 100:
                raise ValueError("Workspace import quota reached (100 snapshots / 500 MiB)")
            previous_stream = db.execute("SELECT cadence,quality FROM datasets WHERE stream=? LIMIT 1", (quality["stream_id"],)).fetchone()
            if previous_stream:
                anchor = pd.Timestamp(json.loads(previous_stream["quality"])["start"])
                if previous_stream["cadence"] != quality["interval_seconds"] or (frame.timestamp.iloc[0]-anchor).total_seconds() % quality["interval_seconds"] != 0:
                    raise ValueError("Stream cadence or interval grid changed; use a new stream version")
            for stream, stamp, value in rows:
                existing = db.execute("SELECT value FROM actuals WHERE stream=? AND timestamp=?", (stream,stamp)).fetchone()
                if existing and existing[0] != value:
                    raise ValueError("Conflicting actuals: corrections require a new stream version")
            db.executemany("INSERT OR IGNORE INTO actuals VALUES(?,?,?)", rows)
            db.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (identifier,actor,time.time(),source,mode,quality["stream_id"],quality["interval_seconds"],checksum,json.dumps(quality),canonical))
            self.audit(db, actor, "dataset_imported", {"id":identifier,"checksum":checksum,"quality":quality})
        return identifier

    def dataset(self, identifier):
        with self.connect() as db:
            self.require(db)
            row = db.execute("SELECT * FROM datasets WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise ValueError("Unknown dataset")
        if sha256(row["content"]).hexdigest() != row["checksum"]:
            raise ValueError("Dataset checksum mismatch")
        frame = pd.read_csv(BytesIO(row["content"]))
        frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
        return dict(row), frame

    def list(self, table):
        fields = {"datasets":"id,created,source,mode,stream,cadence,quality", "jobs":"*",
                  "models":"id,job,dataset,stream,created,metadata", "active_models":"*",
                  "audit":"*", "forecasts":"*", "users":"id,name,role,expires,revoked"}
        if table not in fields:
            raise ValueError("Unknown collection")
        with self.connect() as db:
            user = self.require(db, ("admin",) if table in ("users","audit") else ("admin","operator","viewer"))
            return [dict(row) for row in db.execute(f"SELECT {fields[table]} FROM {table} ORDER BY rowid DESC LIMIT 1000")]

    def enqueue(self, dataset):
        meta, frame = self.dataset(dataset)
        if meta["cadence"] != 3600:
            raise ValueError("This candidate supports hourly counts only; other cadences need a separately validated model")
        if len(frame) < 1000:
            raise ValueError("Need at least 1000 observations for training")
        identifier = uuid4().hex
        with self.connect(True) as db:
            actor = self.require(db, ("admin","operator"))["id"]
            if db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 5:
                raise ValueError("At most five queued/running jobs are allowed")
            db.execute("INSERT INTO jobs(id,actor,dataset,status,created,updated) VALUES(?,?,?,?,?,?)",
                       (identifier,actor,dataset,"queued",time.time(),time.time()))
            self.audit(db, actor, "job_queued", {"id":identifier,"dataset":dataset})
        return identifier

    def claim(self):
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))["id"]
            if db.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone():
                return None  # one bounded worker per workspace
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            lease = uuid4().hex
            db.execute("UPDATE jobs SET status='running',lease=?,updated=? WHERE id=?", (lease,time.time(),row["id"]))
            self.audit(db, actor, "job_started", {"id":row["id"]})
            return {**dict(row), "lease":lease}

    def heartbeat(self, job, lease):
        with self.connect(True) as db:
            self.require(db, ("admin",))
            row = db.execute("SELECT * FROM jobs WHERE id=? AND lease=? AND status='running'", (job,lease)).fetchone()
            if not row:
                return False
            db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time(),job))
            return not row["cancel"]

    def cancel(self, job):
        with self.connect(True) as db:
            actor = self.require(db, ("admin","operator"))
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if not row or (actor["role"] != "admin" and row["actor"] != actor["id"]):
                raise AccessDenied("Cannot cancel another user's job")
            db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?", (job,))
            self.audit(db, actor["id"], "job_cancel_requested", {"id":job})

    def finish(self, job, lease, *, error=None, model=None):
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))["id"]
            row = db.execute("SELECT * FROM jobs WHERE id=? AND lease=? AND status='running'", (job,lease)).fetchone()
            if not row:
                return False  # stale worker cannot commit after recovery/retry
            status = "cancelled" if row["cancel"] else "failed" if error else "succeeded"
            if status == "succeeded":
                if model is None:
                    raise ValueError("Successful job must include model evidence")
                meta,payload = model
                source = db.execute("SELECT mode,checksum FROM datasets WHERE id=?", (row["dataset"],)).fetchone()
                meta = {**meta, "source_mode":source["mode"], "dataset_checksum":source["checksum"], "dataset_id":row["dataset"]}
                record = json.dumps(meta, sort_keys=True)
                identifier = uuid4().hex
                db.execute("INSERT INTO models VALUES(?,?,?,?,?,?,?,?)", (identifier,job,row["dataset"],meta["stream_id"],time.time(),record,payload,sha256(record.encode()).hexdigest()))
            db.execute("UPDATE jobs SET status=?,updated=?,error=? WHERE id=?", (status,time.time(),str(error)[:2000] if error else None,job))
            self.audit(db, actor, "job_"+status, {"id":job,"error":str(error)[:2000] if error else None})
            return True

    def recover(self, stale_seconds=60):
        if stale_seconds < 30:
            raise ValueError("Recovery grace period must be at least 30 seconds")
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))["id"]
            ids = [r[0] for r in db.execute("SELECT id FROM jobs WHERE status='running' AND updated<?", (time.time()-stale_seconds,))]
            for job in ids:
                db.execute("UPDATE jobs SET status='failed',lease=NULL,error='Worker heartbeat expired; explicit retry required' WHERE id=?", (job,))
                self.audit(db, actor, "job_recovered", {"id":job})
            return ids

    def retry(self, job):
        with self.connect() as db:
            actor = self.require(db, ("admin","operator"))
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if not row or row["status"] not in ("failed","cancelled"):
                raise ValueError("Only failed/cancelled jobs can be retried")
            if actor["role"] != "admin" and actor["id"] != row["actor"]:
                raise AccessDenied("Cannot retry another user's job")
            dataset = row["dataset"]
        replacement = self.enqueue(dataset)  # original failure record is retained
        with self.connect(True) as db:
            actor = self.require(db, ("admin","operator"))["id"]
            self.audit(db,actor,"job_retried",{"original":job,"replacement":replacement})
        return replacement

    def model(self, identifier):
        with self.connect() as db:
            self.require(db)
            row = db.execute("SELECT * FROM models WHERE id=?", (identifier,)).fetchone()
        if not row or sha256(row["metadata"].encode()).hexdigest() != row["metadata_hash"]:
            raise ValueError("Unknown model or metadata checksum mismatch")
        meta = json.loads(row["metadata"])
        if sha256(row["payload"]).hexdigest() != meta["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        return dict(row),meta

    def activate(self, identifier):
        row,meta = self.model(identifier)
        interval = meta["evaluation"]["interval"]
        if interval["n"] < 30 or interval["coverage"] < .85:
            raise ValueError("Activation blocked: evaluation interval coverage must be at least 85% on 30+ windows; nominal target is 90%")
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))["id"]
            previous = db.execute("SELECT model FROM active_models WHERE stream=?", (row["stream"],)).fetchone()
            db.execute("INSERT INTO active_models VALUES(?,?) ON CONFLICT(stream) DO UPDATE SET model=excluded.model", (row["stream"],identifier))
            self.audit(db, actor, "model_activated", {"stream":row["stream"],"model":identifier,"previous":previous[0] if previous else None})

    def deactivate(self, stream):
        with self.connect(True) as db:
            actor = self.require(db, ("admin",))["id"]
            db.execute("DELETE FROM active_models WHERE stream=?", (stream,))
            self.audit(db, actor, "model_deactivated", {"stream":stream})

    def rotate_token(self):
        replacement = secrets.token_urlsafe(32)
        with self.connect(True) as db:
            actor = self.require(db)["id"]
            db.execute("UPDATE users SET token_hash=?,expires=? WHERE id=?", (sha256(replacement.encode()).hexdigest(),time.time()+30*86400,actor))
            self.audit(db, actor, "token_rotated", {})
        return replacement

    def forecast(self, dataset, *, replay=False, now=None):
        with self.connect() as db:
            self.require(db, ("admin","operator"))
        item,frame = self.dataset(dataset)
        origin = frame.timestamp.iloc[-1]
        current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if current.tzinfo is None:
            raise ValueError("now must have an explicit timezone")
        reasons = []
        if replay and item["mode"] == "live":
            raise ValueError("Live observations cannot bypass staleness using replay mode")
        if not replay and (current-origin).total_seconds() > 2*item["cadence"]:
            reasons.append("stale_observations")
        if origin > current:
            reasons.append("future_observations")
        tail = frame.tail(24)
        if len(tail) < 24 or not (tail.timestamp.diff().dropna().dt.total_seconds() == 3600).all():
            reasons.append("incomplete_recent_history")
        with self.connect() as db:
            active = db.execute("SELECT model FROM active_models WHERE stream=?", (item["stream"],)).fetchone()
        model_id = active[0] if active else None
        details = {"mode":"historical_replay" if replay else item["mode"], "reasons":reasons}
        if not active:
            reasons.append("no_active_model")
        if active and not reasons:
            try:
                row,meta = self.model(model_id)
                if item["mode"] != meta["source_mode"]:
                    raise ValueError("Observation mode differs from model provenance")
                if origin < pd.Timestamp(meta["evidence_through"]):
                    raise ValueError("Replay origin predates model evidence; would use future-trained information")
                if item["cadence"] != meta["interval_seconds"]:
                    raise ValueError("Cadence differs from model contract")
                recent = tail.vehicle_count.to_numpy()
                outside = np.mean((recent < meta["fit_q01"]) | (recent > meta["fit_q99"]))
                details["out_of_training_range_fraction"] = float(outside)
                if outside > .5:
                    reasons.append("distribution_shift_review_required")
                else:
                    scored = [r for r in self.errors() if r["model"]==model_id and r.get("actual") is not None and r["status"]=="available"][:100]
                    if len(scored) >= 30:
                        coverage = np.mean([r["covered"] for r in scored])
                        mae = np.mean([abs(r["error"]) for r in scored])
                        details["monitoring"] = {"n":len(scored),"interval_coverage":float(coverage),"mae":float(mae)}
                        if coverage < .85:
                            reasons.append("interval_coverage_below_threshold")
                        if mae > max(1.,meta["evaluation"]["mae"])*2:
                            reasons.append("forecast_error_deterioration")
                    if not reasons:
                        from .candidate import predict_candidate
                        details.update(predict_candidate(meta,row["payload"],frame))
                        details["nominal_coverage"] = 1-meta["calibration"]["alpha"]
            except (ValueError, RuntimeError) as exc:
                reasons.append("model_unavailable")
                details["error"] = str(exc)
        status = "unavailable" if reasons else "available"
        identifier = uuid4().hex
        target = (origin+pd.Timedelta(seconds=item["cadence"])).isoformat()
        with self.connect(True) as db:
            actor = self.require(db, ("admin","operator"))["id"]
            if status == "available":
                existing = db.execute("SELECT id,details FROM forecasts WHERE stream=? AND model=? AND origin=? AND status='available' AND json_extract(details,'$.mode')=?", (item["stream"],model_id,origin.isoformat(),details["mode"])).fetchone()
                if existing:
                    return {"id":existing["id"],"status":"available",**json.loads(existing["details"])}
            if db.execute("SELECT count(*) FROM forecasts WHERE created>?", (time.time()-86400,)).fetchone()[0] >= 1000:
                raise ValueError("Daily forecast quota reached (1000 per workspace)")
            db.execute("INSERT INTO forecasts VALUES(?,?,?,?,?,?,?,?,?,?)", (identifier,actor,dataset,item["stream"],model_id,time.time(),origin.isoformat(),target,status,json.dumps(details)))
            self.audit(db, actor, "forecast_"+status, {"id":identifier,"reasons":reasons})
        return {"id":identifier,"status":status,**details}

    def errors(self):
        with self.connect() as db:
            self.require(db)
            rows = db.execute("SELECT f.id,f.stream,f.model,f.target,f.status,f.details,a.value AS actual FROM forecasts f LEFT JOIN actuals a ON a.stream=f.stream AND a.timestamp=f.target ORDER BY f.created DESC LIMIT 1000")
            output = []
            for row in rows:
                record = dict(row)
                details = json.loads(record.pop("details"))
                record.update(details)
                if row["status"] == "available" and row["actual"] is not None:
                    record["error"] = details["value"]-row["actual"]
                    record["covered"] = details["lower"] <= row["actual"] <= details["upper"]
                output.append(record)
            return output

    def backup(self, destination):
        destination = Path(destination)
        with self.connect() as db:
            actor = self.require(db, ("admin",))["id"]
            with destination.open("xb"):
                pass
            with sqlite3.connect(destination) as target:
                db.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Backup integrity check failed")
        with self.connect(True) as db:
            self.audit(db, actor, "backup_created", {"file":destination.name})


def restore_backup(source, destination, token):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Restore requires a new destination directory")
    with sqlite3.connect(f"file:{Path(source).resolve().as_posix()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Corrupt backup")
        valid = db.execute("SELECT 1 FROM users WHERE token_hash=? AND role='admin' AND revoked=0 AND expires>?", (sha256(token.encode()).hexdigest(),time.time())).fetchone()
        if not valid:
            raise AccessDenied("Backup administrator token required")
        destination.mkdir(parents=True)
        with sqlite3.connect(destination / "observatory.sqlite3") as target:
            db.backup(target)
    return Store(destination,token)
