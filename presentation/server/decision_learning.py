"""Human-taught, image-embedding signal recommendations for local review.

This learner has no access to simulator state, creates no labels, and never
actuates lights. Calibrated probabilities estimate agreement with human action
labels in the same image domain; they are not probabilities of road safety.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

import numpy as np

ACTIONS = ("ns", "ew", "walk", "hold")
DOMAINS = ("street", "simulation")
REQUIREMENTS = {"min_samples": 80, "min_groups": 12, "min_per_action": 8}
CONFIDENCE_THRESHOLD = 0.90
SIMILARITY_THRESHOLD = 0.5
RELIABILITY_TARGET = 0.90
TARGET_REQUIREMENTS = {
    "min_predictions": 40, "min_groups": 3, "min_per_action": 5,
    "lower_confidence_bound": RELIABILITY_TARGET,
}
INTERVAL_LIMITATION = (
    "Approximate per-frame Wilson interval; correlated frames within a recording "
    "can make nominal 95% coverage optimistic. Requiring multiple recordings "
    "does not remove this dependence or establish road-safety reliability."
)
MODEL_SCHEMA_VERSION = 2
MAX_DIMENSIONS = 2048
MAX_RECORDS_PER_DOMAIN = 20000


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _domain(value: str) -> str:
    if value not in DOMAINS:
        raise ValueError("source_domain must be street or simulation")
    return value


def _text(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    return value.strip()


def _embedding(value: Any) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("embedding must be a finite numeric vector")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must be a finite numeric vector") from exc
    if array.ndim != 1 or not 1 <= array.size <= MAX_DIMENSIONS:
        raise ValueError(f"embedding requires 1 to {MAX_DIMENSIONS} dimensions")
    if not np.isfinite(array).all() or np.max(np.abs(array)) > 10000:
        raise ValueError("embedding values must be finite and within +/-10000")
    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(values)
    return values / np.sum(values, axis=-1, keepdims=True)


def _support_similarity(vectors: np.ndarray, support_vectors: np.ndarray) -> np.ndarray:
    """Nearest fitted-image cosine, bounded working memory for large test sets."""
    vectors = np.atleast_2d(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors / np.maximum(norms[:, None], 1e-12)
    similarities = np.empty(vectors.shape[0], dtype=np.float64)
    for start in range(0, vectors.shape[0], 128):
        block = normalized[start:start + 128] @ support_vectors.T
        similarities[start:start + 128] = np.max(block, axis=1)
    similarities[norms <= 1e-12] = -1.0
    return similarities


def _wilson_interval(correct: int, count: int) -> dict:
    """Descriptive binomial interval, with explicit within-recording caveat."""
    interval = {"level": 0.95, "method": "wilson", "lower": None,
                "upper": None, "n": count, "correct": correct,
                "approximate_frame_interval": True, "limitation": INTERVAL_LIMITATION}
    if not count:
        return interval
    z = 1.959963984540054
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return {**interval, "lower": max(0.0, center - half_width), "upper": min(1.0, center + half_width)}


def _review_metrics(probabilities: np.ndarray, labels: np.ndarray,
                    group_ids: list[str], similarities: np.ndarray) -> dict:
    """Evaluate exactly the two abstention gates used for predictions."""
    predictions = probabilities.argmax(axis=1)
    accepted = ((probabilities.max(axis=1) >= CONFIDENCE_THRESHOLD)
                & (similarities >= SIMILARITY_THRESHOLD))
    count = int(accepted.sum())
    correct = int(np.sum(predictions[accepted] == labels[accepted]))
    groups = len({group_ids[index] for index in np.flatnonzero(accepted)})
    label_counts = {action: int(np.sum(labels[accepted] == index)) for index, action in enumerate(ACTIONS)}
    interval = _wilson_interval(correct, count)
    reasons = []
    if count < TARGET_REQUIREMENTS["min_predictions"]:
        reasons.append(f'Need at least {TARGET_REQUIREMENTS["min_predictions"]} held-out recommended frames.')
    if groups < TARGET_REQUIREMENTS["min_groups"]:
        reasons.append(f'Need recommendations across at least {TARGET_REQUIREMENTS["min_groups"]} held-out recording/session groups.')
    if any(value < TARGET_REQUIREMENTS["min_per_action"] for value in label_counts.values()):
        reasons.append(f'Need at least {TARGET_REQUIREMENTS["min_per_action"]} recommended held-out examples of every human action.')
    if interval["lower"] is None or interval["lower"] < RELIABILITY_TARGET:
        reasons.append('The approximate 95% interval lower bound has not reached 90% human-label agreement.')
    target_met = not reasons
    return {
        "review_coverage": float(np.mean(accepted)) if len(accepted) else 0.0,
        "review_accuracy": correct / count if count else None,
        "review_count": count, "review_group_count": groups,
        "review_label_counts": label_counts, "review_accuracy_interval": interval,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "reliability_target": RELIABILITY_TARGET, "target_met": target_met,
        "target_requirements": dict(TARGET_REQUIREMENTS),
        "target_reason": ('The descriptive 90% held-out human-agreement criterion is met; this is not a road-safety guarantee.'
                          if target_met else ' '.join(reasons)),
        "target_scope": "human-label agreement among non-abstained held-out frames",
        "approximate_frame_interval": True, "interval_limitation": INTERVAL_LIMITATION,
    }


class DecisionLearner:
    """Persistent human labels, frozen group partitions, and calibrated softmax.

    SQLite stores plain JSON parameters, never executable model pickles. A label
    correction updates the existing image and invalidates the fitted model.
    Groups are never reassigned after the first split; reporting the same test
    groups after later corrections/retraining is explicitly retrospective.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "human_decisions.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS feedback (
                    domain TEXT NOT NULL, image_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL, embedding_version TEXT NOT NULL,
                    dimension INTEGER NOT NULL, group_id TEXT NOT NULL,
                    chosen_action TEXT NOT NULL, captured_at TEXT NOT NULL,
                    proposed_probabilities TEXT, proposed_model_version TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY (domain, image_hash)
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    domain TEXT PRIMARY KEY, revision INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS partitions (
                    domain TEXT NOT NULL, group_id TEXT NOT NULL,
                    partition_name TEXT NOT NULL,
                    PRIMARY KEY (domain, group_id)
                );
                CREATE TABLE IF NOT EXISTS models (
                    domain TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    model_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS feedback_group ON feedback(domain, group_id);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _revision(db: sqlite3.Connection, domain: str) -> int:
        row = db.execute("SELECT revision FROM revisions WHERE domain=?", (domain,)).fetchone()
        return int(row[0]) if row else 0

    def status(self, domain: str) -> dict:
        domain = _domain(domain)
        with self._lock, self._connect() as db:
            return self._status(db, domain)

    def _status(self, db: sqlite3.Connection, domain: str) -> dict:
        rows = db.execute("SELECT chosen_action, COUNT(*) AS n FROM feedback WHERE domain=? GROUP BY chosen_action", (domain,)).fetchall()
        counts = {action: 0 for action in ACTIONS}
        counts.update({row["chosen_action"]: row["n"] for row in rows})
        count = sum(counts.values())
        groups = db.execute("SELECT COUNT(DISTINCT group_id) FROM feedback WHERE domain=?", (domain,)).fetchone()[0]
        saved = db.execute("SELECT revision, model_json FROM models WHERE domain=?", (domain,)).fetchone()
        model = json.loads(saved["model_json"]) if saved else None
        protocol_current = bool(model and model.get("policy_schema_version") == MODEL_SCHEMA_VERSION)
        fresh = bool(saved and protocol_current and saved["revision"] == self._revision(db, domain))
        missing = self._requirements(counts, groups)
        reason = ("Calibrated recommendations available for human review. " + model["metrics"].get("target_reason", "The 90% human-agreement target has not been evaluated.") if fresh else
                  "The evaluation protocol changed; retrain to measure the 90% human-agreement target." if saved and not protocol_current else
                  "Human labels changed; retrain before using recommendations." if saved else
                  "Add human decisions to train this image domain. " + " ".join(missing) if missing else
                  "Enough labels for a training attempt; held-out group coverage still needs checking.")
        return {
            "domain": domain, "trained": fresh, "count": count, "label_counts": counts,
            "group_count": groups, "requirements": dict(REQUIREMENTS), "reason": reason,
            "calibrated": fresh, "model_version": model["model_version"] if model else None,
            "embedding_version": model["embedding_version"] if model else None,
            "metrics": model["metrics"] if model else None,
            "evaluation_scope": "held_out_human_action_labels", "road_safety_validated": False,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "review_only": True,
            "reliability_target": RELIABILITY_TARGET,
            "target_met": model["metrics"].get("target_met") if fresh else None,
            "target_reason": model["metrics"].get("target_reason") if fresh else "No current held-out human-label evaluation is available.",
            "target_requirements": dict(TARGET_REQUIREMENTS),
            "interval_limitation": INTERVAL_LIMITATION,
            "training_eligible": not missing, "missing_requirements": missing,
            "evaluation_reused": bool(model and model["evaluation_reused"]),
        }

    @staticmethod
    def _requirements(counts: dict, groups: int) -> list[str]:
        reasons = []
        if sum(counts.values()) < REQUIREMENTS["min_samples"]:
            reasons.append(f'Need {REQUIREMENTS["min_samples"]} human-labeled images.')
        if groups < REQUIREMENTS["min_groups"]:
            reasons.append(f'Need {REQUIREMENTS["min_groups"]} independent recordings or simulation sessions.')
        deficient = [name for name in ACTIONS if counts[name] < REQUIREMENTS["min_per_action"]]
        if deficient:
            reasons.append(f'Need {REQUIREMENTS["min_per_action"]} labels for each action; missing: {", ".join(deficient)}.')
        return reasons

    def record_feedback(self, *, image_hash: str, embedding: Any, group_id: str,
                        source_domain: str, chosen_action: str, captured_at: Any,
                        embedding_version: str, proposed_probabilities: dict | None = None,
                        model_version: str | None = None) -> dict:
        domain = _domain(source_domain)
        if not isinstance(image_hash, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", image_hash):
            raise ValueError("image_hash must be the SHA256 of the source image")
        image_hash = image_hash.lower()
        vector = _embedding(embedding)
        group_id = _text(group_id, "group_id")
        version = _text(embedding_version, "embedding_version")
        if chosen_action not in ACTIONS:
            raise ValueError("chosen_action must be ns, ew, walk, or hold")
        if isinstance(captured_at, bool) or not isinstance(captured_at, (int, float, str)):
            raise ValueError("captured_at must be a finite number or nonempty timestamp string")
        if isinstance(captured_at, str):
            _text(captured_at, "captured_at", 128)
        elif not math.isfinite(captured_at) or captured_at < 0:
            raise ValueError("captured_at must be finite and nonnegative")
        if model_version is not None:
            model_version = _text(model_version, "model_version")
        if proposed_probabilities is not None:
            if not isinstance(proposed_probabilities, dict) or set(proposed_probabilities) != set(ACTIONS):
                raise ValueError("proposed_probabilities must include exactly four actions")
            probabilities = np.asarray([proposed_probabilities[a] for a in ACTIONS], dtype=float)
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or abs(float(probabilities.sum()) - 1) > 1e-5:
                raise ValueError("proposed probabilities must be finite, nonnegative, and sum to one")
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM feedback WHERE domain=? AND image_hash=?", (domain, image_hash)).fetchone()
            contract = db.execute("SELECT embedding_version, dimension FROM feedback WHERE domain=? LIMIT 1", (domain,)).fetchone()
            if contract and (contract["embedding_version"] != version or contract["dimension"] != vector.size):
                raise ValueError("This domain uses another embedding version or dimension; do not mix image encoders.")
            if existing and existing["group_id"] != group_id:
                raise ValueError("The same image must retain its original recording/session group.")
            if existing and not np.array_equal(np.asarray(json.loads(existing["embedding"])), vector):
                raise ValueError("The same image and encoder must retain their original embedding.")
            if not existing:
                count = db.execute("SELECT COUNT(*) FROM feedback WHERE domain=?", (domain,)).fetchone()[0]
                if count >= MAX_RECORDS_PER_DOMAIN:
                    raise ValueError("Local human-label capacity reached; export records before extending this study.")
            changed = not existing or existing["chosen_action"] != chosen_action
            db.execute("""INSERT INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(domain,image_hash) DO UPDATE SET
                chosen_action=excluded.chosen_action, captured_at=excluded.captured_at,
                proposed_probabilities=excluded.proposed_probabilities,
                proposed_model_version=excluded.proposed_model_version, updated_at=excluded.updated_at""",
                (domain, image_hash, _json(vector.tolist()), version, vector.size, group_id,
                 chosen_action, _json(captured_at), _json(proposed_probabilities) if proposed_probabilities is not None else None,
                 model_version, existing["created_at"] if existing else now, now))
            if changed:
                db.execute("INSERT INTO revisions VALUES(?,1) ON CONFLICT(domain) DO UPDATE SET revision=revision+1", (domain,))
            result = self._status(db, domain)
            return {"ok": True, "created": existing is None, "corrected": bool(existing and changed), **result}

    def export_records(self, domain: str) -> list[dict]:
        domain = _domain(domain)
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT f.*, p.partition_name FROM feedback f LEFT JOIN partitions p
                ON f.domain=p.domain AND f.group_id=p.group_id WHERE f.domain=? ORDER BY f.created_at, f.image_hash""", (domain,)).fetchall()
            return [{
                "image_hash": row["image_hash"], "embedding": json.loads(row["embedding"]),
                "embedding_version": row["embedding_version"], "group_id": row["group_id"],
                "source_domain": row["domain"], "chosen_action": row["chosen_action"],
                "captured_at": json.loads(row["captured_at"]),
                "proposed_probabilities": json.loads(row["proposed_probabilities"]) if row["proposed_probabilities"] else None,
                "model_version": row["proposed_model_version"], "partition": row["partition_name"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "label_source": "human", "road_safety_validated": False,
            } for row in rows]

    def train(self, domain: str) -> dict:
        """One fixed-C fit, scalar temperature on calibration, one test report.

        Test values never select hyperparameters, calibration temperature, or
        the fixed 0.90 review threshold. Repeated evaluation remains labeled as
        retrospective rather than presented as fresh validation.
        """
        from scipy.optimize import minimize_scalar
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
        from sklearn.preprocessing import StandardScaler

        domain = _domain(domain)
        with self._lock:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                status = self._status(db, domain)
                if status["missing_requirements"]:
                    return {**status, "ok": False}
                rows = db.execute("SELECT * FROM feedback WHERE domain=? ORDER BY image_hash", (domain,)).fetchall()
                groups = sorted({row["group_id"] for row in rows}, key=lambda group: hashlib.sha256((domain + ":" + group).encode()).hexdigest())
                assignments = {row["group_id"]: row["partition_name"] for row in db.execute("SELECT * FROM partitions WHERE domain=?", (domain,))}
                if not assignments:
                    fit_end = len(groups) // 2
                    calibration_end = fit_end + (len(groups) - fit_end) // 2
                    assignments = {group: "fit" if index < fit_end else "calibration" if index < calibration_end else "test" for index, group in enumerate(groups)}
                else:
                    for group in groups:
                        if group not in assignments:
                            bucket = int(hashlib.sha256((domain + ":" + group).encode()).hexdigest()[:8], 16) % 4
                            assignments[group] = "fit" if bucket < 2 else "calibration" if bucket == 2 else "test"
                for group, partition in assignments.items():
                    db.execute("INSERT OR IGNORE INTO partitions VALUES(?,?,?)", (domain, group, partition))
                revision = self._revision(db, domain)
                evaluation_reused = db.execute("SELECT 1 FROM models WHERE domain=?", (domain,)).fetchone() is not None
            partitions = {name: [row for row in rows if assignments[row["group_id"]] == name] for name in ("fit", "calibration", "test")}
            split_report = {}
            reasons = []
            for name, subset in partitions.items():
                counts = Counter(row["chosen_action"] for row in subset)
                group_count = len({row["group_id"] for row in subset})
                split_report[name] = {"count": len(subset), "group_count": group_count, "label_counts": {action: counts[action] for action in ACTIONS}}
                minimum = 40 if name == "fit" else 16
                class_minimum = 3 if name == "fit" else 2
                if len(subset) < minimum or group_count < 3 or any(counts[action] < class_minimum for action in ACTIONS):
                    reasons.append(f"{name} requires {minimum} images, 3 independent groups, and {class_minimum} labels per action; add independent examples without reassigning existing groups.")
            if reasons:
                return {**self.status(domain), "ok": False, "reason": " ".join(reasons), "partitions": split_report}
            matrices = {name: np.asarray([json.loads(row["embedding"]) for row in subset]) for name, subset in partitions.items()}
            labels = {name: np.asarray([ACTIONS.index(row["chosen_action"]) for row in subset]) for name, subset in partitions.items()}
            scaler = StandardScaler().fit(matrices["fit"])
            classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            classifier.fit(scaler.transform(matrices["fit"]), labels["fit"])
            calibration_logits = classifier.decision_function(scaler.transform(matrices["calibration"]))
            def calibration_loss(log_temperature):
                probabilities = _softmax(calibration_logits / math.exp(log_temperature))
                return log_loss(labels["calibration"], probabilities, labels=list(range(4)))
            optimization = minimize_scalar(calibration_loss, bounds=(math.log(0.25), math.log(10)), method="bounded")
            temperature = float(math.exp(optimization.x))
            logits = classifier.decision_function(scaler.transform(matrices["test"]))
            probabilities = _softmax(logits / temperature)
            predicted = probabilities.argmax(axis=1)
            expected = np.eye(4)[labels["test"]]
            fit_vectors = matrices["fit"]
            norms = np.linalg.norm(fit_vectors, axis=1, keepdims=True)
            support_vectors = fit_vectors / np.maximum(norms, 1e-12)
            similarities = _support_similarity(matrices["test"], support_vectors)
            selective = _review_metrics(probabilities, labels["test"],
                                        [row["group_id"] for row in partitions["test"]], similarities)
            metrics = {
                "balanced_accuracy": float(balanced_accuracy_score(labels["test"], predicted)),
                "accuracy": float(np.mean(predicted == labels["test"])),
                "brier_score": float(np.mean(np.sum((probabilities - expected) ** 2, axis=1))),
                "brier_definition": "mean summed squared error over four classes (range 0..2)",
                "log_loss": float(log_loss(labels["test"], probabilities, labels=list(range(4)))),
                "confusion_matrix": confusion_matrix(labels["test"], predicted, labels=list(range(4))).tolist(),
                "action_order": list(ACTIONS), "partitions": split_report,
                **selective,
                "temperature": temperature,
                "validation_note": "Held-out recording/session groups, human action agreement only. No safety or traffic-delay validation. Repeated use of these evaluation groups is retrospective.",
            }
            model = {
                "embedding_version": rows[0]["embedding_version"], "dimension": int(matrices["fit"].shape[1]),
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coefficients": classifier.coef_.tolist(), "intercepts": classifier.intercept_.tolist(),
                "temperature": temperature, "support_vectors": support_vectors.tolist(),
                "metrics": metrics, "evaluation_reused": evaluation_reused,
                "trained_at": time.time(), "domain": domain, "revision": revision,
                "policy_schema_version": MODEL_SCHEMA_VERSION,
            }
            model["model_version"] = "human-policy-" + hashlib.sha256(_json(model).encode()).hexdigest()[:16]
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if self._revision(db, domain) != revision:
                    return {**self._status(db, domain), "ok": False, "reason": "Labels changed during training; retry training with the current labels."}
                db.execute("INSERT INTO models VALUES(?,?,?) ON CONFLICT(domain) DO UPDATE SET revision=excluded.revision, model_json=excluded.model_json", (domain, revision, _json(model)))
            return {**self.status(domain), "ok": True}

    def predict(self, embedding: Any, domain: str, embedding_version: str | None = None) -> dict:
        domain = _domain(domain)
        vector = _embedding(embedding)
        with self._lock, self._connect() as db:
            status = self._status(db, domain)
            baseline = {"domain": domain, "action": None, "suggested_action": None,
                        "probabilities": None, "confidence": None, "calibrated": False,
                        "abstain": True, "model_version": status["model_version"],
                        "review_only": True, "road_safety_validated": False,
                        "evaluation_scope": "held_out_human_action_labels",
                        "confidence_threshold": CONFIDENCE_THRESHOLD,
                        "reliability_target": RELIABILITY_TARGET,
                        "target_met": status["target_met"], "target_reason": status["target_reason"],
                        "interval_limitation": INTERVAL_LIMITATION}
            if not status["trained"]:
                return {**baseline, "reason": status["reason"]}
            model = json.loads(db.execute("SELECT model_json FROM models WHERE domain=?", (domain,)).fetchone()[0])
            if vector.size != model["dimension"] or embedding_version != model["embedding_version"]:
                return {**baseline, "reason": "Image encoder version or dimension does not match this domain's trained model."}
            standardized = (vector - np.asarray(model["mean"])) / np.asarray(model["scale"])
            logits = np.asarray(model["coefficients"]) @ standardized + np.asarray(model["intercepts"])
            probabilities = _softmax(logits / model["temperature"])
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            similarity = float(_support_similarity(vector, np.asarray(model["support_vectors"]))[0])
            supported = similarity >= SIMILARITY_THRESHOLD
            abstain = confidence < CONFIDENCE_THRESHOLD or not supported
            reason = ("Image embedding differs from labeled training examples; human review required." if not supported else
                      "Calibrated confidence is below the fixed 90% review threshold; this threshold is not measured reliability." if abstain else
                      "Recommendation predicts human action labels; review before applying through the existing clearance controller.")
            return {**baseline, "action": None if abstain else ACTIONS[index], "suggested_action": ACTIONS[index],
                    "probabilities": {action: float(probabilities[i]) for i, action in enumerate(ACTIONS)},
                    "confidence": confidence, "calibrated": True, "abstain": abstain,
                    "nearest_training_similarity": similarity, "reason": reason}
