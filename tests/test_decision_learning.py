"""Behavioral checks for human-only decision learning, not road validation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "presentation" / "server" / "decision_learning.py"
spec = importlib.util.spec_from_file_location("decision_learning", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
DecisionLearner = module.DecisionLearner
ACTIONS = module.ACTIONS
VERSION = "test-frozen-image-encoder-v1"


def feedback(learner, index=0, action="ns", domain="simulation", group="recording-0", embedding=None):
    return learner.record_feedback(
        image_hash=hashlib.sha256(f"image-{index}".encode()).hexdigest(),
        embedding=embedding if embedding is not None else [1.0, 0.0, 0.0, 0.0],
        group_id=group, source_domain=domain, chosen_action=action,
        captured_at=float(index), embedding_version=VERSION,
        proposed_probabilities={action: 0.25 for action in ACTIONS}, model_version="review-session",
    )


def populate(learner, domain="simulation", groups=12):
    rng = np.random.default_rng(2026)
    for group in range(groups):
        for action_index, action in enumerate(ACTIONS):
            for replica in range(5):
                index = group * 20 + action_index * 5 + replica
                vector = np.eye(4)[action_index] * 4 + rng.normal(0, 0.05, 4)
                feedback(learner, index, action, domain, f"independent-recording-{group}", vector.tolist())


@pytest.fixture(scope="module")
def trained_template(tmp_path_factory):
    learner = DecisionLearner(tmp_path_factory.mktemp("trained_policy"))
    populate(learner)
    result = learner.train("simulation")
    assert result["ok"]
    return learner.database


@pytest.fixture
def trained(tmp_path, trained_template):
    # A closed SQLite database can be copied to keep mutation tests isolated
    # without refitting the same deterministic classifier in every test.
    (tmp_path / "human_decisions.sqlite3").write_bytes(trained_template.read_bytes())
    return DecisionLearner(tmp_path)


def test_empty_state_never_claims_training_or_confidence(tmp_path):
    learner = DecisionLearner(tmp_path)
    status = learner.status("street")
    assert status["trained"] is False
    assert status["count"] == 0 and status["group_count"] == 0
    assert status["metrics"] is None and status["calibrated"] is False
    assert status["target_met"] is None and status["reliability_target"] == 0.90
    assert status["confidence_threshold"] == 0.90
    assert status["requirements"] == {"min_samples": 80, "min_groups": 12, "min_per_action": 8}
    prediction = learner.predict([1, 0, 0, 0], "street", VERSION)
    assert prediction["action"] is None and prediction["abstain"]
    assert prediction["confidence"] is None and prediction["probabilities"] is None
    assert not learner.train("street")["ok"]


def test_feedback_is_persistent_human_data_with_corrected_duplicates(tmp_path):
    learner = DecisionLearner(tmp_path)
    assert feedback(learner)["created"]
    correction = feedback(learner, action="hold")
    assert correction["corrected"] and not correction["created"]
    reloaded = DecisionLearner(tmp_path)
    assert reloaded.status("simulation")["count"] == 1
    records = reloaded.export_records("simulation")
    assert records[0]["chosen_action"] == "hold"
    assert records[0]["label_source"] == "human"
    assert records[0]["proposed_probabilities"] == dict.fromkeys(ACTIONS, 0.25)
    assert records[0]["model_version"] == "review-session"
    assert records[0]["partition"] is None
    with pytest.raises(ValueError, match="original recording"):
        feedback(reloaded, group="another-recording")
    with pytest.raises(ValueError, match="original embedding"):
        feedback(reloaded, embedding=[0, 1, 0, 0])


def test_labels_cannot_be_mixed_between_image_encoders(tmp_path):
    learner = DecisionLearner(tmp_path)
    feedback(learner)
    with pytest.raises(ValueError, match="another embedding"):
        learner.record_feedback(image_hash="a" * 64, embedding=[1, 2], group_id="one",
            source_domain="simulation", chosen_action="ns", captured_at=1,
            embedding_version="another-encoder")
    assert learner.status("simulation")["count"] == 1


@pytest.mark.parametrize("vector", [[float("nan")], [float("inf")], [], [[1, 2]], [10001], [0] * 2049])
def test_nonfinite_or_unbounded_embeddings_rejected(tmp_path, vector):
    learner = DecisionLearner(tmp_path)
    with pytest.raises(ValueError):
        feedback(learner, embedding=vector)
    assert learner.status("simulation")["count"] == 0


def test_many_frames_from_one_recording_cannot_pass_group_requirement(tmp_path):
    learner = DecisionLearner(tmp_path)
    for i in range(100):
        feedback(learner, i, ACTIONS[i % 4])
    result = learner.train("simulation")
    assert not result["ok"]
    assert "12 independent recordings" in result["reason"]


def test_actual_fit_calibration_and_test_report_are_separate(trained):
    status = trained.status("simulation")
    metrics = status["metrics"]
    assert status["trained"] and status["calibrated"]
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["brier_score"] < 0.01
    assert sum(map(sum, metrics["confusion_matrix"])) == metrics["partitions"]["test"]["count"]
    groups = {}
    for record in trained.export_records("simulation"):
        groups.setdefault(record["partition"], set()).add(record["group_id"])
    assert set(groups) == {"fit", "calibration", "test"}
    assert groups["fit"].isdisjoint(groups["calibration"] | groups["test"])
    assert groups["calibration"].isdisjoint(groups["test"])
    assert not status["road_safety_validated"]
    assert not status["evaluation_reused"]


def test_predicts_learned_human_actions_and_abstains_without_support(trained):
    for index, action in enumerate(ACTIONS):
        prediction = trained.predict((np.eye(4)[index] * 4).tolist(), "simulation", VERSION)
        assert prediction["action"] == action and not prediction["abstain"]
        assert prediction["confidence"] > 0.9 and prediction["calibrated"]
        assert sum(prediction["probabilities"].values()) == pytest.approx(1)
        assert prediction["review_only"]
    uncertain = trained.predict([1, 1, 1, 1], "simulation", VERSION)
    assert uncertain["action"] is None and uncertain["abstain"]
    different = trained.predict([-4, 0, 0, 0], "simulation", VERSION)
    assert different["action"] is None and different["abstain"]
    assert "differs" in different["reason"]


def test_model_cannot_cross_domain_or_encoder_boundaries(trained):
    for domain, version, vector in [("street", VERSION, [4, 0, 0, 0]),
                                     ("simulation", "wrong", [4, 0, 0, 0]),
                                     ("simulation", VERSION, [4, 0]),
                                     ("simulation", None, [4, 0, 0, 0])]:
        prediction = trained.predict(vector, domain, version)
        assert prediction["abstain"] and prediction["probabilities"] is None


def test_label_corrections_invalidate_model_but_duplicate_submissions_do_not(trained):
    record = trained.export_records("simulation")[0]
    kwargs = {key: record[key] for key in ("image_hash", "embedding", "group_id", "source_domain", "chosen_action", "captured_at", "embedding_version")}
    trained.record_feedback(**kwargs)
    assert trained.status("simulation")["trained"]
    kwargs["chosen_action"] = ACTIONS[(ACTIONS.index(kwargs["chosen_action"]) + 1) % 4]
    trained.record_feedback(**kwargs)
    assert not trained.status("simulation")["trained"]
    assert trained.predict(record["embedding"], "simulation", VERSION)["abstain"]


def test_test_labels_never_tune_fit_or_calibration_and_groups_stay_frozen(trained):
    original_prediction = trained.predict([4, 0, 0, 0], "simulation", VERSION)
    records = trained.export_records("simulation")
    original_groups = {row["group_id"]: row["partition"] for row in records}
    for record in records:
        if record["partition"] == "test":
            kwargs = {key: record[key] for key in ("image_hash", "embedding", "group_id", "source_domain", "chosen_action", "captured_at", "embedding_version")}
            kwargs["chosen_action"] = ACTIONS[(ACTIONS.index(record["chosen_action"]) + 1) % 4]
            trained.record_feedback(**kwargs)
    result = trained.train("simulation")
    assert result["ok"] and result["evaluation_reused"]
    assert result["metrics"]["accuracy"] == 0
    new_prediction = trained.predict([4, 0, 0, 0], "simulation", VERSION)
    assert new_prediction["probabilities"] == original_prediction["probabilities"]
    assert {row["group_id"]: row["partition"] for row in trained.export_records("simulation")} == original_groups


def test_concurrent_labels_are_not_lost(tmp_path):
    learner = DecisionLearner(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: feedback(learner, i, ACTIONS[i % 4]), range(24)))
    assert learner.status("simulation")["count"] == 24


def test_invalid_feedback_metadata_is_rejected(tmp_path):
    learner = DecisionLearner(tmp_path)
    base = dict(image_hash="a" * 64, embedding=[1, 0], group_id="recording",
                source_domain="street", chosen_action="hold", captured_at=0,
                embedding_version=VERSION)
    for invalid in [dict(image_hash="../unsafe"), dict(chosen_action="automatic"),
                    dict(source_domain="mixed"), dict(captured_at=float("nan")),
                    dict(captured_at=-1), dict(group_id=""),
                    dict(proposed_probabilities=dict.fromkeys(ACTIONS, 0.9))]:
        with pytest.raises(ValueError):
            learner.record_feedback(**(base | invalid))
    assert learner.status("street")["count"] == 0


def test_selective_metrics_apply_both_actual_abstention_gates():
    probabilities = np.asarray([[0.96, 0.02, 0.01, 0.01],
                                [0.96, 0.02, 0.01, 0.01],
                                [0.89, 0.05, 0.03, 0.03]])
    metrics = module._review_metrics(probabilities, np.asarray([0, 0, 0]),
                                     ["one", "two", "three"], np.asarray([0.8, 0.49, 0.8]))
    assert metrics["review_count"] == 1
    assert metrics["review_coverage"] == pytest.approx(1 / 3)
    assert metrics["review_accuracy"] == 1.0
    assert not metrics["target_met"]
    assert metrics["confidence_threshold"] == 0.90
    assert metrics["review_accuracy_interval"]["approximate_frame_interval"]
    assert "correlated frames" in metrics["interval_limitation"]


def test_high_confidence_or_observed_90_percent_does_not_establish_target():
    labels = np.arange(40) % 4
    predictions = labels.copy()
    predictions[:4] = (predictions[:4] + 1) % 4
    probabilities = np.full((40, 4), 0.001)
    probabilities[np.arange(40), predictions] = 0.997
    metrics = module._review_metrics(probabilities, labels,
                                     [f"recording-{i % 4}" for i in range(40)], np.ones(40))
    assert metrics["review_accuracy"] == 0.90
    assert metrics["review_accuracy_interval"]["lower"] < 0.90
    assert not metrics["target_met"]
    assert metrics["review_count"] == 40


def test_target_requires_adequate_examples_groups_and_action_coverage():
    labels = np.arange(40) % 4
    probabilities = np.full((40, 4), 0.001)
    probabilities[np.arange(40), labels] = 0.997
    groups = [f"recording-{i % 4}" for i in range(40)]
    good = module._review_metrics(probabilities, labels, groups, np.ones(40))
    assert good["target_met"]
    assert good["review_accuracy_interval"]["lower"] >= 0.90
    too_few = module._review_metrics(probabilities[:36], labels[:36], groups[:36], np.ones(36))
    assert not too_few["target_met"]
    one_group = module._review_metrics(probabilities, labels, ["same-video"] * 40, np.ones(40))
    assert not one_group["target_met"]
    one_action_probabilities = np.tile([0.997, 0.001, 0.001, 0.001], (40, 1))
    one_action = module._review_metrics(one_action_probabilities, np.zeros(40, dtype=int), groups, np.ones(40))
    assert not one_action["target_met"]


def test_reported_coverage_matches_runtime_predictions(trained):
    records = [row for row in trained.export_records("simulation") if row["partition"] == "test"]
    predictions = [trained.predict(row["embedding"], "simulation", VERSION) for row in records]
    accepted = [i for i, prediction in enumerate(predictions) if not prediction["abstain"]]
    metrics = trained.status("simulation")["metrics"]
    assert metrics["review_count"] == len(accepted)
    assert metrics["review_coverage"] == pytest.approx(len(accepted) / len(records))
    correct = sum(predictions[i]["action"] == records[i]["chosen_action"] for i in accepted)
    assert metrics["review_accuracy"] == pytest.approx(correct / len(accepted))
    assert all(prediction["confidence_threshold"] == 0.90 for prediction in predictions)
    assert all(prediction["reliability_target"] == 0.90 for prediction in predictions)


def test_older_evaluation_protocol_requires_retraining(trained):
    import json
    import sqlite3
    with sqlite3.connect(trained.database) as db:
        model = json.loads(db.execute("SELECT model_json FROM models WHERE domain='simulation'").fetchone()[0])
        model.pop("policy_schema_version")
        db.execute("UPDATE models SET model_json=? WHERE domain='simulation'", (json.dumps(model),))
    status = trained.status("simulation")
    assert not status["trained"]
    assert status["target_met"] is None
    assert "protocol changed" in status["reason"]
    assert trained.predict([4, 0, 0, 0], "simulation", VERSION)["abstain"]
