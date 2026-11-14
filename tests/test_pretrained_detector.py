"""Detector integration contracts; fabricated boxes are test fixtures only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

PATH = Path(__file__).parents[1] / "presentation/server/pretrained_detector.py"
spec = importlib.util.spec_from_file_location("pretrained_detector", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def labels():
    result = ["???"] * 90
    for category_id, name in module.ROAD_CLASSES.items():
        result[category_id - 1] = name
    return result


def detection(index=None, name="car", score=0.9, box=(10, 20, 30, 40)):
    return SimpleNamespace(categories=[SimpleNamespace(index=index, category_name=name, score=score)],
                           bounding_box=SimpleNamespace(origin_x=box[0], origin_y=box[1], width=box[2], height=box[3]))


def result(*detections):
    return SimpleNamespace(detections=list(detections))


@pytest.fixture
def fake_loaded(tmp_path, monkeypatch):
    model = tmp_path / module.MODEL_FILENAME
    model.write_bytes(b"isolated model fixture")
    labelmap = tmp_path / "labelmap.txt"
    labelmap.write_text("\n".join(labels()), encoding="utf-8")
    monkeypatch.setattr(module, "MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "LABELMAP_SHA256", hashlib.sha256(labelmap.read_bytes()).hexdigest())
    calls = {"loads": 0, "images": []}
    class Engine:
        def detect(self, image):
            calls["images"].append(image)
            return result(detection(box=(0, 0, image.shape[1] / 4, image.shape[0] / 4)))
        def close(self): pass
    def create(options):
        calls["loads"] += 1
        calls["options"] = options
        return Engine()
    runtime = SimpleNamespace(
        Image=lambda **kwargs: kwargs["data"], ImageFormat=SimpleNamespace(SRGB="rgb"),
        tasks=SimpleNamespace(BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=SimpleNamespace(ObjectDetectorOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                RunningMode=SimpleNamespace(IMAGE="image"),
                ObjectDetector=SimpleNamespace(create_from_options=create))))
    monkeypatch.setattr(module, "_import_runtime", lambda: runtime)
    return module.PretrainedDetector(tmp_path), calls


def test_missing_or_corrupt_assets_never_claim_model_availability(tmp_path):
    detector = module.PretrainedDetector(tmp_path)
    assert not detector.status()["available"]
    assert not detector.status()["target_met"]
    (tmp_path / module.MODEL_FILENAME).write_bytes(b"not the official model")
    (tmp_path / "labelmap.txt").write_bytes(b"not official labels")
    status = detector.status()
    assert not status["available"] and "SHA256" in status["reason"]
    output = detector.predict(Image.new("RGB", (64, 64)))
    assert output["objects"] == [] and output["latency_ms"] is None


def test_native_missing_category_index_maps_verified_names_to_coco_ids():
    items = module.serialize_detections(result(
        detection(index=None, name="car"), detection(index=None, name="truck"),
        detection(index=None, name="traffic light"), detection(index=None, name="stop sign")), 100, 100, labels())
    assert [item["class_id"] for item in items] == [3, 8, 10, 13]
    assert items[0]["box"] == [0.1, 0.2, 0.3, 0.4]


def test_metadata_index_fallback_and_filters_and_clipping():
    items = module.serialize_detections(result(
        detection(index=0, name=None, box=(-10, -20, 30, 40)),
        detection(index=2, name="unknown"), detection(score=0.29),
        detection(score=float("nan")), detection(box=(150, 150, 20, 20)),
        detection(index=4, name="airplane")), 100, 100, labels())
    assert len(items) == 1
    assert items[0]["class_id"] == 1 and items[0]["box"] == [0.0, 0.0, 0.2, 0.2]


def test_fixed_tiles_cover_both_edges_and_stay_in_bounds():
    windows = module.tile_windows(1920, 1080)
    assert len(windows) == 8
    assert sorted({window[0] for window in windows}) == [0, 512, 1024, 1280]
    assert sorted({window[1] for window in windows}) == [0, 440]
    assert all(0 <= x < right <= 1920 and 0 <= y < bottom <= 1080 for x, y, right, bottom in windows)
    assert module.tile_windows(500, 600) == [(0, 0, 500, 600)]


def test_tile_box_mapping_preserves_full_image_coordinates():
    item = {"class_id": 3, "class_name": "car", "score": .95, "box": [.25, .5, .5, .25]}
    mapped = module.remap_tile_objects([item], (512, 440, 1152, 1080), 1920, 1080)[0]
    assert mapped["box"] == pytest.approx([672 / 1920, 760 / 1080, 320 / 1920, 160 / 1080])
    assert item["box"] == [.25, .5, .5, .25]


def test_same_class_nms_removes_duplicate_tiles_but_keeps_distinct_categories():
    one = {"class_id": 3, "class_name": "car", "score": .95, "box": [.1, .1, .2, .2]}
    duplicate = {**one, "score": .8, "box": [.11, .11, .2, .2]}
    other_class = {**one, "class_id": 8, "class_name": "truck", "score": .9}
    distant = {**one, "score": .7, "box": [.7, .7, .2, .2]}
    merged = module.merge_detections([duplicate, one, other_class, distant])
    assert merged == [one, other_class, distant]


def test_runtime_loads_once_and_handles_rgb_images(fake_loaded):
    detector, calls = fake_loaded
    one = detector.predict(Image.new("RGBA", (640, 480), (120, 20, 30, 200)))
    two = detector.predict(Image.new("RGB", (640, 480)))
    assert calls["loads"] == 1
    assert len(calls["images"]) == 2
    assert calls["images"][0].shape == (480, 640, 3)
    assert calls["images"][0].dtype == np.uint8
    assert one["objects"][0]["class_id"] == 3
    assert one["counts"] == {"car": 1}
    assert one["image_size"] == [640, 480] and not one["tiled"]
    assert two["inference_passes"] == 1
    assert calls["options"].score_threshold == .3
    assert not one["scores_calibrated"] and not one["target_met"]


def test_high_resolution_uses_fixed_resize_tiles_and_preserves_output_size(fake_loaded):
    detector, calls = fake_loaded
    output = detector.predict(Image.new("RGB", (3840, 2160)))
    assert output["image_size"] == [3840, 2160]
    assert output["inference_size"] == [1920, 1080]
    assert output["inference_passes"] == 9 and output["tiled"]
    assert len(calls["images"]) == 9
    assert calls["images"][0].shape[:2] == (1080, 1920)
    assert all(image.shape[:2] == (640, 640) for image in calls["images"][1:])
    assert all(0 <= value <= 1 for obj in output["objects"] for value in obj["box"])


def test_evaluation_requires_exact_weights_and_inference_contract(fake_loaded):
    detector, _ = fake_loaded
    report = {"model_sha256": module.MODEL_SHA256, "model_version": module.MODEL_VERSION,
              "score_threshold": .3, "images": 500, "precision": .95, "recall": .94, "target_met": True}
    path = detector.model_dir / "evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert detector.status()["target_met"]
    for change in [{"model_sha256": "wrong"}, {"model_version": "previous-contract"},
                   {"score_threshold": .9}, {"runtime_version": "different"}]:
        path.write_text(json.dumps({**report, **change}), encoding="utf-8")
        status = detector.status()
        assert status["evaluation"] is None and not status["target_met"]
        assert "not accepted" in status["reason"]
    for change in [{"precision": .89}, {"recall": .89}, {"images": 4}, {"target_met": False}]:
        path.write_text(json.dumps({**report, **change}), encoding="utf-8")
        assert not detector.status()["target_met"]


def test_no_probability_or_street_safety_claim_from_pretrained_weights(fake_loaded):
    detector, _ = fake_loaded
    status = detector.status()
    assert not status["scores_calibrated"]
    assert status["evaluation"] is None and not status["target_met"]
    assert not status["road_safety_validated"]
    assert "SDK usage metrics" in status["telemetry_notice"]
    assert "not sent" in status["telemetry_notice"]
