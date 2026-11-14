"""Official COCO-pretrained EfficientDet-Lite2 through MediaPipe Tasks.

Google's task runtime reads the bundled metadata and performs preprocessing,
anchor decoding and non-maximum suppression. Scores are model scores, not
calibrated probabilities or demonstrated reliability on street footage.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "EfficientDet-Lite2"
MODEL_FILENAME = "efficientdet_lite2.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite"
MODEL_SHA256 = "ad2abbf2b4e10585e15176fd7b5ef03c28dda959ae26fc142549fdd1814db91d"
LABELMAP_SHA256 = "f8803ef7900160c629d570848dfda4175e21667bf7b71f73f8ece4938c9f2bf2"
RUNTIME_VERSION = "0.10.35"
RUNTIME_PACKAGES = ("mediapipe==0.10.35", "opencv-contrib-python==4.13.0.92", "sounddevice==0.5.2")
SCORE_THRESHOLD = 0.30
TARGET = 0.90
TELEMETRY_NOTICE = "MediaPipe may report SDK usage metrics; Google states input images are not sent."
PRIVACY_SOURCE = "https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice"
ROAD_CLASSES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
                6: "bus", 8: "truck", 10: "traffic light", 13: "stop sign"}
MAX_PIXELS = 24_000_000
MAX_INPUT_EDGE = 1920
TILING_TRIGGER_EDGE = 960
TILE_SIZE = 640
TILE_STRIDE = 512
MERGE_IOU = 0.5
INFERENCE_CONTRACT = {
    "model_sha256": MODEL_SHA256, "runtime": "mediapipe.tasks.vision.ObjectDetector",
    "runtime_version": RUNTIME_VERSION, "running_mode": "IMAGE",
    "score_threshold": SCORE_THRESHOLD, "max_results": -1,
    "road_classes": ROAD_CLASSES, "image_preparation": "PIL-exif-transpose-RGB-uint8",
    "max_input_edge": MAX_INPUT_EDGE, "resize": "PIL-LANCZOS-keep-aspect",
    "tiling_trigger_edge": TILING_TRIGGER_EDGE, "tile_size": TILE_SIZE,
    "tile_stride": TILE_STRIDE, "tiles": "global-plus-xy-cover-end-inclusive-v1",
    "merge_nms": "classwise-greedy-score-IoU-strictly-greater-v1", "merge_iou": MERGE_IOU,
    "preprocessing_decode_nms": "official-model-metadata",
    "box_serialization": "clip-original-pixel-xywh-to-image-then-normalize-v1",
    "category_mapping": "verified-sparse-coco-label-name-first-index-fallback-v2",
}
INFERENCE_CONTRACT_SHA256 = hashlib.sha256(json.dumps(INFERENCE_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
MODEL_VERSION = "efficientdet-lite2-fp32-v1-" + MODEL_SHA256[:12] + "-" + INFERENCE_CONTRACT_SHA256[:12]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_runtime():
    local_dependencies = Path(__file__).resolve().parents[2] / ".runtime/detector-deps"
    if local_dependencies.is_dir() and str(local_dependencies) not in sys.path:
        sys.path.insert(0, str(local_dependencies))
    runtime = importlib.import_module("mediapipe")
    if runtime.__version__ != RUNTIME_VERSION:
        raise RuntimeError(f"Expected MediaPipe {RUNTIME_VERSION}; found {runtime.__version__}. Run scripts/setup_pretrained_detector.py --install-runtime.")
    return runtime


def serialize_detections(result: Any, width: int, height: int, labels: list[str]) -> list[dict]:
    """Map Google's zero-based sparse COCO metadata indices to official IDs."""
    objects = []
    for detection in result.detections:
        for category in detection.categories:
            # MediaPipe 0.10.35 can supply a valid metadata name with index=None.
            # Resolve that name through the verified sparse COCO label map.
            name = getattr(category, "category_name", None)
            if name is not None:
                if name not in ROAD_CLASSES.values() or name not in labels:
                    continue
                class_id = labels.index(name) + 1
            else:
                index = category.index
                if not isinstance(index, int) or index < 0 or index >= len(labels):
                    continue
                class_id, name = index + 1, labels[index]
            if ROAD_CLASSES.get(class_id) != name:
                continue
            if not isinstance(category.score, (int, float)):
                continue
            score = float(category.score)
            if not math.isfinite(score) or not SCORE_THRESHOLD <= score <= 1:
                continue
            bounds = detection.bounding_box
            coordinates = [float(bounds.origin_x), float(bounds.origin_y), float(bounds.width), float(bounds.height)]
            if not all(math.isfinite(value) for value in coordinates):
                continue
            x, y, box_width, box_height = coordinates
            left, top = max(0.0, x), max(0.0, y)
            right, bottom = min(float(width), x + box_width), min(float(height), y + box_height)
            if right <= left or bottom <= top:
                continue
            objects.append({"class_id": class_id, "class_name": name, "score": score,
                            "box": [left / width, top / height, (right - left) / width, (bottom - top) / height]})
    return sorted(objects, key=lambda item: item["score"], reverse=True)


def tile_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Full coverage with fixed overlap; final edge positions are never omitted."""
    def positions(length: int) -> list[int]:
        last = max(0, length - TILE_SIZE)
        values = list(range(0, last + 1, TILE_STRIDE))
        if values[-1] != last:
            values.append(last)
        return values
    return [(x, y, min(x + TILE_SIZE, width), min(y + TILE_SIZE, height))
            for y in positions(height) for x in positions(width)]


def remap_tile_objects(objects: list[dict], window: tuple[int, int, int, int],
                       width: int, height: int) -> list[dict]:
    left, top, right, bottom = window
    tile_width, tile_height = right - left, bottom - top
    remapped = []
    for item in objects:
        x, y, box_width, box_height = item["box"]
        remapped.append({**item, "box": [(left + x * tile_width) / width,
                                        (top + y * tile_height) / height,
                                        box_width * tile_width / width,
                                        box_height * tile_height / height]})
    return remapped


def box_iou(one: list[float], two: list[float]) -> float:
    left, top = max(one[0], two[0]), max(one[1], two[1])
    right, bottom = min(one[0] + one[2], two[0] + two[2]), min(one[1] + one[3], two[1] + two[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = one[2] * one[3] + two[2] * two[3] - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(objects: list[dict]) -> list[dict]:
    """Merge global/tile duplicates without suppressing different categories."""
    retained: list[dict] = []
    for candidate in sorted(objects, key=lambda item: item["score"], reverse=True):
        if not any(previous["class_id"] == candidate["class_id"]
                   and box_iou(previous["box"], candidate["box"]) > MERGE_IOU for previous in retained):
            retained.append(candidate)
    return retained


class PretrainedDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        self._lock = threading.RLock()
        self._engine = None
        self._runtime = None
        self._labels: list[str] = []
        self._error: str | None = None

    def _load(self) -> None:
        if self._engine is not None:
            return
        model = self.model_dir / MODEL_FILENAME
        labelmap = self.model_dir / "labelmap.txt"
        if not model.is_file() or not labelmap.is_file():
            raise FileNotFoundError("Pretrained detector assets are missing. Run scripts/setup_pretrained_detector.py --install-runtime.")
        if sha256_file(model) != MODEL_SHA256 or sha256_file(labelmap) != LABELMAP_SHA256:
            raise RuntimeError("The pretrained model or label map failed its SHA256 integrity check. Run setup_pretrained_detector.py to restore the official assets.")
        labels = labelmap.read_text(encoding="utf-8").splitlines()
        if any(labels[index - 1] != name for index, name in ROAD_CLASSES.items()):
            raise RuntimeError("The model label map does not match the expected COCO category IDs.")
        runtime = _import_runtime()
        options = runtime.tasks.vision.ObjectDetectorOptions(
            base_options=runtime.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=runtime.tasks.vision.RunningMode.IMAGE,
            score_threshold=SCORE_THRESHOLD, max_results=-1,
            category_allowlist=list(ROAD_CLASSES.values()))
        engine = runtime.tasks.vision.ObjectDetector.create_from_options(options)
        self._runtime, self._engine, self._labels = runtime, engine, labels
        self._error = None

    def _evaluation(self) -> tuple[dict | None, str | None]:
        path = self.model_dir / "evaluation.json"
        if not path.is_file():
            return None, None
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                raise ValueError("evaluation file exceeds the supported size")
            def reject_constant(value):
                raise ValueError(f"non-finite value {value}")
            report = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
            if not isinstance(report, dict) or report.get("model_sha256") != MODEL_SHA256:
                raise ValueError("evaluation belongs to another model")
            if report.get("model_version") != MODEL_VERSION:
                raise ValueError("evaluation uses another inference contract version")
            if report.get("score_threshold") != SCORE_THRESHOLD:
                raise ValueError("evaluation uses another detection threshold")
            if "runtime_version" in report and report["runtime_version"] != RUNTIME_VERSION:
                raise ValueError("evaluation uses another MediaPipe runtime")
            return report, None
        except (OSError, ValueError, TypeError) as exc:
            return None, f"Evaluation was not accepted: {exc}."

    def _base_status(self, available: bool) -> dict:
        evaluation, evaluation_error = self._evaluation()
        precision = evaluation.get("precision") if evaluation else None
        recall = evaluation.get("recall") if evaluation else None
        images = evaluation.get("images") if evaluation else None
        measured_target = bool(
            available and evaluation and evaluation.get("target_met") is True
            and isinstance(precision, (int, float)) and TARGET <= precision <= 1
            and isinstance(recall, (int, float)) and TARGET <= recall <= 1
            and isinstance(images, int) and images >= 100)
        reason = self._error if not available else (
            "Official COCO-pretrained detector. Its scores are uncalibrated; review the separate evaluation before interpreting reliability.")
        if evaluation_error:
            reason = (reason or "") + " " + evaluation_error
        return {"available": available, "model": MODEL_NAME, "model_version": MODEL_VERSION,
                "model_sha256": MODEL_SHA256, "runtime_version": RUNTIME_VERSION,
                "inference_contract_sha256": INFERENCE_CONTRACT_SHA256,
                "pretrained_on": "COCO", "score_threshold": SCORE_THRESHOLD,
                "scores_calibrated": False, "target": TARGET, "target_met": measured_target,
                "evaluation": evaluation, "reason": reason, "road_safety_validated": False,
                "class_ids": list(ROAD_CLASSES), "source_url": MODEL_URL,
                "telemetry_notice": TELEMETRY_NOTICE, "privacy_source": PRIVACY_SOURCE}

    def status(self) -> dict:
        with self._lock:
            try:
                self._load()
            except (OSError, ImportError, RuntimeError, ValueError) as exc:
                self._error = str(exc)
            return self._base_status(self._engine is not None)

    def predict(self, image: Image.Image) -> dict:
        if not isinstance(image, Image.Image):
            raise ValueError("The detector requires a decoded PIL image.")
        if min(image.size) <= 0 or image.width * image.height > MAX_PIXELS:
            raise ValueError("The detector accepts images up to 24 megapixels.")
        with self._lock:
            status = self.status()
            if not status["available"]:
                return {**status, "objects": [], "counts": {}, "latency_ms": None,
                        "image_size": list(image.size)}
            started = time.perf_counter()
            upright = ImageOps.exif_transpose(image).convert("RGB")
            original_size = list(upright.size)
            upright.thumbnail((MAX_INPUT_EDGE, MAX_INPUT_EDGE), Image.Resampling.LANCZOS)
            def detect_view(view: Image.Image) -> list[dict]:
                pixels = np.ascontiguousarray(np.asarray(view), dtype=np.uint8)
                mp_image = self._runtime.Image(image_format=self._runtime.ImageFormat.SRGB, data=pixels)
                result = self._engine.detect(mp_image)
                return serialize_detections(result, view.width, view.height, self._labels)
            objects = detect_view(upright)
            windows = tile_windows(upright.width, upright.height) if max(upright.size) > TILING_TRIGGER_EDGE else []
            for window in windows:
                tile_objects = detect_view(upright.crop(window))
                objects.extend(remap_tile_objects(tile_objects, window, upright.width, upright.height))
            objects = merge_detections(objects)
            return {**status, "objects": objects,
                    "counts": dict(Counter(item["class_name"] for item in objects)),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "image_size": original_size, "inference_size": list(upright.size),
                    "inference_passes": 1 + len(windows), "tiled": bool(windows)}

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
                self._runtime = None
