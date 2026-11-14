"""Frozen visual features with a calibrated, trained binary anomaly classifier.

This recognizes publisher-labelled aerial-frame anomalies; it is not an object
detector, a calibrated signal policy, or evidence of safe real-road control.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

INPUT_SIZE = (224, 224)
PREPROCESSING = "PIL RGB; bilinear resize 224x224; float32 0..255; MobileNetV3 built-in rescaling"


def prepare_image(image: Image.Image, size=INPUT_SIZE) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR), dtype=np.float32)


def pool_features(output: np.ndarray, mode: str = "avg") -> np.ndarray:
    output = np.asarray(output)
    if mode == "spatial_mean_max_std":
        return np.concatenate([output.mean(axis=(1, 2)), output.max(axis=(1, 2)), output.std(axis=(1, 2))], axis=1)
    return output


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    # Clipping avoids overflow without affecting meaningful probabilities.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700, 700)))


def score_embeddings(embeddings: np.ndarray, parameters: dict) -> np.ndarray:
    features = np.asarray(embeddings, dtype=np.float64)
    scaled = (features - np.asarray(parameters["scaler_mean"])) / np.asarray(parameters["scaler_scale"])
    raw = scaled @ np.asarray(parameters["classifier_coef"]) + float(parameters["classifier_intercept"])
    return sigmoid(raw * float(parameters["calibrator_coef"]) + float(parameters["calibrator_intercept"]))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StreetVision:
    def __init__(self, model_dir: str | Path, data_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self._backbone = None
        self._lock = Lock()
        self._parameters = None
        self._model_verified = False
        self._manifest = None
        self._sample_rows = None
        self._cache = None
        self._input_hashes = None
        self._backbone_metadata = {}
        metadata = self.model_dir / "backbone.json"
        if metadata.exists():
            self._backbone_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        manifest = self.model_dir / "manifest.json"
        parameters = self.model_dir / "parameters.json"
        if manifest.exists() and parameters.exists():
            self._manifest = json.loads(manifest.read_text(encoding="utf-8"))
            self._parameters = json.loads(parameters.read_text(encoding="utf-8"))

    def status(self) -> dict:
        # A server may be opened while the first training job is still running.
        if self._manifest is None and (self.model_dir / "manifest.json").exists():
            self._manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
            self._parameters = json.loads((self.model_dir / "parameters.json").read_text(encoding="utf-8"))
        if not self._backbone_metadata and (self.model_dir / "backbone.json").exists():
            self._backbone_metadata = json.loads((self.model_dir / "backbone.json").read_text(encoding="utf-8"))
        ready = self._manifest is not None and self._parameters is not None and (self.model_dir / "backbone.weights.h5").exists()
        return {
            "available": ready,
            "embedding_available": (self.model_dir / "backbone.weights.h5").exists(),
            "embedding_version": self._backbone_metadata.get("embedding_version"),
            "embedding_dimensions": self._backbone_metadata.get("embedding_dimensions"),
            "model_version": self._manifest.get("model_version") if self._manifest else None,
            "architecture": "ImageNet MobileNetV3Small frozen features + regularized logistic regression + validation sigmoid calibration",
            "task": "binary anomaly classification of real aerial street frames",
            "calibrated": ready,
            "calibration_method": "Sigmoid fitted on validation scores only",
            "validation_result": self._manifest.get("validation_result") if ready else "not evaluated",
            "validation_scope": "Held-out UIT-ADrone source recordings; not an object detector or signal-policy validation",
            "metrics": self._manifest.get("test_metrics") if ready else None,
            "split_counts": self._manifest.get("split_counts") if ready else None,
            "limitations": self._manifest.get("limitations", []) if ready else ["The image encoder can support decision teaching; no anomaly head has been trained yet."],
            "signal_control_validated": False,
            "deployment_ready": False,
            "evaluation_reused": bool(self._manifest.get("evaluation_reused", False)) if self._manifest else False,
        }

    def _require_model(self):
        if not self.status()["available"]:
            raise RuntimeError("Street vision model has not been trained. Run scripts/train_street_vision.py first.")
        if not self._model_verified:
            expected = self._manifest["artifact_sha256"]["parameters.json"]
            if file_sha256(self.model_dir / "parameters.json") != expected:
                raise RuntimeError("Classifier parameters checksum mismatch")
            if self._backbone_metadata and self._parameters["embedding_version"] != self._backbone_metadata["embedding_version"]:
                raise RuntimeError("Classifier and image backbone embedding versions differ")
            self._model_verified = True

    def embed(self, image: Image.Image) -> np.ndarray:
        self.status()  # Refresh metadata if training completed after server startup.
        if not (self.model_dir / "backbone.weights.h5").exists():
            raise RuntimeError("Pretrained image backbone is not available yet.")
        with self._lock:
            if self._backbone is None:
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                import tensorflow as tf
                size = self._backbone_metadata.get("input_size", [224, 224])
                pooling = self._backbone_metadata.get("pooling", "avg")
                backbone = tf.keras.applications.MobileNetV3Small(
                    input_shape=(*size, 3), include_top=False, weights=None,
                    pooling=None if pooling == "spatial_mean_max_std" else "avg", include_preprocessing=True,
                )
                if self._backbone_metadata and file_sha256(self.model_dir / "backbone.weights.h5") != self._backbone_metadata["weights_sha256"]:
                    raise RuntimeError("Image backbone checksum does not match its declared embedding version")
                backbone.load_weights(str(self.model_dir / "backbone.weights.h5"))
                backbone.trainable = False
                self._backbone = backbone
            pixels = prepare_image(image, self._backbone_metadata.get("input_size", [224, 224]))[None, ...]
            output = np.asarray(self._backbone(pixels, training=False))
            return pool_features(output, self._backbone_metadata.get("pooling", "avg"))[0]

    def _prediction(self, embedding: np.ndarray) -> dict:
        self._require_model()
        probability = float(score_embeddings(np.asarray(embedding)[None, :], self._parameters)[0])
        return {
            "anomaly_probability": probability,
            "normal_probability": 1.0 - probability,
            "calibrated": True,
            "decision": "review" if probability >= float(self._parameters["threshold"]) else "normal",
            "threshold": float(self._parameters["threshold"]),
            "model_version": self._manifest["model_version"],
            "task": "binary aerial-frame anomaly classification",
            "signal_control_validated": False,
            "deployment_ready": False,
            "evaluation_reused": bool(self._manifest.get("evaluation_reused", False)) if self._manifest else False,
            "reason": "A trained binary image classifier estimates whether this aerial frame warrants anomaly review.",
            "uncertainty_note": "Estimated probability calibrated on a small validation split; it is not confidence that a traffic-light action is correct.",
        }

    def predict_embedding(self, embedding: np.ndarray, *, embedding_version: str | None = None) -> dict:
        self._require_model()
        if embedding_version is not None and embedding_version != self._parameters["embedding_version"]:
            raise ValueError("Image embedding version does not match the anomaly classifier")
        values = np.asarray(embedding, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(self._parameters["classifier_coef"]) or not np.isfinite(values).all():
            raise ValueError("Expected a finite vector from this model's image encoder")
        return self._prediction(values)

    def predict(self, image: Image.Image) -> dict:
        return self.predict_embedding(self.embed(image))

    def _rows(self):
        if self._sample_rows is None:
            with (self.data_dir / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
                self._sample_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
        return self._sample_rows

    def list_samples(self, split: str = "test", limit: int = 50) -> list[dict]:
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown data split")
        rows = [r for r in self._rows().values() if r["split"] == split]
        # Labels are exposed for reviewing a sample, never as inference inputs.
        return [{"sample_id": r["sample_id"], "split": r["split"], "label": int(r["label"]),
                 "class_name": r["class_name"], "source_video": r["source_video"],
                 "label_origin": r["label_origin"], "review_status": r["review_status"]}
                for r in rows[:max(0, min(int(limit), 1500))]]

    def sample_path(self, sample_id: str) -> Path:
        row = self._rows().get(sample_id)
        if row is None:
            raise ValueError("Unknown sample ID")
        path = (self.data_dir / row["image_path"]).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise ValueError("Sample path must remain within the data directory")
        return path

    def predict_sample(self, sample_id: str) -> dict:
        self._require_model()
        path = self.sample_path(sample_id)  # Validate against the registered dataset.
        if self._input_hashes is None:
            if file_sha256(self.model_dir / "input_hashes.json") != self._manifest["artifact_sha256"]["input_hashes.json"]:
                raise RuntimeError("Input image hash manifest checksum mismatch")
            records = json.loads((self.model_dir / "input_hashes.json").read_text(encoding="utf-8"))["images"]
            self._input_hashes = {record["sample_id"]: record["sha256"] for record in records}
        unchanged = self._input_hashes.get(sample_id) == file_sha256(path)
        if self._cache is None:
            expected = self._manifest["artifact_sha256"]["embeddings.npz"]
            if file_sha256(self.model_dir / "embeddings.npz") != expected:
                raise RuntimeError("Image embedding cache checksum mismatch")
            with np.load(self.model_dir / "embeddings.npz", allow_pickle=False) as cache:
                self._cache = {str(key): feature for key, feature in zip(cache["sample_ids"], cache["embeddings"])}
        if unchanged and sample_id in self._cache:
            result = self._prediction(self._cache[sample_id])
            result["input_source"] = "cached image embedding"
        else:
            with Image.open(self.sample_path(sample_id)) as image:
                result = self.predict(image)
            result["input_source"] = "image pixels"
        return {**result, "sample_id": sample_id}
