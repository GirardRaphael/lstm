"""Separate supervised anomaly network; never replaces action-model embeddings."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from street_vision import file_sha256, prepare_image, sigmoid

INPUT_SIZE = (320, 320)
STEM_LAYER = "expanded_conv_8_project_bn"


def build_models(tf, initial_weights: Path | None = None):
    """Explicit architecture, safe weights-only loading, no serialized custom code."""
    backbone = tf.keras.applications.MobileNetV3Small(input_shape=(*INPUT_SIZE, 3), include_top=False, weights=None, pooling=None, include_preprocessing=True)
    if initial_weights is not None:
        backbone.load_weights(str(initial_weights))
    stem_output = backbone.get_layer(STEM_LAYER).output
    stem = tf.keras.Model(backbone.input, stem_output, name="frozen_anomaly_stem")
    average = tf.keras.layers.GlobalAveragePooling2D(name="anomaly_average")(backbone.output)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="anomaly_maximum")(backbone.output)
    pooled = tf.keras.layers.Concatenate(name="anomaly_pooling")([average, maximum])
    normalized = tf.keras.layers.LayerNormalization(name="anomaly_normalization")(pooled)
    hidden = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001), name="anomaly_hidden")(normalized)
    dropped = tf.keras.layers.Dropout(.3, name="anomaly_dropout")(hidden)
    output = tf.keras.layers.Dense(1, name="anomaly_logit")(dropped)
    full = tf.keras.Model(backbone.input, output, name="signal_anomaly_320")
    cached = tf.keras.Model(stem_output, output, name="anomaly_trainable_tail")
    return backbone, stem, cached, full


def set_backbone_training(tf, backbone, *, fine_tune: bool):
    boundary = next(index for index, layer in enumerate(backbone.layers) if layer.name == STEM_LAYER)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = bool(fine_tune and index > boundary and not isinstance(layer, tf.keras.layers.BatchNormalization))


class FineTunedAnomaly:
    def __init__(self, model_dir: str | Path):
        self.directory = Path(model_dir)
        self._model = None
        self._lock = Lock()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf8"))
        self.parameters = json.loads((self.directory / "calibration.json").read_text(encoding="utf8"))

    def status(self) -> dict:
        return {"available": (self.directory / "anomaly.weights.h5").is_file(), "model_version": self.manifest["model_version"],
                "architecture": self.manifest["architecture"], "metrics": self.manifest["test_metrics"],
                "review_candidate_accepted": self.manifest["review_candidate_accepted"],
                "validation_result": self.manifest["validation_result"], "evaluation_reused": True,
                "signal_control_validated": False, "deployment_ready": False, "separate_from_action_encoder": True}

    def predict(self, image: Image.Image) -> dict:
        with self._lock:
            if self._model is None:
                for name, expected in self.manifest["artifact_sha256"].items():
                    if file_sha256(self.directory / name) != expected:
                        raise RuntimeError("Fine-tuned anomaly artifact checksum mismatch: " + name)
                import tensorflow as tf
                _, _, _, candidate = build_models(tf)
                candidate.load_weights(str(self.directory / "anomaly.weights.h5"))
                self._model = candidate
            pixels = prepare_image(image, INPUT_SIZE)[None]
            logit = float(np.asarray(self._model(pixels, training=False)).reshape(-1)[0])
        probability = float(sigmoid(logit * self.parameters["coef"] + self.parameters["intercept"]))
        return {"anomaly_probability": probability, "normal_probability": 1 - probability,
                "threshold": self.parameters["threshold"], "decision": "review" if probability >= self.parameters["threshold"] else "normal",
                "calibrated": True, "model_version": self.manifest["model_version"], "task": "binary aerial-frame anomaly review",
                "reason": "A separately fine-tuned visual network estimates publisher-labelled aerial-frame anomalies; inspect the frame before interpreting the score.",
                "uncertainty_note": "Calibration is fitted on limited held-out recording groups. This is not validated traffic-light or violation recognition.",
                "evaluation_reused": True, "deployment_ready": False, "signal_control_validated": False,
                "review_candidate_accepted": self.manifest["review_candidate_accepted"], "separate_from_action_encoder": True}
