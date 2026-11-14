"""Install a frozen pretrained image encoder without a dataset or anomaly head."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation" / "server"))
from street_vision import PREPROCESSING, file_sha256


def metadata_for(weights: Path) -> dict:
    digest = file_sha256(weights)
    # This exact version contract is shared with the original 224px experiment.
    version = hashlib.sha256(json.dumps({"architecture": "MobileNetV3Small", "weights_sha256": digest,
                 "preprocessing": PREPROCESSING, "pooling": "avg"}, sort_keys=True).encode()).hexdigest()
    return {"embedding_version": version, "weights_sha256": digest, "embedding_dimensions": 576,
            "preprocessing": PREPROCESSING}


def setup(model_dir: Path) -> dict:
    directory = model_dir.resolve()
    weights = directory / "backbone.weights.h5"
    metadata_path = directory / "backbone.json"
    manifest_path = directory / "manifest.json"
    if weights.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata_for(weights)
        if any(existing.get(key) != value for key, value in expected.items()) or existing.get("pooling", "avg") != "avg" or existing.get("input_size", [224, 224]) != [224, 224]:
            raise ValueError("Existing encoder has a different configuration/checksum; refusing to change it.")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("embedding_version") != existing["embedding_version"]:
                raise ValueError("Evaluated model and encoder versions differ; refusing to modify either.")
        return {"available": True, "already_present": True, **existing,
                "anomaly_head_trained": manifest_path.exists() and (directory / "parameters.json").exists()}
    if weights.exists() != metadata_path.exists():
        raise ValueError("Existing encoder files are incomplete; restore the original pair or choose an empty directory.")
    if manifest_path.exists() or (directory / "parameters.json").exists():
        raise ValueError("An evaluated model exists but encoder files are incomplete; refusing to overwrite its contract.")
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_HOME", str(directory / "keras_cache"))
    import tensorflow as tf
    backbone = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False,
        weights=None if weights.exists() else "imagenet", pooling="avg", include_preprocessing=True)
    if weights.exists():
        backbone.load_weights(str(weights))
    else:
        temporary = directory / "backbone.setup.weights.h5"
        backbone.save_weights(str(temporary))
        temporary.replace(weights)
    expected = metadata_for(weights)
    temporary_metadata = directory / "backbone.json.tmp"
    temporary_metadata.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    # Deliberately no anomaly parameters, calibration, or evaluation manifest.
    return {"available": True, "already_present": False, **expected, "anomaly_head_trained": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".runtime" / "street_vision")
    args = parser.parse_args()
    print(json.dumps(setup(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
