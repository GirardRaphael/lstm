"""Fetch verified Google detector assets and optionally install a local runtime.

Usage (from the project root):
    python scripts/setup_pretrained_detector.py --install-runtime

The runtime is pinned under .runtime/detector-deps; --no-deps prevents replacing
NumPy/TensorFlow from the project's Python environment. Install the project's
normal Python requirements first. Google MediaPipe Tasks performs preprocessing,
decoding and NMS; this project does not guess the model's raw output encoding.
Official documentation:
https://developers.google.com/edge/mediapipe/solutions/vision/object_detector
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation/server"))
from pretrained_detector import (MODEL_FILENAME, MODEL_URL, MODEL_SHA256,
    LABELMAP_SHA256, RUNTIME_VERSION, RUNTIME_PACKAGES, MODEL_VERSION,
    INFERENCE_CONTRACT_SHA256, sha256_file)


def setup(model_dir: Path, install_runtime: bool = False) -> dict:
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / MODEL_FILENAME
    if not model.is_file() or sha256_file(model) != MODEL_SHA256:
        temporary = model_dir / (MODEL_FILENAME + "." + uuid4().hex + ".part")
        try:
            with urlopen(MODEL_URL, timeout=120) as response, temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
                    if output.tell() > 32 * 1024 * 1024:
                        raise RuntimeError("Detector download exceeded its expected size.")
            if sha256_file(temporary) != MODEL_SHA256:
                raise RuntimeError("Detector download did not match the pinned official SHA256.")
            temporary.replace(model)
        finally:
            if temporary.exists():
                temporary.unlink()
    with zipfile.ZipFile(model) as archive:
        labels = archive.read("labels.txt")
    if hashlib.sha256(labels).hexdigest() != LABELMAP_SHA256:
        raise RuntimeError("Embedded official labels did not match their pinned SHA256.")
    (model_dir / "labelmap.txt").write_bytes(labels)
    runtime_dir = ROOT / ".runtime/detector-deps"
    if install_runtime:
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(runtime_dir),
                        "--upgrade", "--no-deps", "--disable-pip-version-check", *RUNTIME_PACKAGES], check=True)
    manifest = {"model": "EfficientDet-Lite2", "model_sha256": MODEL_SHA256,
                "model_version": MODEL_VERSION, "inference_contract_sha256": INFERENCE_CONTRACT_SHA256,
                "labelmap_sha256": LABELMAP_SHA256, "source_url": MODEL_URL,
                "pretrained_on": "COCO", "runtime_version": RUNTIME_VERSION,
                "runtime_packages": list(RUNTIME_PACKAGES), "installed_at": datetime.now(timezone.utc).isoformat(),
                "scores_calibrated": False, "target": 0.90,
                "validation_note": "Downloading pretrained weights does not validate 90% reliability or traffic-light decisions."}
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"ok": True, "model_dir": str(model_dir), "runtime_dir": str(runtime_dir), **manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".runtime/pretrained_detector")
    parser.add_argument("--install-runtime", action="store_true")
    args = parser.parse_args()
    print(json.dumps(setup(args.model_dir, args.install_runtime), indent=2))


if __name__ == "__main__":
    main()
