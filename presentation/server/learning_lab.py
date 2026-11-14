"""Local review API: immutable image snapshots, real-image inference and human teaching.

Only human feedback creates signal-policy labels. Publisher anomaly labels never
become light-action labels. Images and model files stay on the local computer.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
from datetime import datetime, timezone
from collections import Counter
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000


class LabError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def clean_id(value: object, name: str, limit: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise LabError(f'{name} must be a non-empty string of at most {limit} characters.')
    if any(ord(c) < 32 for c in value):
        raise LabError(f'{name} contains invalid characters.')
    return value.strip()


def decode_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise LabError('Choose an image smaller than 10 MB.')
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ('PNG', 'JPEG', 'WEBP'):
                raise LabError('Choose a PNG, JPEG or WebP image.')
            if source.width * source.height > MAX_PIXELS or min(source.size) < 32:
                raise LabError('Image dimensions must be at least 32 pixels and at most 24 megapixels.')
            source.load()
            result = ImageOps.exif_transpose(source).convert('RGB')
            result.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
            return result
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise LabError('The image could not be decoded.') from exc


ROAD_CLASS_IDS = (1, 2, 3, 4, 6, 8, 10, 13)

def detection_features(embedding: np.ndarray, version: str, detection: dict) -> tuple[np.ndarray, str]:
    """Add observed object locations, never simulator state or publisher labels."""
    grid = np.zeros((len(ROAD_CLASS_IDS), 3, 3), dtype=float)
    for obj in detection.get('objects', []):
        class_id = obj.get('class_id')
        if class_id not in ROAD_CLASS_IDS:
            continue
        x, y, width, height = obj['box']
        column = min(2, max(0, int((x + width / 2) * 3)))
        row = min(2, max(0, int((y + height / 2) * 3)))
        grid[ROAD_CLASS_IDS.index(class_id), row, column] += 1
    model_version = clean_id(detection.get('model_version'), 'Detector model version', 120)
    combined = hashlib.sha256((version + ':' + model_version + ':road-grid3x3-counts-log1p-v1').encode()).hexdigest()
    return np.concatenate([embedding, np.log1p(grid).reshape(-1)]), 'street-perception-' + combined


class LearningLab:
    def __init__(self, root: Path, *, vision=None, learner=None, detector=None, data_dir: Path | None = None):
        self.root = Path(root).resolve()
        default_data = self.root / 'data/intersection_anomalies_expanded'
        if not (default_data / 'labels.csv').is_file():
            default_data = self.root / 'data/intersection_anomalies'
        self.data_dir = Path(data_dir or os.environ.get('TRAFFIC_IMAGE_DATASET', default_data)).resolve()
        default_model = self.root / '.runtime/street_vision_expanded'
        if not (default_model / 'manifest.json').is_file():
            default_model = self.root / 'models/street_vision'
        if not default_model.exists():
            default_model = self.root / '.runtime/street_vision'
        self.model_dir = Path(os.environ.get('TRAFFIC_STREET_MODEL_DIR', default_model)).resolve()
        self.storage = self.root / 'var/traffic_learning'
        self.snapshots = self.storage / 'snapshots'
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self._vision = vision
        self._detector = detector
        self._learner = learner
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._records: dict[str, dict] = {}
        labels = self.data_dir / 'labels.csv'
        if labels.is_file():
            with labels.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    if re.fullmatch(r'[a-f0-9]{16,64}', row.get('sample_id', '')):
                        self._records[row['sample_id']] = row

    @property
    def vision(self):
        with self._lock:
            if self._vision is None:
                from street_vision import StreetVision
                self._vision = StreetVision(self.model_dir, self.data_dir)
            return self._vision

    @property
    def detector(self):
        with self._lock:
            if self._detector is None:
                from pretrained_detector import PretrainedDetector
                directory = Path(os.environ.get("TRAFFIC_DETECTOR_MODEL_DIR", self.root / ".runtime/pretrained_detector"))
                self._detector = PretrainedDetector(directory)
            return self._detector

    @property
    def learner(self):
        with self._lock:
            if self._learner is None:
                from decision_learning import DecisionLearner
                self._learner = DecisionLearner(self.storage / 'policy')
            return self._learner

    def status(self) -> dict:
        try:
            vision = self.vision.status()
        except (OSError, ValueError, ImportError) as exc:
            vision = {'available': False, 'reason': str(exc)}
        try:
            detection = self.detector.status()
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            detection = {'available': False, 'reason': str(exc), 'target': .9, 'target_met': False}
        return {'ok': True, 'vision': vision, 'detection': detection,
                'policy': {domain: self.learner.status(domain) for domain in ('street', 'simulation')},
                'dataset': {'available': bool(self._records), 'total': len(self._records),
                            'counts': dict(Counter(r['split'] for r in self._records.values()))}}

    def samples(self, split: str = 'test', offset: int = 0, limit: int = 12) -> dict:
        if split not in ('train', 'validation', 'test') or offset < 0 or not 1 <= limit <= 48:
            raise LabError('Invalid sample page.')
        records = sorted((r for r in self._records.values() if r['split'] == split), key=lambda r: r['sample_id'])
        return {'ok': True, 'total': len(records), 'samples': [
            {**{k: row[k] for k in ('sample_id', 'source_video', 'split', 'label', 'class_name')},
             'image_url': f"/api/learning/image/{row['sample_id']}"}
            for row in records[offset:offset + limit]]}

    def sample_path(self, sample_id: str) -> Path:
        if not isinstance(sample_id, str) or not re.fullmatch(r'[a-f0-9]{16,64}', sample_id):
            raise LabError('Invalid image sample identifier.')
        row = self._records.get(sample_id)
        if not row:
            raise LabError('Image sample not found.', 404)
        path = (self.data_dir / row['image_path']).resolve()
        if not path.is_relative_to(self.data_dir / 'images') or not path.is_file():
            raise LabError('Image sample is unavailable.', 404)
        return path

    def _snapshot(self, snapshot_id: object) -> dict:
        if not isinstance(snapshot_id, str) or not re.fullmatch(r'[a-f0-9]{32}', snapshot_id):
            raise LabError('Invalid snapshot identifier.')
        path = self.snapshots / f'{snapshot_id}.json'
        if not path.is_file():
            raise LabError('Snapshot not found. Analyze the image again.', 404)
        return json.loads(path.read_text(encoding='utf-8'))

    def snapshot_image(self, snapshot_id: str) -> Path:
        self._snapshot(snapshot_id)
        return self.snapshots / f'{snapshot_id}.png'

    def analyze(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise LabError('Expected a JSON object.')
        sample_id = payload.get('sample_id')
        if 'sample_id' in payload:
            raw = self.sample_path(sample_id).read_bytes()
            domain = 'street'
            group_id = 'uit:' + self._records[sample_id]['source_video']
            teaching_allowed = False
        else:
            domain = payload.get('domain')
            if domain not in ('street', 'simulation'):
                raise LabError('Select street footage or a simulation screenshot.')
            group_id = clean_id(payload.get('group_id'), 'Recording/session ID')
            data_url = payload.get('image_data_url')
            if not isinstance(data_url, str) or len(data_url) > MAX_IMAGE_BYTES * 1.4:
                raise LabError('Choose an image smaller than 10 MB.')
            if not re.match(r'^data:image/(png|jpeg|webp);base64,', data_url):
                raise LabError('Expected a PNG, JPEG or WebP image.')
            try:
                raw = base64.b64decode(data_url.split(',', 1)[1], validate=True)
            except (ValueError, TypeError) as exc:
                raise LabError('Image encoding is invalid.') from exc
            teaching_allowed = True
        image = decode_image(raw)
        # Hash canonical pixels, so metadata/re-encoding cannot duplicate an example.
        image_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        with self._inference_lock:
            status = self.vision.status()
            embedding = np.asarray(self.vision.embed(image), dtype=float).reshape(-1)
            if not np.isfinite(embedding).all():
                raise LabError('The image encoder returned invalid features.', 503)
            status = self.vision.status()
            embedding_version = status.get('embedding_version')
            if not embedding_version:
                raise LabError('Image encoder version is unavailable.', 503)
            if domain == 'street':
                try:
                    if hasattr(self.vision, 'predict_embedding'):
                        anomaly = self.vision.predict_embedding(embedding, embedding_version=embedding_version)
                    else:
                        anomaly = self.vision.predict(image)
                except (OSError, ValueError, ImportError, RuntimeError) as exc:
                    anomaly = {'available': False, 'calibrated': False, 'anomaly_probability': None, 'reason': str(exc)}
                try:
                    detection = self.detector.predict(image)
                except (OSError, ValueError, ImportError, RuntimeError) as exc:
                    detection = {'available': False, 'reason': str(exc), 'target': .9, 'target_met': False}
                if detection.get('available'):
                    embedding, embedding_version = detection_features(embedding, embedding_version, detection)
                else:
                    teaching_allowed = False
            else:
                anomaly = {'available': False, 'calibrated': False, 'anomaly_probability': None,
                           'reason': 'Street-image anomaly probabilities are not validated on simulated graphics.'}
                detection = {'available': False, 'reason': 'The pretrained real-image detector is not validated on simulation graphics.',
                             'target': .9, 'target_met': False}
        if domain == 'street' and not detection.get('available'):
            policy = {'action': None, 'confidence': None, 'probabilities': None, 'calibrated': False, 'abstain': True,
                      'reason': 'The pretrained detector must be available before teaching or recommending street actions.'}
        else:
            policy = self.learner.predict(embedding.tolist(), domain, embedding_version=embedding_version)
        snapshot_id = uuid4().hex
        saved = {'snapshot_id': snapshot_id, 'domain': domain, 'group_id': group_id,
                 'image_hash': image_hash, 'embedding': embedding.tolist(),
                 'embedding_version': embedding_version, 'sample_id': sample_id,
                 'captured_at': datetime.now(timezone.utc).isoformat(),
                 'anomaly': anomaly, 'detection': detection, 'policy': policy, 'decision_teaching_allowed': teaching_allowed}
        image.save(self.snapshots / f'{snapshot_id}.png')
        (self.snapshots / f'{snapshot_id}.json').write_text(json.dumps(saved, allow_nan=False), encoding='utf-8')
        return {'ok': True, **{k: v for k, v in saved.items() if k not in ('embedding', 'image_hash')},
                'image_url': f'/api/learning/snapshot/{snapshot_id}'}

    def feedback(self, payload: dict) -> dict:
        snapshot = self._snapshot(payload.get('snapshot_id'))
        if not snapshot['decision_teaching_allowed']:
            if snapshot.get('sample_id'):
                raise LabError('Roundabout anomaly examples do not label correct signal phases. Upload signalized intersection footage to teach light choices.')
            raise LabError('The pretrained detector must be available before teaching street actions. Analyze the image again after setup.')
        action = payload.get('action')
        if action not in ('ns', 'ew', 'walk', 'hold'):
            raise LabError('Choose NS, EW, walk or hold.')
        result = self.learner.record_feedback(
            image_hash=snapshot['image_hash'], embedding=snapshot['embedding'],
            group_id=snapshot['group_id'], source_domain=snapshot['domain'],
            chosen_action=action, captured_at=snapshot['captured_at'],
            embedding_version=snapshot['embedding_version'],
            proposed_probabilities=snapshot['policy'].get('probabilities'),
            model_version=snapshot['policy'].get('model_version'))
        return {**result, 'ok': True}

    def train(self, payload: dict) -> dict:
        domain = payload.get('domain')
        if domain not in ('street', 'simulation'):
            raise LabError('Invalid training domain.')
        return self.learner.train(domain)

    def export(self, domain: str) -> dict:
        if domain not in ('street', 'simulation'):
            raise LabError('Invalid export domain.')
        return {'ok': True, 'domain': domain, 'label_origin': 'human_action_preferences',
                'records': self.learner.export_records(domain)}
