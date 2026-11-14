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
import math
import os
from pathlib import Path
import re
import threading
import tempfile
from datetime import datetime, timezone
from collections import Counter
from urllib.parse import urlparse
from uuid import uuid4
import zipfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_VIDEO_FRAMES = 5000
MAX_PACKAGE_BYTES = 512 * 1024 * 1024


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


def validate_source_url(value: object) -> str:
    value = clean_id(value, 'Video source URL', 2048)
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise LabError('Video source URL is invalid.') from exc
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise LabError('Video source URL must be an HTTP or HTTPS page without credentials.')
    return value


def video_metadata(value: object, domain: str, group_id: str) -> dict:
    if not isinstance(value, dict) or domain != 'street':
        raise LabError('Video frames must use the street domain and include recording metadata.')
    recording_id = clean_id(value.get('recording_id'), 'Video recording ID')
    if group_id != recording_id:
        raise LabError('Every video frame must keep the same recording ID as its group ID.')
    timestamp = value.get('timestamp_seconds')
    if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or timestamp < 0 or timestamp > 1.7976931348623157e308 or not math.isfinite(timestamp):
        raise LabError('Video timestamp must be a finite, nonnegative number of seconds.')
    result = {'recording_id': recording_id, 'name': clean_id(value.get('name'), 'Video name', 240),
              'timestamp_seconds': timestamp}
    if 'source_url' in value:
        result['source_url'] = validate_source_url(value['source_url'])
    if 'license' in value:
        result['license'] = clean_id(value['license'], 'Video license', 512)
    return result


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
    def __init__(self, root: Path, *, vision=None, learner=None, detector=None, anomaly_model=None, data_dir: Path | None = None):
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
        from anomaly_reviews import AnomalyReviewStore
        self.anomaly_reviews = AnomalyReviewStore(self.storage)
        self._vision = vision
        self._anomaly_model = anomaly_model
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
    def trained_anomaly(self):
        """An explicitly activated anomaly network never changes action embeddings."""
        with self._lock:
            if self._anomaly_model is not None:
                return self._anomaly_model
            directory = Path(os.environ.get('TRAFFIC_ANOMALY_MODEL_DIR', self.root / '.runtime/street_anomaly_finetuned'))
            activation = directory / 'activation.json'
            if not activation.is_file():
                return None
            record = json.loads(activation.read_text(encoding='utf-8'))
            if record.get('enabled') is not True:
                return None
            from finetuned_anomaly import FineTunedAnomaly
            candidate = FineTunedAnomaly(directory)
            status = candidate.status()
            if status.get('review_candidate_accepted') is not True or status.get('model_version') != record.get('model_version'):
                raise RuntimeError('The requested anomaly model has not passed its development gate or activation identity check.')
            self._anomaly_model = candidate
            return candidate

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
            anomaly_model = self.trained_anomaly
            if anomaly_model is not None:
                vision = {**vision, **anomaly_model.status()}
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            vision = {'available': False, 'reason': str(exc)}
        try:
            detection = self.detector.status()
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            detection = {'available': False, 'reason': str(exc), 'target': .9, 'target_met': False}
        return {'ok': True, 'vision': vision, 'detection': detection, 'anomaly_review': self.anomaly_reviews.status(),
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

    def _public_snapshot(self, saved: dict) -> dict:
        public = {'ok': True, **{k: v for k, v in saved.items() if k not in ('embedding', 'image_hash')}}
        public['anomaly_review'] = self.anomaly_reviews.public_review(saved)
        if saved.get('snapshot_id'):
            public['image_url'] = f"/api/learning/snapshot/{saved['snapshot_id']}"
        return public

    def analyze(self, payload: dict) -> dict:
        return self._analyze(payload, persist=True)

    def monitor(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or 'video' not in payload:
            raise LabError('Video monitoring requires recording metadata.')
        return self._analyze(payload, persist=False)

    def _analyze(self, payload: dict, *, persist: bool) -> dict:
        if not isinstance(payload, dict):
            raise LabError('Expected a JSON object.')
        sample_id = payload.get('sample_id')
        if 'sample_id' in payload:
            if 'video' in payload:
                raise LabError('Dataset sample images cannot be relabeled as video frames.')
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
        video = video_metadata(payload['video'], domain, group_id) if 'video' in payload else None
        if video and group_id.startswith('video-'):
            from video_library import VideoLibrary
            group_id = VideoLibrary(self.root).canonical_recording(group_id)
            video['recording_id'] = group_id
        image = decode_image(raw)
        # Hash canonical pixels, so metadata/re-encoding cannot duplicate an example.
        image_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        # Stable identity deduplicates repeated captures without inventing new recordings.
        snapshot_id = hashlib.sha256((group_id + ':' + image_hash).encode()).hexdigest()[:32] if video else uuid4().hex
        if persist and video:
            with self._lock:
                exists = (self.snapshots / f'{snapshot_id}.json').is_file()
                if not exists and sum(1 for _ in self.snapshots.glob('*.json')) >= MAX_VIDEO_FRAMES:
                    raise LabError('Local frame capacity reached. Export your reviewed frames before collecting more.', 409)
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
                    anomaly_model = self.trained_anomaly
                    if anomaly_model is not None:
                        anomaly = anomaly_model.predict(image)
                    elif hasattr(self.vision, 'predict_embedding'):
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
            policy = {'trained': False, 'action': None, 'confidence': None, 'probabilities': None, 'calibrated': False, 'abstain': True,
                      'reason': 'The pretrained detector must be available before teaching or recommending street actions.'}
        else:
            policy = self.learner.predict(embedding.tolist(), domain, embedding_version=embedding_version)
        analyzed_at = datetime.now(timezone.utc).isoformat()
        saved = {'domain': domain, 'group_id': group_id,
                 'image_hash': image_hash, 'embedding': embedding.tolist(),
                 'embedding_version': embedding_version, 'sample_id': sample_id,
                 'captured_at': analyzed_at, 'analyzed_at': analyzed_at,
                 'anomaly': anomaly, 'detection': detection, 'policy': policy,
                 'decision_teaching_allowed': teaching_allowed, 'reviewed_action': None}
        if video:
            saved['video'] = video
        if not persist:
            return self._public_snapshot(saved)
        saved['snapshot_id'] = snapshot_id
        duplicate = False
        with self._lock:
            if video and (self.snapshots / f'{snapshot_id}.json').is_file():
                original = self._snapshot(snapshot_id)
                # Refresh model outputs after training without changing frame provenance.
                saved['captured_at'] = original['captured_at']
                saved['video'] = original['video']
                saved = self._with_human_review(saved, self._human_reviews())
                duplicate = True
            else:
                if video and sum(1 for _ in self.snapshots.glob('*.json')) >= MAX_VIDEO_FRAMES:
                    raise LabError('Local frame capacity reached. Export your reviewed frames before collecting more.', 409)
                image.save(self.snapshots / f'{snapshot_id}.png')
            self._write_snapshot(saved)
        return {**self._public_snapshot(saved), **({'duplicate': True} if duplicate else {})}

    def _write_snapshot(self, saved: dict) -> None:
        target = self.snapshots / f"{saved['snapshot_id']}.json"
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(saved, allow_nan=False), encoding='utf-8')
        temporary.replace(target)

    def review_frame(self, snapshot_id: str) -> dict:
        with self._lock:
            saved = self._snapshot(snapshot_id)
            if not saved.get('video'):
                raise LabError('This snapshot is not a saved video frame.', 404)
            return self._public_snapshot(self._with_human_review(saved, self._human_reviews()))

    def _human_reviews(self) -> dict:
        return {record['image_hash']: record for record in self.learner.export_records('street')}

    @staticmethod
    def _with_human_review(saved: dict, reviews: dict) -> dict:
        # Canonical feedback can be corrected through another snapshot of these pixels.
        record = reviews.get(saved['image_hash'])
        result = {**saved, 'reviewed_action': None, 'reviewed_at': None}
        if record and record['group_id'] == saved['group_id']:
            result['reviewed_action'] = record['chosen_action']
            result['reviewed_at'] = datetime.fromtimestamp(record['updated_at'], timezone.utc).isoformat()
        return result

    def _video_frames(self, group_id: str | None = None) -> list[dict]:
        if group_id is not None:
            group_id = clean_id(group_id, 'Recording ID')
            if group_id.startswith('video-'):
                from video_library import VideoLibrary
                group_id = VideoLibrary(self.root).canonical_recording(group_id)
        result = []
        with self._lock:
            reviews = self._human_reviews()
            anomaly_reviews = self.anomaly_reviews.records()
            for path in self.snapshots.glob('*.json'):
                saved = self._snapshot(path.stem)
                if saved.get('video') and (group_id is None or saved['group_id'] == group_id):
                    reviewed = self._with_human_review(saved, reviews)
                    reviewed['anomaly_review'] = self.anomaly_reviews.public_review(saved, anomaly_reviews)
                    result.append(reviewed)
        return sorted(result, key=lambda frame: (frame['group_id'], frame['video']['timestamp_seconds'], frame['snapshot_id']))

    def review_frames(self, group_id: str | None = None, offset: int = 0, limit: int = 48) -> dict:
        if offset < 0 or not 1 <= limit <= 100:
            raise LabError('Invalid saved-frame page.')
        saved = self._video_frames(group_id)
        return {'ok': True, 'total': len(saved), 'frames': [
            {**{key: frame.get(key) for key in ('snapshot_id', 'video', 'domain', 'group_id', 'reviewed_action', 'anomaly_review', 'captured_at')},
             'image_url': f"/api/learning/snapshot/{frame['snapshot_id']}"}
            for frame in saved[offset:offset + limit]]}

    def training_package(self, group_id: str | None = None):
        """Return a bounded, seekable ZIP stream; callers must close it after delivery."""
        frames = self._video_frames(group_id)
        if not frames:
            raise LabError('Save at least one video frame before exporting a training package.', 404)
        if len(frames) > MAX_VIDEO_FRAMES:
            raise LabError('Export one recording at a time; this package has too many frames.', 413)
        paths = [self.snapshot_image(frame['snapshot_id']) for frame in frames]
        if sum(path.stat().st_size for path in paths) > MAX_PACKAGE_BYTES:
            raise LabError('Export one recording at a time; this package exceeds 512 MB.', 413)
        from video_library import VideoLibrary
        sources = VideoLibrary(self.root).recording_sources({frame['group_id'] for frame in frames})
        attribution = ['# Traffic video source attribution', '',
                       'These pictures are extracted from recorded footage. Human action labels are separate from the source authors and do not imply their endorsement.', '']
        for source in sources:
            attribution.extend([f"## {source['title']}", '',
                                source.get('attribution', 'Author attribution was not supplied in the local manifest.'), '',
                                f"Source: {source['source_url']}", f"License: {source['license']}",
                                f"License terms: {source.get('license_url', 'See the source page.')}", '',
                                f"Changes: {source.get('changes', 'Frames extracted from the recorded video.')}", '',
                                f"Recording group: {source['recording_id']}", ''])
        attribution.extend(['Export processing: frame images are saved as PNG files without model boxes. Images larger than 1920 pixels are resized before analysis. Preserve the source credits and license terms when sharing these images.', '',
                            'For user-provided recordings outside the included library, consult the source URL and license in each frame entry; permission and author attribution have not been verified.', ''])
        manifest = {
            'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'recording_sources': sources,
            'purpose': 'Human review of real traffic video frames; observation only.',
            'label_origin': 'human_action_preferences', 'anomaly_label_origin': 'human_anomaly_reviews', 'road_safety_validated': False,
            'unreviewed_frames_are_labeled': False,
            'notes': [
                'Only explicit human feedback supplies human_action. Null means unreviewed.',
                'Human anomaly reviews are separate: normal=0, anomaly=1. Uncertain or unreviewed frames have no anomaly training label.',
                'Anomaly reviews have not retrained the anomaly model and never modify publisher labels or evaluation results.',
                'Model detections and recommendations are predictions, not ground truth; they are excluded from labels.',
                'Images contain the decoded source pixels, resized only when larger than 1920 pixels; no detection boxes are baked in.',
                'Keep every frame of a recording in the same train/calibration/test partition.',
                'One recording cannot validate a decision policy across independent streets or recordings.',
            ], 'frames': [],
        }
        stream = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode='w+b')
        try:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for frame, path in zip(frames, paths):
                    archive_path = f"images/{frame['snapshot_id']}.png"
                    archive.write(path, archive_path)
                    action = frame.get('reviewed_action')
                    anomaly_review = frame.get('anomaly_review') or {}
                    manifest['frames'].append({
                        'snapshot_id': frame['snapshot_id'], 'image_path': archive_path,
                        'domain': frame['domain'], 'group_id': frame['group_id'], 'video': frame['video'],
                        'captured_at': frame['captured_at'], 'human_action': action,
                        'review_status': 'reviewed' if action else 'unreviewed',
                        'label_source': 'human' if action else None,
                        'reviewed_at': frame.get('reviewed_at'),
                        'human_anomaly_label': anomaly_review.get('label'),
                        'anomaly_review_status': anomaly_review.get('judgement', 'unreviewed'),
                        'anomaly_review_reason': anomaly_review.get('reason'),
                        'anomaly_reviewed_at': anomaly_review.get('reviewed_at'),
                        'anomaly_label_source': 'human' if anomaly_review.get('label') is not None else None,
                    })
                archive.writestr('manifest.json', json.dumps(manifest, indent=2, allow_nan=False))
                archive.writestr('ATTRIBUTION.md', '\n'.join(attribution))
            stream.seek(0)
            return stream
        except Exception:
            stream.close()
            raise

    def anomaly_feedback(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise LabError('Expected a JSON object.')
        with self._lock:
            snapshot = self._snapshot(payload.get('snapshot_id'))
            review = self.anomaly_reviews.save(snapshot, payload.get('judgement'), payload.get('reason', ''))
            return {'ok': True, 'snapshot_id': snapshot['snapshot_id'], 'anomaly_review': review}

    def feedback(self, payload: dict) -> dict:
        with self._lock:
            return self._feedback(payload)

    def _feedback(self, payload: dict) -> dict:
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
        # Labels are written only after the learner accepts explicit human feedback.
        snapshot['reviewed_action'] = action
        snapshot['reviewed_at'] = datetime.now(timezone.utc).isoformat()
        self._write_snapshot(snapshot)
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
