"""Explicit human anomaly reviews; separate from publisher labels and signal preferences."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

from learning_lab import LabError

MAX_ANOMALY_REVIEWS = 5000
MAX_REVIEW_FILE_BYTES = 20 * 1024 * 1024


class AnomalyReviewStore:
    def __init__(self, storage: Path):
        self.path = Path(storage) / 'human_anomaly_reviews.json'
        self._lock = threading.RLock()

    @staticmethod
    def key(snapshot: dict) -> str:
        return hashlib.sha256((snapshot['group_id'] + ':' + snapshot['image_hash']).encode()).hexdigest()

    def records(self) -> dict:
        with self._lock:
            if not self.path.is_file():
                return {}
            if self.path.stat().st_size > MAX_REVIEW_FILE_BYTES:
                raise LabError('The local anomaly-review file exceeds its supported size.', 503)
            try:
                data = json.loads(self.path.read_text(encoding='utf-8'))
            except (ValueError, OSError) as exc:
                raise LabError('The local anomaly reviews could not be read; existing reviews were preserved.', 503) from exc
            if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('records'), dict):
                raise LabError('The local anomaly-review file has an unsupported format.', 503)
            return data['records']

    def public_review(self, snapshot: dict, records: dict | None = None) -> dict | None:
        if snapshot.get('domain') != 'street' or snapshot.get('sample_id'):
            return None
        record = (self.records() if records is None else records).get(self.key(snapshot))
        if not record:
            return None
        return {name: record[name] for name in ('judgement', 'label', 'reason', 'reviewed_at', 'label_source')}

    def save(self, snapshot: dict, judgement: object, reason: object = '') -> dict:
        if snapshot.get('domain') != 'street' or snapshot.get('sample_id'):
            raise LabError('Review a captured real street image. Simulation and publisher dataset labels are kept separate.')
        if judgement not in ('normal', 'anomaly', 'uncertain'):
            raise LabError('Choose normal, anomaly, or uncertain.')
        if not isinstance(reason, str) or len(reason) > 2000 or any(ord(char) < 32 and char not in '\r\n\t' or ord(char) == 127 for char in reason):
            raise LabError('Your anomaly-review reason must be text of at most 2000 characters without invalid controls.')
        with self._lock:
            records = self.records()
            key = self.key(snapshot)
            if key not in records and len(records) >= MAX_ANOMALY_REVIEWS:
                raise LabError('Local anomaly-review capacity reached. Export your reviewed frames before extending this collection.', 409)
            record = {'image_hash': snapshot['image_hash'], 'group_id': snapshot['group_id'],
                      'domain': 'street', 'judgement': judgement,
                      'label': {'normal': 0, 'anomaly': 1, 'uncertain': None}[judgement],
                      'reason': reason.strip(), 'reviewed_at': datetime.now(timezone.utc).isoformat(),
                      'label_source': 'human', 'road_safety_validated': False}
            records[key] = record
            serialized = json.dumps({'schema_version': 1, 'records': records}, allow_nan=False)
            if len(serialized.encode('utf-8')) > MAX_REVIEW_FILE_BYTES:
                raise LabError('Local anomaly-review capacity reached; export your records before extending this collection.', 409)
            temporary = self.path.with_suffix('.json.tmp')
            temporary.write_text(serialized, encoding='utf-8')
            temporary.replace(self.path)
            return self.public_review(snapshot, records)

    def status(self) -> dict:
        records = list(self.records().values())
        labeled = [record for record in records if record['label'] is not None]
        return {'total': len(records), 'counts': dict(Counter(record['judgement'] for record in records)),
                'labeled_count': len(labeled), 'labeled_group_count': len({record['group_id'] for record in labeled}),
                'training_available': False, 'label_origin': 'human_anomaly_reviews',
                'reason': 'Reviews are collected separately. They have not trained or validated the anomaly model.'}
