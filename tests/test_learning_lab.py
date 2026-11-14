"""Integration checks for local image review and human-only action labels."""
from __future__ import annotations
import base64
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SERVER = Path(__file__).resolve().parents[1] / 'presentation/server'
sys.path.insert(0, str(SERVER))
from learning_lab import LearningLab, LabError, detection_features


class FakeVision:
    def status(self):
        return {'available': True, 'embedding_version': 'fixture-encoder', 'model_version': 'fixture-anomaly'}
    def embed(self, image):
        return np.asarray(image.resize((1, 1)), dtype=float).reshape(-1) / 255
    def predict(self, image):
        return {'anomaly_probability': 0.6, 'calibrated': True, 'decision': 'review'}


class FakeDetector:
    def status(self):
        return {'available': True, 'model_version': 'fixture-detector', 'target_met': False}
    def predict(self, image):
        return {**self.status(), 'objects': [{'class_id': 3, 'class_name': 'car', 'score': .8, 'box': [.1, .1, .1, .1]}]}

def png_data(color='red'):
    stream = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


class LearningLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.lab = LearningLab(self.root, vision=FakeVision(), detector=FakeDetector())
    def tearDown(self):
        self.temp.cleanup()

    def test_analyze_freezes_pixels_and_human_feedback_is_required(self):
        result = self.lab.analyze({'image_data_url': png_data(), 'domain': 'simulation', 'group_id': 'episode-one'})
        self.assertFalse(result['detection']['available'])
        self.assertFalse(result['anomaly']['calibrated'])
        self.assertIsNone(result['anomaly']['anomaly_probability'])
        self.assertEqual(self.lab.learner.export_records('simulation'), [])
        self.assertTrue(self.lab.snapshot_image(result['snapshot_id']).is_file())
        self.lab.feedback({'snapshot_id': result['snapshot_id'], 'action': 'ew'})
        records = self.lab.learner.export_records('simulation')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['chosen_action'], 'ew')
        self.assertEqual(records[0]['group_id'], 'episode-one')

    def test_duplicate_correction_and_reload_preserve_one_example(self):
        one = self.lab.analyze({'image_data_url': png_data(), 'domain': 'street', 'group_id': 'recording-one'})
        self.lab.feedback({'snapshot_id': one['snapshot_id'], 'action': 'hold'})
        two = self.lab.analyze({'image_data_url': png_data(), 'domain': 'street', 'group_id': 'recording-one'})
        self.lab.feedback({'snapshot_id': two['snapshot_id'], 'action': 'ns'})
        other = LearningLab(self.root, vision=FakeVision(), detector=FakeDetector())
        records = other.learner.export_records('street')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['chosen_action'], 'ns')

    def test_publisher_anomaly_never_becomes_a_light_label(self):
        dataset = self.root / 'dataset'
        (dataset / 'images').mkdir(parents=True)
        sample_id = 'abcdef1234567890'
        Image.new('RGB', (64, 64), 'blue').save(dataset / 'images' / f'{sample_id}.jpg')
        with (dataset / 'labels.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'source_video', 'split', 'label', 'class_name', 'image_path'])
            writer.writeheader()
            writer.writerow({'sample_id': sample_id, 'source_video': 'DJI_one', 'split': 'test', 'label': 1,
                             'class_name': 'anomaly', 'image_path': f'images/{sample_id}.jpg'})
        lab = LearningLab(self.root, vision=FakeVision(), detector=FakeDetector(), data_dir=dataset)
        result = lab.analyze({'sample_id': sample_id})
        self.assertFalse(result['decision_teaching_allowed'])
        with self.assertRaisesRegex(LabError, 'Roundabout'):
            lab.feedback({'snapshot_id': result['snapshot_id'], 'action': 'walk'})
        self.assertEqual(lab.learner.export_records('street'), [])

    def test_rejects_invalid_image_and_snapshot_paths(self):
        for url in ['https://example.com/image.jpg', 'data:image/png;base64,broken', png_data()]:
            with self.assertRaises(LabError):
                self.lab.analyze({'image_data_url': url, 'domain': 'unsupported', 'group_id': 'one'})
        with self.assertRaises(LabError):
            self.lab.analyze({'image_data_url': 'data:image/png;base64,broken', 'domain': 'street', 'group_id': 'one'})
        with self.assertRaises(LabError):
            self.lab.feedback({'snapshot_id': '../../outside', 'action': 'ns'})
        with self.assertRaises(LabError):
            self.lab.samples('test', -1)
        with self.assertRaises(LabError):
            self.lab.analyze({'image_data_url': png_data(), 'domain': 'street', 'group_id': ''})

    def test_real_detector_positions_feed_policy_without_creating_labels(self):
        result = self.lab.analyze({'image_data_url': png_data(), 'domain': 'street', 'group_id': 'footage-one'})
        self.assertTrue(result['detection']['available'])
        self.assertEqual(result['detection']['objects'][0]['class_name'], 'car')
        stored = self.lab._snapshot(result['snapshot_id'])
        self.assertEqual(len(stored['embedding']), 3 + 8 * 9)
        self.assertTrue(stored['embedding_version'].startswith('street-perception-'))
        self.assertAlmostEqual(stored['embedding'][3 + 2 * 9], np.log(2))
        self.assertEqual(self.lab.learner.export_records('street'), [])
        left, version = detection_features(np.zeros(3), 'encoder', FakeDetector().predict(None))
        right, same_version = detection_features(np.zeros(3), 'encoder', {'model_version': 'fixture-detector', 'objects': [{'class_id': 3, 'box': [.8,.8,.1,.1]}]})
        self.assertEqual(version, same_version)
        self.assertFalse(np.array_equal(left, right))

    def test_missing_detector_disables_street_teaching(self):
        class Unavailable(FakeDetector):
            def predict(self, image):
                return {'available': False, 'reason': 'Missing detector'}
        self.lab._detector = Unavailable()
        result = self.lab.analyze({'image_data_url': png_data(), 'domain': 'street', 'group_id': 'footage'})
        self.assertFalse(result['decision_teaching_allowed'])
        self.assertTrue(result['policy']['abstain'])
        with self.assertRaisesRegex(LabError, 'detector'):
            self.lab.feedback({'snapshot_id': result['snapshot_id'], 'action': 'ns'})

    def test_malformed_sample_ids_return_a_validation_error(self):
        for value in ({'id': 1}, [], None, '', 14):
            with self.assertRaises(LabError):
                self.lab.analyze({'sample_id': value})

    def test_insufficient_feedback_does_not_create_a_trained_policy(self):
        result = self.lab.train({'domain': 'street'})
        self.assertFalse(result.get('ok', False))
        self.assertFalse(self.lab.learner.status('street')['trained'])


if __name__ == '__main__':
    unittest.main()
