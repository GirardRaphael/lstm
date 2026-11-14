"""Real-video API contracts; colored fixtures are deliberately not training data."""
from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import pytest

SERVER = Path(__file__).resolve().parents[1] / 'presentation/server'
sys.path.insert(0, str(SERVER))
from learning_lab import LabError, LearningLab
from video_library import VideoLibrary, parse_byte_range


class FixtureVision:
    def status(self):
        return {'available': True, 'embedding_version': 'video-fixture-encoder'}

    def embed(self, image):
        return np.asarray(image.resize((1, 1)), dtype=float).reshape(-1) / 255

    def predict(self, image):
        return {'anomaly_probability': .6, 'calibrated': False, 'decision': 'review'}


class FixtureDetector:
    def predict(self, image):
        return {'available': True, 'model_version': 'video-fixture-detector', 'objects': [
            {'class_id': 3, 'class_name': 'car', 'score': .8, 'box': [.1, .2, .3, .4]}]}


def png_data(color='red'):
    stream = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def payload(color='red', seconds=3.5, group='recording-one'):
    return {'image_data_url': png_data(color), 'domain': 'street', 'group_id': group,
            'video': {'recording_id': group, 'name': 'Signalized intersection example',
                      'timestamp_seconds': seconds, 'source_url': 'https://example.org/footage',
                      'license': 'CC BY 4.0'}}


@pytest.fixture
def lab(tmp_path):
    return LearningLab(tmp_path, vision=FixtureVision(), detector=FixtureDetector())


def test_monitor_runs_real_inference_contract_without_saving_frames_or_labels(lab):
    for timestamp in (0, 2, 5):
        result = lab.monitor(payload(seconds=timestamp))
        assert result['video']['timestamp_seconds'] == timestamp
        assert result['group_id'] == 'recording-one'
        assert result['detection']['objects'][0]['class_name'] == 'car'
        assert result['policy']['abstain'] is True
        assert result['reviewed_action'] is None
        for hidden in ('snapshot_id', 'image_url', 'image_hash', 'embedding'):
            assert hidden not in result
    assert list(lab.snapshots.iterdir()) == []
    assert lab.learner.export_records('street') == []
    assert lab.review_frames()['total'] == 0


@pytest.mark.parametrize('video,domain,group', [
    ({'recording_id': 'other', 'name': 'clip', 'timestamp_seconds': 1}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': 1}, 'simulation', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': True}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': float('nan')}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': float('inf')}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': 10**400}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': -1}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip\n', 'timestamp_seconds': 1}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'x'*241, 'timestamp_seconds': 1}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': 1, 'source_url': 'javascript:alert(1)'}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': 1, 'source_url': 'file:///secret'}, 'street', 'one'),
    ({'recording_id': 'one', 'name': 'clip', 'timestamp_seconds': 1, 'source_url': 'https://user:password@example.org'}, 'street', 'one'),
])
def test_video_provenance_is_validated_before_inference(lab, video, domain, group):
    with pytest.raises(LabError):
        lab.analyze({'image_data_url': png_data(), 'domain': domain, 'group_id': group, 'video': video})
    assert list(lab.snapshots.iterdir()) == []


def test_monitor_requires_video_metadata_and_dataset_cannot_impersonate_video(lab):
    with pytest.raises(LabError):
        lab.monitor({'image_data_url': png_data(), 'domain': 'street', 'group_id': 'one'})
    with pytest.raises(LabError):
        lab.analyze({'sample_id': 'abcdef1234567890', 'video': payload()['video']})


def test_capture_deduplicates_pixels_per_recording_and_keeps_first_timestamp(lab):
    first = lab.analyze(payload())
    second = lab.analyze(payload(seconds=10))
    assert second['duplicate'] is True
    assert first['snapshot_id'] == second['snapshot_id']
    assert second['video']['timestamp_seconds'] == 3.5
    assert len(list(lab.snapshots.glob('*.png'))) == 1
    other = lab.analyze(payload(group='recording-two'))
    assert other['snapshot_id'] != first['snapshot_id']
    assert lab.review_frames('recording-one')['total'] == 1
    assert lab.review_frames()['total'] == 2


def test_reanalyzing_saved_video_refreshes_policy_without_changing_provenance_or_label(lab):
    first = lab.analyze(payload())
    lab.feedback({'snapshot_id': first['snapshot_id'], 'action': 'hold'})
    image = lab.snapshot_image(first['snapshot_id'])
    original_pixels, original_modified = image.read_bytes(), image.stat().st_mtime_ns
    new_policy = {'trained': True, 'action': 'ns', 'abstain': False, 'calibrated': True,
                  'confidence': .97, 'probabilities': {'ns': .97, 'ew': .01, 'walk': .01, 'hold': .01},
                  'model_version': 'newly-trained-fixture'}
    changed_metadata = payload(seconds=12)
    changed_metadata['video']['name'] = 'A later capture of the same pixels'
    with patch.object(lab.learner, 'predict', return_value=new_policy) as predict, \
         patch.object(lab.detector, 'predict', wraps=lab.detector.predict) as detect:
        refreshed = lab.analyze(changed_metadata)
    assert predict.call_count == detect.call_count == 1
    assert refreshed['snapshot_id'] == first['snapshot_id'] and refreshed['duplicate'] is True
    assert refreshed['captured_at'] == first['captured_at']
    assert refreshed['video'] == first['video']
    assert refreshed['analyzed_at'] >= first['analyzed_at']
    assert refreshed['policy'] == new_policy and refreshed['reviewed_action'] == 'hold'
    assert lab.review_frame(first['snapshot_id'])['policy'] == new_policy
    assert image.read_bytes() == original_pixels and image.stat().st_mtime_ns == original_modified
    records = lab.learner.export_records('street')
    assert len(records) == 1 and records[0]['chosen_action'] == 'hold'
    assert len(list(lab.snapshots.glob('*.png'))) == 1


def test_review_and_zip_contain_human_labels_only_and_raw_pixels(lab):
    first = lab.analyze(payload())
    second = lab.analyze(payload('blue', seconds=8))
    unrelated = lab.analyze({'image_data_url': png_data('green'), 'domain': 'simulation', 'group_id': 'sim'})
    assert lab.review_frames()['total'] == 2
    with pytest.raises(LabError):
        lab.review_frame(unrelated['snapshot_id'])
    assert all(frame['reviewed_action'] is None for frame in lab.review_frames()['frames'])
    lab.feedback({'snapshot_id': first['snapshot_id'], 'action': 'ew'})
    assert lab._snapshot(first['snapshot_id'])['reviewed_action'] == 'ew'
    # A correction via an older image-only snapshot must update video exports too.
    correction = lab.analyze({'image_data_url': png_data(), 'domain': 'street', 'group_id': 'recording-one'})
    lab.feedback({'snapshot_id': correction['snapshot_id'], 'action': 'hold'})
    reviewed = lab.review_frame(first['snapshot_id'])
    assert reviewed['reviewed_action'] == 'hold'
    assert 'image_hash' not in reviewed and 'embedding' not in reviewed
    assert lab.review_frame(second['snapshot_id'])['reviewed_action'] is None
    with lab.training_package('recording-one') as package, zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        frames = manifest['frames']
        assert len(frames) == 2
        assert [frame['human_action'] for frame in frames] == ['hold', None]
        assert [frame['label_source'] for frame in frames] == ['human', None]
        assert [frame['review_status'] for frame in frames] == ['reviewed', 'unreviewed']
        assert manifest['road_safety_validated'] is False
        assert manifest['unreviewed_frames_are_labeled'] is False
        for frame in frames:
            assert frame['group_id'] == frame['video']['recording_id'] == 'recording-one'
            assert 'policy' not in frame and 'detection' not in frame and 'embedding' not in frame
            assert archive.read(frame['image_path']) == lab.snapshot_image(frame['snapshot_id']).read_bytes()
        image = Image.open(io.BytesIO(archive.read(frames[0]['image_path'])))
        assert image.getpixel((10, 20)) == (255, 0, 0)


def test_rejected_human_label_does_not_mark_a_frame_reviewed(lab):
    first = lab.analyze(payload())
    with pytest.raises(LabError):
        lab.feedback({'snapshot_id': first['snapshot_id'], 'action': 'invalid'})
    assert lab.review_frame(first['snapshot_id'])['reviewed_action'] is None
    with patch.object(lab.learner, 'record_feedback', side_effect=ValueError('Rejected fixture')):
        with pytest.raises(ValueError):
            lab.feedback({'snapshot_id': first['snapshot_id'], 'action': 'ns'})
    assert lab._snapshot(first['snapshot_id'])['reviewed_action'] is None


def test_frame_limits_paging_and_package_limit(lab):
    with pytest.raises(LabError):
        lab.training_package()
    with patch('learning_lab.MAX_VIDEO_FRAMES', 1):
        lab.analyze(payload())
        with pytest.raises(LabError) as blocked:
            lab.analyze(payload('blue'))
        assert blocked.value.status == 409
    assert lab.review_frames(offset=1, limit=1)['frames'] == []
    for offset, limit in [(-1, 1), (0, 101), (0, 0)]:
        with pytest.raises(LabError):
            lab.review_frames(offset=offset, limit=limit)
    with patch('learning_lab.MAX_PACKAGE_BYTES', 1):
        with pytest.raises(LabError) as blocked:
            lab.training_package()
        assert blocked.value.status == 413


def library_fixture(root, filename='example.mp4'):
    directory = root / 'data/signalized_traffic'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'example.mp4').write_bytes(b'0123456789abcdef')
    entry = {'id': 'sample-one', 'filename': filename, 'title': 'Test signalized intersection',
             'recording_id': 'fixture-recording', 'source_url': 'https://example.org/clip',
             'license': 'CC BY 4.0', 'description': 'Test fixture only', 'attribution': 'Fixture author',
             'license_url': 'https://creativecommons.org/licenses/by/4.0/',
             'changes': 'Extracted clip and resized frames for this fixture.'}
    (directory / 'manifest.json').write_text(json.dumps({'videos': [entry]}), encoding='utf-8')
    return VideoLibrary(root)


def test_examples_expose_metadata_and_allowlisted_url_but_no_paths(tmp_path):
    library = library_fixture(tmp_path)
    example = library.examples()['videos'][0]
    assert example['video_url'] == '/api/learning/video/sample-one'
    assert example['recording_id'] == 'fixture-recording'
    expected = hashlib.sha256(b'video-size:16;' + b'0123456789abcdef' * 3).hexdigest()
    assert example['upload_fingerprint'] == expected
    assert 'filename' not in example and 'path' not in example
    path, mime = library.video_path('sample-one')
    assert path.read_bytes() == b'0123456789abcdef'
    assert mime == 'video/mp4'
    for identifier in ('../outside', 'unknown', 'https://example.org/video'):
        with pytest.raises(LabError):
            library.video_path(identifier)


def test_reuploaded_included_video_keeps_its_original_recording_group(tmp_path, lab):
    library = library_fixture(tmp_path)
    example = library.examples()['videos'][0]
    imported = lab.analyze(payload(group=example['recording_id']))
    lab.feedback({'snapshot_id': imported['snapshot_id'], 'action': 'hold'})
    reuploaded = lab.analyze(payload(group='video-' + example['upload_fingerprint']))
    assert reuploaded['group_id'] == reuploaded['video']['recording_id'] == example['recording_id']
    assert reuploaded['snapshot_id'] == imported['snapshot_id'] and reuploaded['duplicate'] is True
    assert reuploaded['reviewed_action'] == 'hold'
    assert lab.review_frames()['total'] == 1
    assert lab.review_frames('video-' + example['upload_fingerprint'])['total'] == 1
    assert lab.learner.status('street')['group_count'] == 1
    unseen = lab.monitor(payload(group='video-' + 'a' * 64))
    assert unseen['group_id'] == 'video-' + 'a' * 64


def test_export_preserves_known_source_credits_license_and_changes_without_arbitrary_files(tmp_path, lab):
    library = library_fixture(tmp_path)
    manifest_path = library.directory / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text())
    source_manifest['videos'][0]['attribution_file'] = '../../private.txt'
    manifest_path.write_text(json.dumps(source_manifest), encoding='utf-8')
    (tmp_path / 'private.txt').write_text('This must never be exported.')
    lab.analyze(payload(group='fixture-recording'))
    with lab.training_package('fixture-recording') as package, zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        source = manifest['recording_sources'][0]
        assert source['attribution'] == 'Fixture author'
        assert source['license_url'] == 'https://creativecommons.org/licenses/by/4.0/'
        assert source['changes'] == 'Extracted clip and resized frames for this fixture.'
        credit = archive.read('ATTRIBUTION.md').decode('utf-8')
        for value in ('Fixture author', source['source_url'], source['license_url'], source['changes']):
            assert value in credit
        assert 'This must never be exported.' not in credit
        assert all(name in ('manifest.json', 'ATTRIBUTION.md') or name.startswith('images/') for name in archive.namelist())


@pytest.mark.parametrize('filename', ['../../outside.mp4', 'example.txt'])
def test_video_manifest_rejects_unconfined_or_nonvideo_paths(tmp_path, filename):
    library = library_fixture(tmp_path, filename)
    with pytest.raises(LabError):
        library.examples()


def test_video_manifest_rejects_symlink_escape_when_supported(tmp_path):
    library = library_fixture(tmp_path, 'linked.mp4')
    outside = tmp_path / 'private.mp4'
    outside.write_bytes(b'private')
    try:
        (library.directory / 'linked.mp4').symlink_to(outside)
    except OSError:
        pytest.skip('This host does not permit creating symlinks without elevation.')
    with pytest.raises(LabError):
        library.examples()


@pytest.mark.parametrize('header,expected', [
    (None, None), ('bytes=0-3', (0, 3)), ('bytes=4-', (4, 15)),
    ('bytes=-4', (12, 15)), ('bytes=-100', (0, 15)), ('bytes=3-99', (3, 15)),
])
def test_range_parser_supports_browser_seeking(header, expected):
    assert parse_byte_range(header, 16) == expected


@pytest.mark.parametrize('header', ['bytes=99-', 'bytes=5-3', 'bytes=-0', 'bytes=-', 'bytes=0-1,3-4', 'bytes=x-y', 'items=0-4'])
def test_range_parser_rejects_invalid_or_unsatisfiable_ranges(header):
    with pytest.raises(LabError) as blocked:
        parse_byte_range(header, 16)
    assert blocked.value.status == 416


def test_http_video_range_and_training_review_routes(tmp_path, lab):
    import sidecar
    library = library_fixture(tmp_path)
    with patch.object(sidecar, 'LAB', lab), patch.object(sidecar, 'VIDEOS', library):
        server = ThreadingHTTPServer(('127.0.0.1', 0), sidecar.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
        def request(method, path, data=None, extra=None):
            headers = {'Host': '127.0.0.1:43126', **(extra or {})}
            if data is not None:
                headers['Content-Type'] = 'application/json'
            connection.request(method, path, json.dumps(data) if data is not None else None, headers)
            response = connection.getresponse()
            return response, response.read()
        try:
            response, body = request('GET', '/api/learning/video-examples')
            assert response.status == 200 and json.loads(body)['videos'][0]['id'] == 'sample-one'
            response, body = request('GET', '/api/learning/video/sample-one', extra={'Range': 'bytes=4-7'})
            assert response.status == 206 and body == b'4567'
            assert response.getheader('Content-Range') == 'bytes 4-7/16'
            assert response.getheader('Accept-Ranges') == 'bytes'
            response, body = request('GET', '/api/learning/video/sample-one', extra={'Range': 'bytes=999-'})
            assert response.status == 416 and response.getheader('Content-Range') == 'bytes */16'
            response, body = request('POST', '/api/learning/monitor', payload())
            assert response.status == 200 and 'snapshot_id' not in json.loads(body)
            assert list(lab.snapshots.iterdir()) == []
            response, body = request('POST', '/api/learning/analyze', payload())
            snapshot = json.loads(body)
            assert response.status == 200
            response, body = request('GET', '/api/learning/review-frames?group_id=recording-one')
            assert response.status == 200 and json.loads(body)['total'] == 1
            response, body = request('GET', f"/api/learning/review-frame/{snapshot['snapshot_id']}")
            assert response.status == 200 and json.loads(body)['reviewed_action'] is None
            response, body = request('POST', '/api/learning/anomaly-feedback', {'snapshot_id': snapshot['snapshot_id'], 'judgement': 'uncertain', 'reason': 'Cannot see the signal'})
            assert response.status == 200 and json.loads(body)['anomaly_review']['label'] is None
            response, body = request('GET', '/api/learning/training-package?group_id=recording-one')
            assert response.status == 200 and response.getheader('Content-Type') == 'application/zip'
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                assert json.loads(archive.read('manifest.json'))['frames'][0]['human_action'] is None
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


def test_anomaly_reviews_are_explicit_separate_and_survive_reanalysis_and_reload(lab):
    first = lab.analyze(payload())
    assert first['anomaly_review'] is None
    assert not lab.anomaly_reviews.path.exists()
    assert lab.monitor(payload())['anomaly_review'] is None
    normal = lab.anomaly_feedback({'snapshot_id': first['snapshot_id'], 'judgement': 'normal', 'reason': 'Visible traffic stays within the marked lanes.'})
    assert normal['anomaly_review']['label'] == 0
    assert normal['anomaly_review']['label_source'] == 'human'
    assert lab.learner.export_records('street') == []
    other = lab.analyze(payload(group='a-different-recording'))
    assert other['anomaly_review'] is None
    refreshed = lab.analyze(payload(seconds=14))
    assert refreshed['snapshot_id'] == first['snapshot_id']
    assert refreshed['anomaly_review'] == normal['anomaly_review']
    assert refreshed['reviewed_action'] is None
    correction = lab.anomaly_feedback({'snapshot_id': first['snapshot_id'], 'judgement': 'anomaly', 'reason': 'Reconsidered after watching the surrounding video.'})
    assert correction['anomaly_review']['label'] == 1
    uncertain = lab.anomaly_feedback({'snapshot_id': first['snapshot_id'], 'judgement': 'uncertain', 'reason': 'The signal is obscured.'})
    assert uncertain['anomaly_review']['label'] is None
    reloaded = LearningLab(lab.root, vision=FixtureVision(), detector=FixtureDetector())
    assert reloaded.review_frame(first['snapshot_id'])['anomaly_review'] == uncertain['anomaly_review']
    assert len(reloaded.anomaly_reviews.records()) == 1
    assert reloaded.anomaly_reviews.status()['labeled_count'] == 0
    assert reloaded.anomaly_reviews.status()['training_available'] is False
    assert reloaded.learner.export_records('street') == []


def test_zip_anomaly_labels_keep_uncertain_and_unreviewed_examples_unlabeled(lab):
    for index, (color, judgement) in enumerate([('red', 'normal'), ('blue', 'anomaly'), ('green', 'uncertain'), ('yellow', None)]):
        frame = lab.analyze(payload(color, seconds=index))
        if judgement:
            lab.anomaly_feedback({'snapshot_id': frame['snapshot_id'], 'judgement': judgement, 'reason': 'Human fixture review'})
    with lab.training_package() as package, zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        frames = manifest['frames']
        assert [frame['human_anomaly_label'] for frame in frames] == [0, 1, None, None]
        assert [frame['anomaly_review_status'] for frame in frames] == ['normal', 'anomaly', 'uncertain', 'unreviewed']
        assert [frame['anomaly_label_source'] for frame in frames] == ['human', 'human', None, None]
        assert [frame['anomaly_review_reason'] for frame in frames] == ['Human fixture review'] * 3 + [None]
        assert all(frame['human_action'] is None for frame in frames)
        assert manifest['anomaly_label_origin'] == 'human_anomaly_reviews'
    assert lab.learner.export_records('street') == []


def test_anomaly_review_rejects_simulation_and_publisher_examples_without_mutating_source(lab):
    simulated = lab.analyze({'image_data_url': png_data(), 'domain': 'simulation', 'group_id': 'sim'})
    with pytest.raises(LabError):
        lab.anomaly_feedback({'snapshot_id': simulated['snapshot_id'], 'judgement': 'normal'})
    publisher = lab.analyze(payload())
    stored = lab._snapshot(publisher['snapshot_id'])
    stored['sample_id'] = 'abcdef1234567890'
    lab._write_snapshot(stored)
    before = (lab.snapshots / f"{publisher['snapshot_id']}.json").read_bytes()
    with pytest.raises(LabError):
        lab.anomaly_feedback({'snapshot_id': publisher['snapshot_id'], 'judgement': 'anomaly'})
    assert (lab.snapshots / f"{publisher['snapshot_id']}.json").read_bytes() == before
    assert lab.anomaly_reviews.records() == {}


@pytest.mark.parametrize('judgement,reason', [('automatic', ''), (True, ''), ('anomaly', None), ('normal', 'x'*2001), ('uncertain', 'bad\x00control')])
def test_invalid_anomaly_reviews_never_create_labels(lab, judgement, reason):
    frame = lab.analyze(payload())
    with pytest.raises(LabError):
        lab.anomaly_feedback({'snapshot_id': frame['snapshot_id'], 'judgement': judgement, 'reason': reason})
    assert lab.anomaly_reviews.records() == {}


def test_anomaly_review_atomic_failure_preserves_existing_human_review(lab):
    frame = lab.analyze(payload())
    first = lab.anomaly_feedback({'snapshot_id': frame['snapshot_id'], 'judgement': 'normal'})
    with patch.object(Path, 'replace', side_effect=OSError('Fixture full disk')):
        with pytest.raises(OSError):
            lab.anomaly_feedback({'snapshot_id': frame['snapshot_id'], 'judgement': 'anomaly'})
    assert lab.review_frame(frame['snapshot_id'])['anomaly_review'] == first['anomaly_review']
