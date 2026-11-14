"""Analyze the installed sample's verified pictures and save unlabelled review frames.

Start the local intersection app first. This calls inference only; it never creates
human labels or trains the light-decision policy.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    directory = (args.root / 'data/signalized_traffic').resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8-sig'))
    pending = []
    for video in manifest['videos']:
        for frame in video['frames']:
            path = (directory / frame['filename']).resolve()
            if not path.is_relative_to(directory) or path.suffix.lower() != '.jpg':
                raise ValueError('Frame path must be a JPEG inside the installed sample directory.')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != frame['sha256']:
                raise ValueError(f'Frame checksum failed: {path.name}')
            if frame['recording_id'] != video['recording_id'] or frame.get('preferred_action') is not None or frame['review_status'] != 'unreviewed':
                raise ValueError('The import requires unreviewed frames with consistent recording groups.')
            pending.append((video, frame, raw))
    if not 1 <= len(pending) <= 60:
        raise ValueError('Expected between 1 and 60 verified sample frames.')
    # Validate every source frame before making any inference request.
    for index, (video, frame, raw) in enumerate(pending, start=1):
        metadata = {'recording_id': video['recording_id'], 'name': video['title'],
                    'timestamp_seconds': frame['timestamp_seconds'],
                    'source_url': video['source_url'], 'license': video['license']}
        payload = {'domain': 'street', 'group_id': video['recording_id'], 'video': metadata,
                   'image_data_url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')}
        request = Request('http://127.0.0.1:43126/api/learning/analyze', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        if not result.get('ok') or not result.get('snapshot_id') or result.get('group_id') != video['recording_id']:
            raise RuntimeError('The service did not save the expected review frame.')
        print(f'{index}/{len(pending)} saved at {frame["timestamp_seconds"]}s; human labels unchanged', flush=True)

if __name__ == '__main__':
    main()
