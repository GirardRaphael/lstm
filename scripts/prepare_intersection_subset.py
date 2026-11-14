"""Select a reproducible, recording-disjoint UIT-ADrone binary image subset.

Uses publisher frame masks. It does not invent violation subtypes or claim that
all video-derived labels can be independently established from a single image.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

QUOTAS = {'train': {0: 840, 1: 210}, 'validation': {0: 180, 1: 45}, 'test': {0: 180, 1: 45}}


def candidates(root: Path, seed: int = 42) -> tuple[dict, dict]:
    index = json.loads((root / 'source/frame_index.json').read_text())
    pools = {}
    audit = {}
    for video, files in sorted(index.items()):
        mask = np.load(root / 'source/masks' / (video + '.npy'), allow_pickle=False)
        if mask.ndim != 1 or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid binary mask: ' + video)
        numbers = sorted(int(Path(f['title']).stem) for f in files)
        if len(numbers) != len(set(numbers)):
            raise ValueError('Duplicate source frame: ' + video)
        if any(n % 5 or n // 5 >= len(mask) for n in numbers):
            raise ValueError('Unexpected frame numbering: ' + video)
        # All complete public folders establish the same zero-based step-five
        # mapping; every partial folder must contain its expected terminal frame.
        if max(numbers) != (len(mask) - 1) * 5:
            raise ValueError('Cannot establish mask/frame endpoint: ' + video)
        complete = len(files) == len(mask)
        if complete and numbers != list(range(0, len(mask) * 5, 5)):
            raise ValueError('Unexpected complete-folder indexing: ' + video)
        by_number = {int(Path(f['title']).stem): f for f in files}
        eligible = []
        for n in numbers:
            i = n // 5
            # Exclude half a second on either side of an annotation transition
            # at the published nominal 30 fps / sampling interval of five.
            if i < 3 or i + 3 >= len(mask) or not (mask[i-3:i+4] == mask[i]).all():
                continue
            eligible.append(n)
        # At least 60 source frame numbers (nominally two seconds) between
        # selected candidates, across both classes within each recording.
        selected = []
        last = -60
        for n in eligible:
            if n - last >= 60:
                f = by_number[n]
                selected.append({**f, 'source_video': video,
                                 'source_frame_number': n, 'mask_index': n // 5,
                                 'label': int(mask[n // 5])})
                last = n
        audit[video] = {'listed_frames': len(files), 'mask_frames': len(mask),
                        'complete_listing': complete,
                        'eligible_spaced_counts': dict(Counter(x['label'] for x in selected))}
        pools[video] = selected
    if not any(v['complete_listing'] for v in audit.values()):
        raise ValueError('No complete folders to check the indexing convention')
    return pools, audit


def select(root: Path, seed: int = 42) -> list[dict]:
    pools, audit = candidates(root, seed)
    rng = random.Random(seed)
    videos = sorted(pools)
    groups = None
    for _ in range(10000):
        rng.shuffle(videos)
        attempt = {'train': videos[:35], 'validation': videos[35:43], 'test': videos[43:]}
        if all(all(sum(x['label'] == label for v in names for x in pools[v]) >= quota
                   for label, quota in QUOTAS[split].items()) for split, names in attempt.items()):
            groups = attempt
            break
    if groups is None:
        raise ValueError('Not enough independent spaced frames for the requested quotas')
    rows = []
    for split, names in groups.items():
        for label, quota in QUOTAS[split].items():
            per_video = {v: [dict(x) for x in pools[v] if x['label'] == label] for v in names}
            for pool in per_video.values():
                rng.shuffle(pool)
            order = list(names)
            rng.shuffle(order)
            picked = []
            while len(picked) < quota:
                for v in order:
                    if per_video[v] and len(picked) < quota:
                        picked.append(per_video[v].pop())
            for item in picked:
                opaque = hashlib.sha256((item['source_video'] + '/' + item['title']).encode()).hexdigest()[:16]
                rows.append({
                    'sample_id': opaque, 'image_path': 'images/' + opaque + '.jpg',
                    'split': split, 'label': label, 'class_name': 'anomaly' if label else 'normal',
                    'source_dataset': 'UIT-ADrone', 'source_partition': 'test',
                    'source_video': item['source_video'], 'source_filename': item['title'],
                    'source_frame_number': item['source_frame_number'], 'mask_index': item['mask_index'],
                    'drive_file_id': item['id'], 'size_bytes': item['size'],
                    'source_url': 'https://drive.google.com/file/d/' + item['id'] + '/view',
                    'label_origin': 'publisher_frame_mask', 'anomaly_subtype': '',
                    'review_status': 'publisher_labeled_not_independently_reviewed',
                })
    rng.shuffle(rows)
    (root / 'selection.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (root / 'source/indexing_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (root / 'source/split_recordings.json').write_text(json.dumps(groups, indent=2), encoding='utf-8')
    with (root / 'labels.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'selected': len(rows), 'counts': {s: dict(Counter(r['class_name'] for r in rows if r['split'] == s)) for s in QUOTAS},
                      'recordings': len({r['source_video'] for r in rows})}, indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/intersection_anomalies'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    select(args.root, args.seed)
