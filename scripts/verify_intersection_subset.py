"""Verify downloaded aerial images, source labels, split isolation and counts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from prepare_intersection_subset import QUOTAS


def verify(root: Path) -> dict:
    rows = json.loads((root / 'selection.json').read_text())
    if len(rows) != 1500 or len({r['sample_id'] for r in rows}) != 1500:
        raise ValueError('Expected 1,500 distinct sample IDs')
    masks = {}
    hashes = {}
    decoded_hashes = {}
    records = defaultdict(set)
    frame_numbers = defaultdict(list)
    sizes = Counter()
    for split, quotas in QUOTAS.items():
        counts = Counter(r['label'] for r in rows if r['split'] == split)
        if dict(counts) != quotas:
            raise ValueError('Incorrect class counts: ' + split)
    checksums = []
    for i, row in enumerate(rows, 1):
        path = root / row['image_path']
        if path.stat().st_size != row['size_bytes']:
            raise ValueError('Incorrect file size: ' + row['sample_id'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate image bytes: ' + row['sample_id'])
        hashes[digest] = row['sample_id']
        with Image.open(path) as im:
            im.load()
            if im.format != 'JPEG' or min(im.size) < 480:
                raise ValueError('Unexpected image format or resolution')
            sizes[str(im.size)] += 1
            pixel_digest = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            if pixel_digest in decoded_hashes:
                raise ValueError('Duplicate decoded image: ' + row['sample_id'])
            decoded_hashes[pixel_digest] = row['sample_id']
        video = row['source_video']
        if video not in masks:
            masks[video] = np.load(root / 'source/masks' / (video + '.npy'), allow_pickle=False)
        if int(masks[video][row['mask_index']]) != row['label']:
            raise ValueError('Source label mismatch: ' + row['sample_id'])
        if row['source_frame_number'] != 5 * row['mask_index']:
            raise ValueError('Frame alignment mismatch')
        records[video].add(row['split'])
        frame_numbers[video].append(row['source_frame_number'])
        checksums.append({'sample_id': row['sample_id'], 'sha256': digest, 'decoded_rgb_sha256': pixel_digest})
        if i % 250 == 0:
            print(f'Verified {i}/1500 images', flush=True)
    if any(len(splits) != 1 for splits in records.values()):
        raise ValueError('Recording appears in multiple splits')
    spacings = [b-a for numbers in frame_numbers.values()
                for a, b in zip(sorted(numbers), sorted(numbers)[1:])]
    if min(spacings) < 60:
        raise ValueError('Images sampled too closely within a recording')
    actual = set(p.name for p in (root / 'images').glob('*.jpg'))
    expected = set(Path(r['image_path']).name for r in rows)
    if actual != expected:
        raise ValueError('Unexpected or missing image files')
    report = {
        'status': 'passed', 'images': len(rows), 'normal': 1200, 'anomaly': 300,
        'source_recordings': len(records), 'recording_disjoint_splits': True,
        'identical_file_duplicates': 0, 'identical_decoded_image_duplicates': 0,
        'source_label_mismatches': 0, 'minimum_source_frame_spacing': min(spacings),
        'resolutions': dict(sizes), 'bytes': sum(r['size_bytes'] for r in rows),
        'limits': ['Publisher binary labels; not an independent review of all scenes.',
                   'No verified anomaly subtypes or anomaly-object bounding boxes.',
                   'Recording-disjoint; locations may overlap across splits.',
                   'Video-derived anomalies are not always decidable from one still.',
                   'This is a new supervised subset, not the official benchmark split.',
                   'Exact duplicate checks do not establish event independence.'],
    }
    (root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (root / 'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
    # A separate overview only: labels here never alter training images.
    chosen = []
    for split in QUOTAS:
        for label in [0, 1]:
            seen = set()
            for r in rows:
                if r['split'] == split and r['label'] == label and r['source_video'] not in seen:
                    chosen.append(r)
                    seen.add(r['source_video'])
                    if len(seen) == 3:
                        break
    sheet = Image.new('RGB', (1200, 6 * 254), '#101820')
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(chosen):
        x, y = (i % 3)*400, (i // 3)*254
        with Image.open(root / r['image_path']) as im:
            thumb = ImageOps.contain(im, (392, 220))
            sheet.paste(thumb, (x+4, y+2))
        draw.text((x+6, y+225), r['split'] + ' | ' + r['class_name'] + ' | ' + r['source_video'], fill='white')
    sheet.save(root / 'contact_sheet.jpg', quality=90)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/intersection_anomalies'))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
