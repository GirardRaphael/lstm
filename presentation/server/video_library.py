"""Allowlisted local video examples and single-range HTTP delivery helpers."""
from __future__ import annotations

import json
import hashlib
import mimetypes
from pathlib import Path
import re

from learning_lab import LabError, clean_id, validate_source_url


def video_file_fingerprint(path: Path) -> str:
    """Match the browser's stable, sampled-byte identity without reading the whole clip."""
    size = path.stat().st_size
    block_size = 64 * 1024
    starts = (0, max(0, (size - block_size) // 2), max(0, size - block_size))
    digest = hashlib.sha256(f'video-size:{size};'.encode())
    with path.open('rb') as source:
        for start in starts:
            source.seek(start)
            digest.update(source.read(min(block_size, size - start)))
    return digest.hexdigest()


class VideoLibrary:
    def __init__(self, root: Path):
        self.directory = (Path(root) / 'data/signalized_traffic').resolve()

    def _entries(self) -> list[dict]:
        manifest = self.directory / 'manifest.json'
        if not manifest.is_file():
            return []
        if manifest.stat().st_size > 1_000_000:
            raise LabError('The local video manifest is too large.', 503)
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        entries = data.get('videos') if isinstance(data, dict) else None
        if not isinstance(entries, list) or len(entries) > 40:
            raise LabError('The local video manifest is invalid.', 503)
        result, seen = [], set()
        for entry in entries:
            if not isinstance(entry, dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', str(entry.get('id', ''))):
                raise LabError('A local video identifier is invalid.', 503)
            identifier = entry['id']
            if identifier in seen:
                raise LabError('Local video identifiers must be unique.', 503)
            seen.add(identifier)
            filename = clean_id(entry.get('filename'), 'Video filename', 240)
            source = (self.directory / filename).resolve()
            if not source.is_relative_to(self.directory) or source.suffix.lower() not in ('.mp4', '.webm', '.ogg'):
                raise LabError('The local video path is invalid.', 503)
            if not source.is_file():
                continue
            public = {
                'id': identifier, 'title': clean_id(entry.get('title'), 'Video title', 240),
                'recording_id': clean_id(entry.get('recording_id'), 'Recording ID'),
                'source_url': validate_source_url(entry.get('source_url')),
                'license': clean_id(entry.get('license'), 'Video license', 512),
                'description': clean_id(entry.get('description'), 'Video description', 2000),
                'video_url': f'/api/learning/video/{identifier}',
                'upload_fingerprint': video_file_fingerprint(source),
            }
            if entry.get('attribution'):
                public['attribution'] = clean_id(entry['attribution'], 'Video attribution', 1000)
            if entry.get('license_url'):
                public['license_url'] = validate_source_url(entry['license_url'])
            if entry.get('changes'):
                public['changes'] = clean_id(entry['changes'], 'Video adaptation details', 2000)
            result.append({'public': public, 'path': source})
        return result

    def examples(self) -> dict:
        return {'ok': True, 'videos': [entry['public'] for entry in self._entries()]}

    def recording_sources(self, group_ids: set[str]) -> list[dict]:
        return [entry['public'] for entry in self._entries() if entry['public']['recording_id'] in group_ids]

    def canonical_recording(self, recording_id: str) -> str:
        if re.fullmatch(r'video-[a-f0-9]{64}', recording_id):
            fingerprint = recording_id[6:]
            for entry in self._entries():
                if entry['public']['upload_fingerprint'] == fingerprint:
                    return entry['public']['recording_id']
        return recording_id

    def video_path(self, identifier: str) -> tuple[Path, str]:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier):
            raise LabError('Invalid video identifier.')
        for entry in self._entries():
            if entry['public']['id'] == identifier:
                return entry['path'], mimetypes.guess_type(entry['path'].name)[0] or 'application/octet-stream'
        raise LabError('Local example video not found.', 404)


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Resolve a single inclusive byte range; reject invalid/unsatisfiable ranges."""
    if header is None:
        return None
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or not any(match.groups()) or size <= 0:
        raise LabError('Requested video byte range is unavailable.', 416)
    first, last = match.groups()
    # Avoid arbitrarily large integers from a malformed request header.
    if max(len(first), len(last)) > 20:
        raise LabError('Requested video byte range is unavailable.', 416)
    if not first:
        length = int(last)
        if length <= 0:
            raise LabError('Requested video byte range is unavailable.', 416)
        return max(0, size - length), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise LabError('Requested video byte range is unavailable.', 416)
    return start, end
