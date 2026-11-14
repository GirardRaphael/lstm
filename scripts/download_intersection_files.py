"""Materialize connector-issued download URLs without logging credentials.

The task file is temporary and must not be distributed with the dataset.
Uses only the Python standard library; never executes downloaded content.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.request


def download(task: dict, root: Path) -> str:
    target = (root / task['path']).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError('Download target must be inside dataset directory')
    if not task['url'].startswith('https://'):
        raise ValueError('HTTPS required')
    expected = int(task['size'])
    if target.is_file() and target.stat().st_size == expected:
        return task['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(task['url'], timeout=45) as response:
                with part.open('wb') as output:
                    count = 0
                    while chunk := response.read(256 * 1024):
                        count += len(chunk)
                        if count > expected:
                            raise ValueError('Unexpected download size')
                        output.write(chunk)
            if part.stat().st_size != expected:
                raise ValueError('Incomplete download')
            part.replace(target)
            return task['path']
        except Exception:
            if attempt == 2:
                # Do not include exceptions that may embed a signed URL.
                raise RuntimeError('Download failed: ' + task['path']) from None
            time.sleep(attempt + 1)
    raise AssertionError('unreachable')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding='utf-8'))
    root = args.root.resolve()
    paths = [t['path'] for t in tasks]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate download targets')
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, t, root) for t in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except RuntimeError as exc:
                errors.append(str(exc))
            if completed % 50 == 0 or completed == len(tasks):
                print(f'Processed {completed}/{len(tasks)} files; failures={len(errors)}', flush=True)
    if errors:
        raise SystemExit('\n'.join(errors))


if __name__ == '__main__':
    main()
