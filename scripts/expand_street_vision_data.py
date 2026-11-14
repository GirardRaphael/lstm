"""Expand training recordings only; keep all original examples and held-out rows."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from PIL import Image
from prepare_intersection_subset import candidates
from download_intersection_files import download


def prepare(source: Path, target: Path):
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Expansion must use a separate destination")
    with (source / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        original = list(csv.DictReader(handle))
    train_groups = {row["source_video"] for row in original if row["split"] == "train"}
    pools, _ = candidates(source)
    original_ids = {row["drive_file_id"] for row in original}
    additions = sorted([row for group in train_groups for row in pools[group] if row["id"] not in original_ids],
                       key=lambda row: (row["source_video"], row["source_frame_number"]))[:1500]
    new_rows = []
    for item in additions:
        sample_id = hashlib.sha256((item["source_video"] + "/" + item["title"]).encode()).hexdigest()[:16]
        new_rows.append({"sample_id": sample_id, "image_path": "images/" + sample_id + ".jpg", "split": "train",
            "label": item["label"], "class_name": "anomaly" if item["label"] else "normal", "source_dataset": "UIT-ADrone",
            "source_partition": "test", "source_video": item["source_video"], "source_filename": item["title"],
            "source_frame_number": item["source_frame_number"], "mask_index": item["mask_index"], "drive_file_id": item["id"],
            "size_bytes": item["size"], "source_url": "https://drive.google.com/file/d/" + item["id"] + "/view",
            "label_origin": "publisher_frame_mask", "anomaly_subtype": "", "review_status": "publisher_labeled_not_independently_reviewed"})
    rows = original + new_rows
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("An expansion example duplicates an original")
    for group in train_groups:
        frames = sorted(int(row["source_frame_number"]) for row in rows if row["source_video"] == group)
        if any(b - a < 60 for a, b in zip(frames, frames[1:])):
            raise ValueError("Expanded training frames violate the original temporal spacing")
    target.mkdir(parents=True, exist_ok=True)
    (target / "images").mkdir(exist_ok=True)
    for row in original:
        destination = target / row["image_path"]
        if not destination.exists():
            try:
                os.link(source / row["image_path"], destination)
            except OSError:
                shutil.copy2(source / row["image_path"], destination)
    with (target / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(rows)
    (target / "additions.json").write_text(json.dumps(new_rows, indent=2), encoding="utf-8")
    summary = {"original_count": len(original), "new_train_count": len(new_rows), "total": len(rows),
               "new_classes": dict(Counter(row["class_name"] for row in new_rows)),
               "split_counts": {split: dict(Counter(row["class_name"] for row in rows if row["split"] == split)) for split in ("train", "validation", "test")},
               "train_recordings": sorted(train_groups), "heldout_rows_unchanged": True, "minimum_source_frame_spacing": 60,
               "class_balance_note": "All remaining spaced training frames were added. Training balance changes; held-out80/20 balance is unchanged.",
               "original_labels_sha256": hashlib.sha256((source / "labels.csv").read_bytes()).hexdigest()}
    (target / "expansion_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def verify(source: Path, target: Path):
    with (source / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        original = list(csv.DictReader(handle))
    with (target / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    original_heldout = {row["sample_id"]: row for row in original if row["split"] != "train"}
    heldout = {row["sample_id"]: row for row in rows if row["split"] != "train"}
    if original_heldout != heldout:
        raise ValueError("Held-out rows changed")
    for index, row in enumerate(rows):
        path = target / row["image_path"]
        if not path.exists() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("Missing/incomplete image: " + row["sample_id"])
        with Image.open(path) as image:
            image.verify()
        if (index + 1) % 500 == 0:
            print(f"Verified {index + 1}/{len(rows)} images", flush=True)
    print(json.dumps({"verified_images": len(rows), "unchanged_heldout_rows": len(heldout)}), flush=True)


def download_additions(target: Path):
    root = target.resolve()
    additions = json.loads((root / "additions.json").read_text(encoding="utf-8"))
    tasks = [{"url": "https://drive.google.com/uc?export=download&id=" + row["drive_file_id"],
              "path": row["image_path"], "size": row["size_bytes"]} for row in additions]
    failures = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(download, task, root): task["path"] for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception:
                failures.append(futures[future])
            if count % 100 == 0 or count == len(tasks):
                print(f"Processed {count}/{len(tasks)} new frames; failures={len(failures)}", flush=True)
    result = {"downloaded": len(tasks) - len(failures), "failed": failures, "registered": len(tasks)}
    (root / "download_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} public downloads failed; details are in download_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.source, args.target)
    elif args.download:
        download_additions(args.target)
    else:
        prepare(args.source, args.target)
