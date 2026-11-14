# Real aerial intersection anomaly dataset

The local dataset prepared for this project contains 1,500 original aerial photographs selected from
[UIT-ADrone](https://uit-together.github.io/datasets/UIT-ADrone/):
1,200 publisher-labeled normal frames and 300 publisher-labeled anomaly frames.
The images show real roundabout traffic in Ho Chi Minh City, Vietnam. They are
not generated illustrations, and they are not 1,500 independent incidents.

## Where to find it

- Dataset: `data/intersection_anomalies/`
- Original JPEGs: `images/`
- Image-level labels and source references: `labels.csv`
- Selection metadata: `selection.json`
- Original binary annotation arrays: `source/masks/`
- File and decoded-image checksums: `checksums.json`
- Integrity results: `verification.json`
- Separate labeled preview: `contact_sheet.jpg`
- Standalone gallery: run `./run_app.ps1 -Images`, then open
  [the local gallery](http://127.0.0.1:8501/).

The entire dataset directory, including images, labels, masks and selection
metadata, is intentionally kept outside Git. A fresh clone includes the tools
and this specification but not the prepared dataset. Restore that local directory
to browse the existing subset. The labels are in CSV and JSON;
no class labels, captions, or annotation boxes are burned into the training images.

## Class balance and partitions

| Partition | Normal (0) | Anomaly (1) | Total |
| --- | ---: | ---: | ---: |
| Train | 840 | 210 | 1,050 |
| Validation | 180 | 45 | 225 |
| Test | 180 | 45 | 225 |
| Total | 1,200 | 300 | 1,500 |

The 80/20 ratio describes the two classes. The separate training/validation/test
ratio is 70/15/15. Entire source recordings stay within one partition.
The subset contains images from 51 recordings; recording IDs and partition
assignments are retained in `source/split_recordings.json`.

## What the labels establish

The binary label is copied from the publisher's frame-level annotation array.
The source describes traffic anomalies including wrong-way movement, pedestrians
in the roadway, and vehicles on sidewalks. This subset retains the binary labels;
it does **not** provide a verified per-image violation subtype or identify the
responsible object with a bounding box.

The `anomaly_subtype` field is deliberately empty and `review_status` records
that these are publisher labels, not an independent review of every image.
A normal label means normal under the source annotation scheme, not proof that
every person in the image complies with local traffic law.

The requested signal violations, blocked crossings, and other specific anomalies
remain possible extensions. Their presence and counts have not been verified
in this subset. A single still cannot establish signal-entry timing, speed,
or movement history; use the corresponding video or a sequence when that
evidence is required.

## Selection and label alignment

Selection uses seed 42 and only the publisher's labeled test-image pool. This is
a newly constructed **supervised** subset and is not the official unsupervised
benchmark partition. Do not compare results as if the original benchmark split
had been preserved.

The public folder listing is capped at 1,000 images per recording. The local
catalog therefore contains 46,534 candidate image records, not the full source
dataset. Larger folders are biased toward the files returned first by the provider.

Public image filenames use source-frame numbers 0, 5, 10, and so on. The mask
index is the numeric filename divided by five. Complete folders were checked
against the entire zero-based sequence; every partial folder was checked for
divisibility, index bounds, and its expected terminal frame. These checks are
saved in `source/indexing_audit.json`.

Frames within three mask positions of an annotation transition are excluded.
Selected frames within each recording are at least 60 source-frame numbers
apart (nominally two seconds at the source's stated 30 fps). This reduces
adjacent-frame redundancy but does not make the selected images independent
events. Locations and backgrounds may still overlap between partitions.

## Quality checks and interpretation

The verification script checks exact class counts, image readability, original
file sizes, source-label alignment, unique IDs, recording isolation, temporal
spacing, SHA-256 file duplicates, and duplicates of decoded RGB pixels.
The separate contact sheet supports visual spot checks. Exact-duplicate checks
do not establish absence of visually similar scenes.

A classifier predicting normal for every image would score **80% accuracy**.
Report anomaly precision, anomaly recall, balanced accuracy, and a confusion
matrix. Evaluate on additional locations before claiming generalization.

The current traffic-volume LSTM accepts numerical time series. It has not been
trained on these images. Image classification, or video anomaly detection for
motion-dependent labels, is separate model work.

## Reproduce and verify

The following commands require the locally retained `source/frame_index.json`
and `source/masks/*.npy` plus the selected original JPEGs for verification. These
inputs are not included in Git; the preparation script does not crawl the public
dataset automatically. With the local inputs restored and `./setup.ps1` complete,
the selection can be regenerated and verified:

```powershell
.venv/Scripts/python.exe scripts/prepare_intersection_subset.py
.venv/Scripts/python.exe scripts/verify_intersection_subset.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_intersection_subset.py
```

The download helper consumes temporary authenticated download references from
the Google Drive connector; expired references must be refreshed from each
sample's stable `source_url`. Temporary download references are removed after
the completed download. Stable source file IDs remain in the manifest.

## Provenance

Original project: [UIT-ADrone](https://uit-together.github.io/datasets/UIT-ADrone/).
Original public download:
[Google Drive dataset folder](https://drive.google.com/drive/folders/1ygWcuOZ7kEwS-rC35gWv6f6qeNx0BHyg).

T. M. Tran, T. N. Vu, T. V. Nguyen, and K. Nguyen,
“UIT-ADrone: A Novel Drone Dataset for Traffic Anomaly Detection,”
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
2023. DOI: [10.1109/JSTARS.2023.3285905](https://doi.org/10.1109/JSTARS.2023.3285905).

Original images and labels retain their source rights; this project assigns no
new license to them. Files were downloaded without image edits. The changes are
selection, grouping, split assignment, and added provenance/checksum metadata.
