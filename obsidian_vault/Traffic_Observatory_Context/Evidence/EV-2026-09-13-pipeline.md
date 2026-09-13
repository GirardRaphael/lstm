---
id: ev-2026-09-13-pipeline
type: evidence
status: verified
owner: principal
updated_utc: 2026-09-13T13:50:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# EV-2026-09-13-pipeline

| Field | Value |
| --- | --- |
| Command | `$env:PYTHONPATH='src'; C:\Users\azulr\.venvs\traffic_lstm\Scripts\python.exe tests/test_pipeline.py` |
| Environment | Windows 10.0.26200, Python 3.12.10, TF CPU 2.21.0 |
| Commit | `a652a04` (`main` tip at recovery) |
| Date (local) | 2026-09-13 |
| Outcome | **16 passed, 0 failed** |
| Notes | Matches historical HANDOFF count of 16 on pre-readiness main; later continuity notes claiming 18 pipeline checks refer to unpublished v2 tree |
