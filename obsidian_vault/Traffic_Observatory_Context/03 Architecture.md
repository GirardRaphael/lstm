---
id: toc-03-architecture
type: architecture
status: active
owner: principal
updated_utc: 2026-09-13T14:05:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Architecture

## Current shape (verified on main)

```mermaid
flowchart LR
  CSV[CSV observations] --> Data[traffic_lstm.data]
  Data --> Train[train / benchmark]
  Train --> Artifacts[models/*.keras + *_artifacts.json]
  Artifacts --> App[Streamlit app]
  Artifacts --> Vault[Traffic_LSTM_Brain export]
  Artifacts --> Predict[predict CLI]
```

Single Python package under `src/traffic_lstm/`, Streamlit UI in `app/`, scripts for reports/experiments. No durable job worker, no relational metadata store, no authenticated API.

## Candidate target (planned — not implemented)

Incremental vertical slice, not a rewrite:

1. Keep ML package as the core library.
2. Add a small Python service + bounded job worker.
3. Persist workspace/dataset/job/model/forecast metadata relationally.
4. Versioned artifact storage with atomic publish.
5. Authenticated operational UI; retain Streamlit for research/education until a dedicated frontend is justified.

## Decision log

See `Decisions/`. First decisions pending recovery of pipeline v2 evidence.
