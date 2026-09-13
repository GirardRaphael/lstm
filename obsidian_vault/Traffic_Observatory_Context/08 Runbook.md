---
id: toc-08-runbook
type: runbook
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Runbook (local research workbench)

## Environment

```powershell
# Interpreter (outside OneDrive)
C:\Users\azulr\.venvs\traffic_lstm\Scripts\python.exe

cd C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project
$env:PYTHONPATH = "src"
```

## Launchers

```powershell
.\run_app.ps1      # Streamlit
.\notebook.ps1     # JupyterLab
.\train.ps1        # retrain + figures + educational vault
.\test.ps1         # pipeline tests wrapper
```

## Baseline checks on main

```powershell
$env:PYTHONPATH = "src"
python tests/test_pipeline.py
```

## Obsidian vaults

| Vault | Path | Purpose |
| --- | --- | --- |
| Educational | `obsidian_vault/Traffic_LSTM_Brain` | neuron/gate explanations (generated) |
| Project context | `obsidian_vault/Traffic_Observatory_Context` | build state / tasks / evidence |

Open each with **Open folder as vault**. Start educational at `00 Start Here`; start context at [[00 Start Here]].

## Recovery notes

- Prefer project `.venv` if present; else `~\.venvs\traffic_lstm`.
- Retraining overwrites figures/vault unless `--vault-dir` / `--no-figures`.
- Do not compact Docker disk without admin and user intent.
