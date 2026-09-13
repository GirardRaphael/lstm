# Pick up here

Current continuation — 2026-09-13 (M0 recovery)

**Authoritative build state** lives in the Obsidian project-context vault, not in this file:

- Open folder as vault → `obsidian_vault/Traffic_Observatory_Context`
- Start at `00 Start Here.md`, then `09 Next Session.md`

| | |
| --- | --- |
| Real project path | `C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project` |
| GitHub | https://github.com/GirardRaphael/lstm |
| Verified tip of `main` | `a652a04` |
| Working branch | `observatory/m0-recover` |
| Educational vault | `obsidian_vault/Traffic_LSTM_Brain` |
| Implementation brief | [MASTER_BUILD_PROMPT.md](MASTER_BUILD_PROMPT.md) |
| Product plan (from readiness tip) | [ROAD_PRODUCT_PLAN.md](ROAD_PRODUCT_PLAN.md) |
| Audit | [AUDIT.md](AUDIT.md) |

## Verified now (this session)

- Pipeline tests on `a652a04`: **16 / 16 passed**
- Python 3.12.10 + TF CPU 2.21.0 at `~\.venvs\traffic_lstm`
- Remote branches present: `main`, `codex/lstm-reliability-fixes`, `codex/street-shadow-readiness`
- **Missing:** `codex/temporal-pipeline-v2` / commit `646d3a7` (blocked — see vault `07 Risks and Blockers`)

## Labels

Do not treat the superseded historical handoff below, or prior “18/21/12 checks” notes, as current verification. Those counts belong to trees not present on this machine.

## Next owner action

Provide the unpublished `codex/temporal-pipeline-v2` checkout, git bundle, or patch **or** explicitly authorize a fresh pipeline-v2 implementation that does not claim identity with `646d3a7`.

```powershell
cd C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project
$env:PYTHONPATH = "src"
~\.venvs\traffic_lstm\Scripts\python.exe tests/test_pipeline.py
.\run_app.ps1
```

---

## Historical handoff — 2026-09-12

The following is preserved from the previous session and is **superseded** by the project-context vault and the section above. Its test counts and conclusions are not current.

### 30-second orientation

| | |
| --- | --- |
| Project | `C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project` |
| GitHub | https://github.com/GirardRaphael/lstm |
| Published page | https://claude.ai/code/artifact/6a8193c3-85bb-408f-9317-f03f7082b089 |
| Python | 3.12 — TensorFlow has no 3.13 wheels on Windows |
| Virtualenv | `C:\Users\azulr\.venvs\traffic_lstm` — outside OneDrive |

```powershell
.\run_app.ps1     # Streamlit workbench
.\notebook.ps1    # JupyterLab
.\train.ps1       # retrain + figures + vault
.\test.ps1        # pipeline tests
```

Everything else needs `$env:PYTHONPATH = "src"` first.

Open the educational vault with **Open folder as vault** → `obsidian_vault/Traffic_LSTM_Brain` → start at `00 Start Here`.

Stored motorway comparison figures (historical artifacts, not re-run in M0): XGBoost+calendar MAE ~154.3; LSTM+calendar ~201.4; univariate LSTM ~228.8; persistence ~585.6. See `models/*_artifacts.json` and the educational vault comparison note.
