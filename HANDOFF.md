# Pick up here

Written 2026-09-12. Everything below is verified state, not intention.

---

## 30-second orientation

| | |
| --- | --- |
| Project | `C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project` |
| GitHub | https://github.com/GirardRaphael/lstm — **public**, branch `main`, 6 commits, clean |
| Published page | https://claude.ai/code/artifact/6a8193c3-85bb-408f-9317-f03f7082b089 |
| Python | **3.12** — TensorFlow has no 3.13 wheels on Windows |
| Virtualenv | `C:\Users\azulr\.venvs\traffic_lstm` — **deliberately outside OneDrive**, so 1.5 GB of TensorFlow is not synced to the cloud |

```powershell
.\run_app.ps1     # Streamlit workbench
.\notebook.ps1    # JupyterLab, for the presentation
.\train.ps1       # retrain + figures + vault
.\test.ps1        # 16 tests
```

Everything else needs `$env:PYTHONPATH = "src"` first.

---

## State: all green as of the last commit

| Check | Result |
| --- | --- |
| Tests | **16 / 16 passing** |
| Streamlit app | 0 exceptions across all 6 pages; in-app training verified end to end |
| Obsidian vault | 115 notes, 27 canvases, **990 wikilinks with 0 broken**, 13 image embeds all resolving, 27/27 canvases valid JSON |
| Notebook | 42 cells, every code cell parses |
| Models trained | 9 runs, all with stored artifacts and XGBoost benchmarks |

Open the vault with **Open folder as vault** → `obsidian_vault/Traffic_LSTM_Brain`
→ start at `00 Start Here`.

---

## What the project found

Best results, motorway traffic, MAE in vehicles per hour:

| Model | MAE | Train time |
| --- | --- | --- |
| XGBoost, traffic + calendar | **154.3** | 14s |
| LSTM, traffic + calendar | 201.4 | 688s |
| LSTM, past traffic only *(the one documented in the vault)* | 228.8 | 1,072s |
| Naive "last hour" | 585.6 | — |

Three findings worth leading with:

1. **The LSTM-vs-XGBoost verdict flips between datasets.** XGBoost wins by
   23.4% on traffic; the LSTM wins by 0.3% on bike rentals. Same code, same
   evaluation. The question is empirical, and one dataset cannot settle it.
2. **The calendar is the signal; the weather is noise that costs.** Hour and
   weekday as sine/cosine pairs: 228.8 → 201.4. Adding `rain_1h` / `snow_1h`
   on top: back to 241.1. A control run (same model, stronger dropout, no new
   features) does not move, so it is not about regularisation. XGBoost scores
   the same with and without weather — it ignores the useless column where the
   LSTM overfits it.
3. **28.7% of training windows span a gap in the series.** Real flaw,
   disclosed, and measured: dropping them costs a third of the data and does
   not improve the score.

The vault numbers are trustworthy because `introspect.py` re-implements the
LSTM cell in NumPy from the trained weights and checks itself against Keras on
every export — max absolute difference **1.17e-07**, and a test asserts
`c = f·c_prev + i·g` exactly.

---

## Two things left undone

### 1. Docker disk space — needs an elevated terminal, 5 minutes

C: is at **~2.5 GB free**. 35 GB was deleted *inside* Docker (the vLLM images
and the build cache) but the virtual disk file never shrinks on its own, so
none of it has come back to Windows yet. Docker Desktop was left **closed** so
the file is unlocked and ready.

```powershell
# Right-click Start → Terminal (Administrator)
cd "$env:USERPROFILE\OneDrive\Desktop"
powershell -ExecutionPolicy Bypass -File .\compact_docker_disk.ps1
```

The script uses `diskpart compact vdisk`, the supported Microsoft path. It
deliberately does **not** use WSL's `--set-sparse --allow-unsafe`, which
Microsoft flags as a data-corruption risk.

Your n8n stack was **not** touched — its containers, volumes and images are all
intact. Start Docker Desktop again whenever you need it.

**Still reclaimable:** `docker/model-runner:latest-vllm-cuda` is 19.6 GB and is
held by an internal container belonging to Docker Desktop's *Model Runner*
feature. Its model volume is empty (42 bytes), so the feature is unused —
removing the image means disabling Model Runner in Docker Desktop settings
first. That is a settings change, so it was left for you to decide.

### 2. The experiment nobody has run yet

Every result above shares one architecture and one 24-hour window, which is
close to the ideal case for a tree ensemble. The honest open question is
whether an LSTM starts to pay for itself where a lag table cannot follow:

- several junctions or stations sharing one model
- dependencies longer than the input window
- variable-length or irregular sequences

That is a different experiment, not a bigger network. `--sequence-length 72`
is the cheapest first probe.

---

## Where things live

```
src/traffic_lstm/
  config.py            every hyper-parameter, with the reasoning attached
  data.py              CSV → clean → chronological split → scale → sequences
  features.py          weather + cyclical calendar; target stays column 0
  model.py             the stacked LSTM
  train.py             orchestration, artifacts, CLI
  evaluate.py          MAE/RMSE/MAPE, two naive baselines, training_diagnosis()
  introspect.py        replays the LSTM cell in NumPy — the vault depends on it
  benchmark.py         XGBoost on the same tensors, flattened
  plots.py             every figure
  obsidian_canvas.py   JSON Canvas builders
  obsidian_export.py   generates the whole vault from a trained model

scripts/
  build_notebook.py         regenerates notebooks/traffic_lstm.ipynb
  rebuild_vault.py          regenerates the vault WITHOUT retraining (seconds)
  compare_runs.py           benchmarks every run, ranks them, writes the vault note
  build_report.py           regenerates reports/REPORT.md from the artifacts
  build_web_report.py       inlines figures into reports/web/results.html
  prepare_sample_dataset.py rebuilds the bike-sharing CSV from UCI
```

Nothing in `reports/` or `obsidian_vault/` is hand-written. All of it
regenerates from `models/*_artifacts.json`, so the numbers cannot drift from
the numbers the models actually produced.

---

## Gotchas that cost time once already

- **`train.py --export-vault` overwrites the vault.** Use `--vault-dir` for a
  second dataset, or `scripts/rebuild_vault.py` when you only changed the
  export code — retraining takes 18 minutes, rebuilding takes seconds.
- **A run's figures overwrite the previous run's.** Pass `--no-figures` for
  comparison runs; only the run whose figures you want in the vault should
  write them.
- **`compare_runs.py` groups by source file on purpose.** MAE is in the units
  of the target, so traffic and bike runs must never share a ranking or chart.
- **The `.venv` is outside the project** — if you open this in an IDE, point the
  interpreter at `C:\Users\azulr\.venvs\traffic_lstm\Scripts\python.exe`.
- **Republishing the web page:** `scripts/build_web_report.py` writes
  `reports/web/results.html`; publishing that content to the *existing*
  artifact URL requires the same file path that was published originally, or
  passing the URL explicitly.

---

## To update the published page after new results

```powershell
$env:PYTHONPATH = "src"
python scripts/compare_runs.py      # re-benchmark and re-rank
python scripts/build_report.py      # reports/REPORT.md
python scripts/build_web_report.py  # reports/web/results.html
```

Then edit `reports/web/results.src.html` if the narrative changed — the tables
in it are written by hand, not generated, because the prose around them has to
change with the numbers.
