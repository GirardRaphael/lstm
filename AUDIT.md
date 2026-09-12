# Repository audit

> Follow-up: [ROAD_PRODUCT_PLAN.md](ROAD_PRODUCT_PLAN.md) documents additional
> look-ahead and validation-preprocessing risks uncovered after this first audit.
> The checks below are scoped smoke tests, not evidence of a fully leakage-free
> pipeline or permission to deploy on streets.

Audit performed on 2026-09-12 with Python 3.12 and TensorFlow CPU 2.21.

## Verified

- The chronological train/test split and train-only scaler prevent the main
  forms of time-series leakage covered by the project.
- All six committed Keras models load and produce their documented output
  shapes, including multivariate and multi-horizon variants.
- The saved-model CLI produces a forecast from the bundled dataset.
- A fresh small LSTM can train and predict.
- The Streamlit data page renders with the bundled 48,204-row dataset.
- The XGBoost benchmark executes from a clean requirements installation.

## Defects fixed

1. Saved artifacts contained an absolute path from the author's Windows PC.
   A clone on another machine could not find the bundled dataset. Dataset
   paths are now stored relative to the repository and legacy paths are
   resolved by filename.
2. `xgboost` was required by a documented command but absent from
   `requirements.txt`.
3. The PowerShell launchers ignored the documented project `.venv`. They now
   prefer `.venv`, retain the previous environment as a fallback, explain how
   to repair a missing environment, and propagate command failures.
4. Invalid model settings such as zero sequence length or 100% dropout now
   fail before training begins.
5. Continuous integration now checks the pipeline, all saved models, and the
   Streamlit application's first page.

## Model conclusion

For the current motorway holdout period, the strongest stored result is the
multivariate XGBoost model (MAE about 154.5 vehicles/hour). The univariate
XGBoost rerun scored about 173.5, while the univariate LSTM artifact scores
about 228.8. The LSTM is therefore valuable as a sequence-learning and model
introspection project, but it is not the best production forecaster in the
current experiment.

On the bike-rental dataset, the conclusion is different: the multivariate
LSTM (MAE about 38.5) narrowly beats its XGBoost comparison (about 38.6).
That contrast is a useful and honest finding: model choice depends on the
dataset and evaluation period.

## Recommended next upgrades

1. Add rolling-origin evaluation with several folds. One holdout period can
   make a model look unusually good or bad because of seasonal regime changes.
2. Add uncertainty intervals using quantile loss, conformal prediction, or an
   ensemble. A traffic operator needs both a forecast and its confidence.
3. Tune architectures systematically rather than adding layers: compare GRU,
   one-layer LSTM, 24/48/72/168-hour windows, learning rates, and batch sizes.
4. Add truly forward-looking inputs such as forecast weather, planned events,
   road closures, and holiday calendars. Do not use future observations that
   would be unavailable at prediction time.
5. Record package versions and experiment metadata (Git commit, seed, dataset
   checksum) for exact repeatability.
6. Add drift monitoring and a fallback to XGBoost or a seasonal baseline
   before connecting any forecast to a control system.
7. Normalize the notebook cell IDs during its next rebuild; current notebook
   tools accept it, but `nbformat` warns that missing IDs may become an error in
   a future version.
