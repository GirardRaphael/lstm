"""Metrics, and the baselines that make those metrics mean something.

A MAE of 312 vehicles is meaningless on its own. It only becomes an argument
once you can say "and the naive baseline scores 430, so the network is 27%
better". Both naive baselines below are free to compute and are the first
thing a reviewer will ask for.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, RMSE and MAPE for one horizon."""
    actual = np.asarray(actual, dtype="float64").ravel()
    predicted = np.asarray(predicted, dtype="float64").ravel()
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    # Guard against the handful of hours with near-zero traffic.
    mask = np.abs(actual) > 1.0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": int(len(actual))}


def _window_ends(series_length: int, sequence_length: int, horizon: int,
                 indices: np.ndarray | None) -> np.ndarray:
    """The window end-indices the baselines must line up with.

    Passing `indices` matters as soon as some windows are dropped (see
    `TrainingConfig.drop_gapped_windows`): the baselines have to score exactly
    the windows the model scored, not every position in the series.
    """
    if indices is not None:
        return np.asarray(indices, dtype="int64")
    return np.arange(sequence_length, series_length - horizon + 1, dtype="int64")


def naive_persistence(series: np.ndarray, sequence_length: int, horizon: int,
                      indices: np.ndarray | None = None) -> np.ndarray:
    """Baseline 1 - "the next hour looks like the last hour we saw"."""
    series = np.asarray(series, dtype="float64").ravel()
    ends = _window_ends(len(series), sequence_length, horizon, indices)
    return series[ends - 1]


def naive_seasonal(series: np.ndarray, sequence_length: int, horizon: int,
                   indices: np.ndarray | None = None) -> np.ndarray:
    """Baseline 2 - "the same hour yesterday" (a strong baseline for traffic)."""
    series = np.asarray(series, dtype="float64").ravel()
    ends = _window_ends(len(series), sequence_length, horizon, indices)
    targets = ends + horizon - 1
    # Fall back to persistence for the first day, where yesterday is unknown.
    return np.where(targets - 24 >= 0, series[np.maximum(targets - 24, 0)], series[ends - 1])


def compare_against_baselines(
    actual: np.ndarray,
    predicted: np.ndarray,
    bridged_series: np.ndarray,
    sequence_length: int,
    horizon: int,
    indices: np.ndarray | None = None,
) -> dict:
    """Model metrics side by side with both naive baselines."""
    model = regression_metrics(actual, predicted)
    persistence = regression_metrics(
        actual, naive_persistence(bridged_series, sequence_length, horizon, indices)
    )
    seasonal = regression_metrics(
        actual, naive_seasonal(bridged_series, sequence_length, horizon, indices)
    )
    best_naive = min(persistence["mae"], seasonal["mae"])
    return {
        "lstm": model,
        "naive_persistence": persistence,
        "naive_same_hour_yesterday": seasonal,
        "improvement_over_best_naive_pct": round(
            (best_naive - model["mae"]) / best_naive * 100, 2
        ),
    }


def training_diagnosis(history: dict, patience: int = 5) -> dict:
    """Did the run plateau, overfit, or simply run out of epochs?

    Worth computing automatically, because the three look identical in a final
    metric and call for opposite fixes:

    * **capped** - the epoch limit stopped it while it was still improving.
      Train longer.
    * **overfit** - validation loss bottomed out early and then climbed.
      More patience cannot help: `restore_best_weights` hands back the same
      epoch either way. Regularise, or shrink the model.
    * **plateaued** - it converged. The result is what the setup can do.
    """
    val = [float(v) for v in history.get("val_loss", [])]
    if len(val) < 3:
        return {"verdict": "unknown", "detail": "not enough epochs to judge"}

    best = min(val)
    best_epoch = val.index(best) + 1
    total = len(val)
    final = val[-1]
    rise = (final - best) / best * 100 if best else 0.0

    if best_epoch >= total - 1:
        verdict = "capped"
        detail = ("still improving at epoch {} of {} - raise the epoch limit"
                  .format(best_epoch, total))
    elif best_epoch <= max(2, total // 3) and rise > 3.0:
        verdict = "overfit"
        detail = ("best at epoch {} of {}, then validation loss rose {:.1f}% - "
                  "more patience would restore the same weights; regularise instead"
                  .format(best_epoch, total, rise))
    else:
        verdict = "plateaued"
        detail = ("best at epoch {} of {}, validation loss flat afterwards ({:+.1f}%)"
                  .format(best_epoch, total, rise))

    return {"verdict": verdict, "detail": detail, "best_epoch": best_epoch,
            "epochs_run": total, "best_val_loss": best,
            "final_vs_best_pct": round(rise, 2)}


def traffic_level(volume: float, low: int = 2000, high: int = 4000) -> tuple[str, str]:
    """Turn a number of vehicles into something a dispatcher can act on."""
    if volume < low:
        return "LOW", "Free flow"
    if volume < high:
        return "MODERATE", "Steady traffic"
    return "HIGH", "Congestion risk"
