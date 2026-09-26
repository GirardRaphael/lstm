"""Separate calibration residuals from evaluation; no guarantee under time shift."""
import math
import numpy as np


def calibrate(actual, predicted, alpha=0.1):
    actual,predicted = np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Calibration actuals and predictions must have identical shapes")
    residuals = np.abs(actual - predicted)
    if residuals.ndim != 1 or len(residuals) < 30 or not np.isfinite(residuals).all():
        raise ValueError("Need at least 30 finite aligned calibration residuals")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    if rank > len(residuals):
        raise ValueError("Insufficient calibration samples for requested alpha")
    return {"radius": float(np.sort(residuals)[rank - 1]), "alpha": alpha,
            "n": len(residuals), "method": "absolute-residual split calibration",
            "limitation": "nominal coverage is not guaranteed under temporal dependence or shift"}


def bounds(predicted, calibration):
    predicted = np.maximum(0, np.asarray(predicted, dtype=float))
    return np.maximum(0, predicted - calibration["radius"]), predicted + calibration["radius"]


def interval_metrics(actual, predicted, calibration):
    actual = np.asarray(actual, dtype=float)
    low, high = bounds(predicted, calibration)
    covered = (actual >= low) & (actual <= high)
    return {"coverage": float(covered.mean()), "mean_width": float((high-low).mean()),
            "n": len(actual), "nominal_coverage": 1-calibration["alpha"]}
