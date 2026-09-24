"""Regression metrics used consistently across candidate reruns."""

from __future__ import annotations

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("Target and prediction must be non-empty and have equal shape")
    error = estimate - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    total = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / total) if total > 0.0 else float("nan")
    return {"normalized_mae": mae, "rmse": rmse, "r2": r2}

