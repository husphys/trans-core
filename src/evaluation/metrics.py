from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(target: np.ndarray, prediction: np.ndarray, near_zero: bool = False) -> dict[str, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    absolute = np.abs(prediction - target)
    denominator = np.abs(target) + 1e-12
    if near_zero:
        percentage = 200.0 * absolute / (np.abs(target) + np.abs(prediction) + 1e-12)
        percentage_name = "smape"
    else:
        percentage = 100.0 * absolute / denominator
        percentage_name = "mape"
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        percentage_name: float(np.mean(percentage)),
        "err95": float(np.quantile(percentage, 0.95)),
    }

