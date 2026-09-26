# MEPI v1.5 post-hoc final-test metrics

Status: **POST-HOC REPORTING ONLY**

No training, model selection, hyperparameter change, checkpoint change, or uncertainty calibration was performed.

## Guard state

```text
POSTHOC_ANALYSIS_ONLY = TRUE
TRAINING_PERFORMED = FALSE
MODEL_SELECTION_PERFORMED = FALSE
HYPERPARAMETER_CHANGED = FALSE
CHECKPOINT_CHANGED = FALSE
FINAL_TEST_LEDGER_MODIFIED = FALSE
```

## Metrics

| Target | MAE | RMSE | R2 | MAPE (%) | Err95 (%) |
|---|---:|---:|---:|---:|---:|
| Efficiency (percentage points) | 1.1456658998 | 1.48337475849 | 0.981763696937 | — | 4.43847570723 |
| Core loss (W) | 0.0136925429242 | 0.0164594178334 | 0.970616638689 | — | 25.2739479617 |
| LSP | 0.0279506840434 | 0.0363657659508 | 0.916250935697 | 4.78804757549 | 13.018527699 |

Err95 uses epsilon `1e-12` from `src/evaluation/metrics.py`.

Predicted LSP sigma is an uncalibrated model output; it is not a confidence interval or calibrated uncertainty.

LSP is a dimensionless Arrhenius-informed relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or measured degradation.
