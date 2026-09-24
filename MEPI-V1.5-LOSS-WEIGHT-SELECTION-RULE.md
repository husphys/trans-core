# MEPI v1.5 validation-only loss-weight selection rule

**Status:** FROZEN  
**Date:** 2026-09-24  
**Scope:** Cross-configuration selection for the 12-pair MEPI v1.5 xLSTM loss-weight sensitivity grid only.

## Primary selection metric

For every numerically valid configuration compute:

```text
VAL_TASK_SCORE = validation_L_electrical + validation_L_LSP
```

Select by the **minimum** `VAL_TASK_SCORE`. Both component definitions are
identical for every grid configuration, so their sum is directly comparable.

`weighted_validation_total_loss` is prohibited for cross-configuration
ranking because `lambda3` and `lambda4` change its definition. It remains a
required within-configuration training, stopping, and reporting quantity.

## Numerical eligibility

Exclude a configuration from selection if any required validation loss,
physical validation metric, prediction, target, weighted contribution, or
`VAL_TASK_SCORE` is NaN or infinite. Record `numerical_failure` and
`eligible_for_selection` for every configuration.

If multiple eligible configurations have exactly the same minimum
`VAL_TASK_SCORE`, retain them as tied co-best candidates. Do not invent an
unfrozen secondary tie-breaker.

## Required comparison fields

Each configuration must report:

- efficiency MAE, RMSE, and R2;
- P_loss MAE, RMSE, and R2;
- LSP_raw MAE, RMSE, R2, and MAPE_PERCENT;
- validation_L_electrical, validation_L_LSP, validation_L_physics, and validation_L_UQ;
- weighted_validation_total_loss;
- VAL_TASK_SCORE;
- best_epoch, stop_epoch, and epochs_completed;
- numerical_failure and eligible_for_selection.

## Baseline reuse and isolation

The completed `(lambda3, lambda4) = (0.3, 0.20)` v1.5 baseline may replace
that grid execution only after a machine-readable audit proves identical
seed-42 initialization, selected representation checkpoint, dataset/splits,
preprocessing, optimizer, fixed learning rate, weight decay, batch size,
100-epoch maximum, early stopping, AMP behavior, loss definitions, and exact
continuation provenance. No other fine-tuned configuration may initialize a
grid run.

The test subset remains inaccessible throughout selection. This rule changes
no MEPI-FROZEN-PROTOCOL v1.5 scientific training setting.
