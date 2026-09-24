# MEPI v1.5 final configuration freeze

**Status:** FROZEN  
**Date:** 2026-09-24  
**Purpose:** Freeze the completed validation-only loss-weight selection and identify the existing best checkpoint for exactly-once final test evaluation.

## Frozen selection

The complete 12-configuration grid was ranked only by the already-frozen rule:

```text
VAL_TASK_SCORE = validation_L_electrical + validation_L_LSP
direction      = minimize
```

`weighted_validation_total_loss` was not used for cross-configuration selection because its definition changes with `lambda3` and `lambda4`.

The unique eligible minimum is:

```text
selected configuration ID = l3_1_l4_0p05
lambda1                   = 1.0
lambda2                   = 1.0
lambda3                   = 1.0
lambda4                   = 0.05
selected_by               = VAL_TASK_SCORE

validation_L_electrical   = 0.0504307309494299
validation_L_LSP          = 0.1519513356335023
VAL_TASK_SCORE            = 0.2023820665829322
```

All 12 grid configurations are recorded as numerically valid and eligible. The grid result declares `l3_1_l4_0p05` the unique minimum with no tie.

## Completed selected run

```text
best epoch                = 91 (zero-based; the 92nd epoch)
completed epochs          = 100
max_epochs                = 100
patience                  = 10
seed                      = 42
selected checkpoint       = experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt
selected checkpoint SHA256 = 0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33
```

This completed train/validation checkpoint is the final selected model. It must be loaded directly for final evaluation. No final refit, retraining, lambda change, seed change, or checkpoint reselection is authorized.

## Frozen evidence hashes

```text
final config SHA256         = 4b18d275be0a56f09371f2fd91d3fe03bba72a7cee6ca732014fbbf043dcdf96
selection-rule SHA256       = 8c4eda9035e7e5e4f75a4b0ccde643c8946fd3b990575ec0323977cb8497a5be
protocol v1.5 SHA256        = 0f68d6ebd5d16177e5238471639f861dfa91522130ac6a4a3ac48634b9af1e39
complete grid-result SHA256 = 908c872450e0811ec2e9cc8ca2e193925f9c302100e413e51d66f80d3419245f
```

The machine-readable freeze is `configs/final_model_v1_5.yaml`.

## Final-evaluation policy

The historical Notebook 34 filename does not authorize training. Notebook 34 is now a final-evaluation interface for this frozen checkpoint. Its default `RUN_FINAL_TEST = False` path verifies hashes, selection, checkpoint compatibility, and exact strict model-state loading without constructing a test dataset or loader.

Only an explicit later change to `RUN_FINAL_TEST = True` may consume the frozen 9-group / 90-row test subset. The access ledger is written before test rows are read so an interrupted evaluation cannot silently be repeated. Test results cannot be used for tuning, checkpoint selection, lambda modification, retraining, or comparison against alternative configurations.

```text
HYPERPARAMETER_SELECTION_COMPLETE = TRUE
FINAL_CONFIG_FROZEN = TRUE
TEST_ACCESSED = FALSE
```
