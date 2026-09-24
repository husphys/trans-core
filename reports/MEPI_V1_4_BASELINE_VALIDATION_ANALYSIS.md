# MEPI v1.4 baseline validation analysis

## Scope and evidence protection

**OBSERVED**

- Completed run: `experiments/finetune_v1_4_xlstm_depth8/baseline_l3_0p3_l4_0p20`; no file in this directory was modified.
- Best-checkpoint SHA-256: `22aca18c37b4526c46e3044eeeda4e36a6a0364891e767cace755961eef44138` (expected hash matched).
- Root/docs protocol SHA-256: `685f79b6b700ae131ff2a19af524417ead46ee9c073f1c2ddf9d96191f59b487` (byte-identical and config-matched).
- Completion metadata: 50 epochs, best epoch 49, best validation total loss `0.4636393350713393`.
- Completion metadata and this evaluator record `test_accessed = false`, `test_dataset_created = false`, and `test_loader_created = false`.
- Validation sample count: 85; evaluation device: `cuda`; the frozen CUDA-conditional autocast path was used.

**INTERPRETATION**

The supplied checkpoint and frozen-evidence checks validate the completed baseline. This was existing-checkpoint validation only, not retraining and not test evaluation.

## Complete history dataframe

**OBSERVED**

The actual `history.json` supplied exactly one row for each epoch 0..49. The companion JSON contains all 50 flattened records under `history_dataframe.records`.

| Dataframe column | Original history field |
|---|---|
| epoch | epoch |
| learning_rate | learning_rate |
| train_L_LSP | train.LSP |
| train_L_UQ | train.uncertainty_nll |
| train_L_electrical | train.electrical |
| train_L_physics | train.physics |
| train_total_loss | train.total |
| validation_L_LSP | validation.LSP |
| validation_L_UQ | validation.uncertainty_nll |
| validation_L_electrical | validation.electrical |
| validation_L_physics | validation.physics |
| validation_total_loss | validation.total |

## Convergence

**OBSERVED**

- Minimum and final validation loss: `0.463639335071` at epoch 49.
- Epoch 44→49 absolute/relative improvement: `0.040353109556` / `8.0067%`.
- Epoch 39→49 absolute/relative improvement: `0.098406396193` / `17.5086%`.

| Epoch | Validation total loss |
|---:|---:|
| 39 | 0.562045731264 |
| 40 | 0.563148650001 |
| 41 | 0.529530504171 |
| 42 | 0.538321336578 |
| 43 | 0.533730378572 |
| 44 | 0.503992444627 |
| 45 | 0.490869196723 |
| 46 | 0.481261064726 |
| 47 | 0.502098899028 |
| 48 | 0.482730866881 |
| 49 | 0.463639335071 |

**INTERPRETATION**

`A_STILL_CLEARLY_IMPROVING_AT_EPOCH_49`. Epoch 49 is the strict minimum. Despite short-term fluctuations, both endpoint improvements are material; the history does not support declaring convergence or a plateau. This audit does not change `max_epochs`.

## Train-versus-validation gap at epoch 49

**OBSERVED**

Relative gap is `(validation - training) / abs(training)`. A material flag requires the best absolute gap to exceed 1.5 times the median absolute gap at epochs 39..48 and to exceed that median by more than 0.01.

| Component | Train | Validation | Validation - train | Relative gap | Material recent increase? |
|---|---:|---:|---:|---:|---:|
| L_LSP | 0.143670381287 | 0.151141057470 | 0.007470676183 | 5.1999% | no |
| L_UQ | 0.287968587819 | 0.383404977883 | 0.095436390064 | 33.1413% | no |
| L_electrical | 0.145126533821 | 0.206811994665 | 0.061685460844 | 42.5046% | no |
| L_physics | 0.043435973561 | 0.096684300374 | 0.053248326812 | 122.5904% | no |
| total | 0.359421418180 | 0.463639335071 | 0.104217916891 | 28.9960% | no |

**INTERPRETATION**

`NO_MATERIAL_RECENT_GAP_INCREASE`. No component met the stated material-increase rule. This is descriptive; no automatic overfitting conclusion was applied. Total validation loss is still the run minimum.

## Loss-component contributions at epoch 49

**OBSERVED**

| Component | Raw validation | Weight | Weighted contribution | Percent of total |
|---|---:|---:|---:|---:|
| L_LSP | 0.151141057470 | 1.00 | 0.151141057470 | 32.5988% |
| L_UQ | 0.383404977883 | 0.20 | 0.076680995577 | 16.5389% |
| L_electrical | 0.206811994665 | 1.00 | 0.206811994665 | 44.6062% |
| L_physics | 0.096684300374 | 0.30 | 0.029005290112 | 6.2560% |

Weighted sum `0.463639337824` and recorded total `0.463639335071` differ by only `2.75e-09` from stored batch-aggregation rounding.

**INTERPRETATION**

`L_ELECTRICAL_LARGEST_NO_SINGLE_COMPONENT_OVER_50_PERCENT`. This describes the baseline objective only and does not establish optimal loss weights.

## Best-checkpoint physical-space validation metrics

**OBSERVED**

Frozen train-only target scalers were used. LSP MAPE is physical `LSP_raw` MAPE, not normalized-space MAPE.

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Efficiency (%) | 2.24236580344 | 2.77032476912 | 0.901521637894 |
| P_loss | 0.0316138867946 | 0.0486487558417 | 0.803167998264 |
| LSP_raw | 0.0223299703177 | 0.026702036478 | 0.946246457949 |

- LSP_raw MAPE: `3.98520040512%`; zero target count: 0.
- Recomputed validation total loss: `0.463639335071`.

**INTERPRETATION**

These are frozen-validation metrics for the existing best checkpoint only. They are not test metrics and do not establish external generalization.

## Numerical and UQ audit

**OBSERVED**

- Any NaN / Inf: `false` / `false`.
- `var_LSP_z` min / median / max: `0.0667130500078` / `0.113450057805` / `0.500246286392`.
- Variance floor `1e-6` hit: `false`.
- Mean full Gaussian NLL (normalized LSP): `0.383404977883`.

**INTERPRETATION**

Numerical failure: `False`. No UQ calibration experiment was introduced.

## Final status

```text
BASELINE_RUN_VALID = True
BEST_EPOCH = 49
BEST_VALIDATION_TOTAL_LOSS = 0.4636393350713393
LAST_5_EPOCH_IMPROVEMENT = 0.0403531095560859
LAST_10_EPOCH_IMPROVEMENT = 0.09840639619266284
CONVERGENCE_STATUS = A_STILL_CLEARLY_IMPROVING_AT_EPOCH_49
TRAIN_VALIDATION_GAP_STATUS = NO_MATERIAL_RECENT_GAP_INCREASE
LOSS_COMPONENT_BALANCE_STATUS = L_ELECTRICAL_LARGEST_NO_SINGLE_COMPONENT_OVER_50_PERCENT
VAL_EFFICIENCY_MAE = 2.2423658034380742
VAL_EFFICIENCY_RMSE = 2.7703247691154087
VAL_EFFICIENCY_R2 = 0.9015216378938009
VAL_PLOSS_MAE = 0.03161388679462321
VAL_PLOSS_RMSE = 0.0486487558417053
VAL_PLOSS_R2 = 0.8031679982639643
VAL_LSP_MAE = 0.02232997031772838
VAL_LSP_RMSE = 0.0267020364780092
VAL_LSP_R2 = 0.9462464579488546
VAL_LSP_MAPE_PERCENT = 3.9852004051208496
NUMERICAL_FAILURE = False
TEST_ACCESSED = FALSE
READY_FOR_LOSS_WEIGHT_SENSITIVITY = True
```
