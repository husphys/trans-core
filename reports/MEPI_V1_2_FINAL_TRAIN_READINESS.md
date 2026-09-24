# MEPI v1.2 Final Train Readiness

## Protocol

Both frozen v1.2 copies are byte-identical at SHA-256 `c0fd2fe8aafc0c55dc0b6c525e341d0458782391f6476831da2e3cc5cf401988`. Historical v1.1 remains unchanged at `a515052cd2f8cf2970b731cfb48dee055c316a5d4acc0f0ca90313436c344c0f`.

## Dataset and QC

Primary measured rows: 900. QC-valid rows: 862 (445 FE, 417 COMMERCIAL). The old Commercial 6.x session is preserved as 75 supplementary rows and excluded from all active subsets and normalization. Every active primary row uses Scope #2 CH1 primary Vin for B; `B1024_v1_2.npy` shape is (862, 1024). No demo or duplicate sample entered the dataset.

## LSP

The exact v1.2 float64/no-epsilon definition is implemented. `LSP_raw` min/median/max = 0.302085964148/0.605125887514/0.776920969806. It equals one at 298.15 K, is finite and positive, and decreases monotonically with increasing core temperature.

## Frozen split and normalization

Group counts train/validation/test = 72/9/9; row counts = 687/85/90. No group crosses subsets. QC-valid rows occur in 89/90 identities; `F1000_V5p6_RL49p6025_SINE` is retained in the group manifest with zero eligible rows. Training-only LSP population mean/std (`ddof=0`) = 0.60305503339713129/0.083751447010489311. The target, Arrhenius reference, and PIRL Arrhenius residual use this identical scaler. Validation/test were not refitted, and test was not used for model-development decisions.

## Leakage and downstream boundary

Exactly nine tabular inputs: `frequency_hz, vin_rms_v, phase_shift_deg, temperature_ambient_c, B_peak_t, B_rms, B_thd_percent, dBdt_max, form_factor`. Core identity, corrected core temperature, absolute core temperature, LSP construction fields, targets, and power quantities are excluded from predictive inputs. Transfer audit passed for `experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt` at SHA-256 `fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c`; only `waveform_encoder` and matching depth-8 xLSTM `backbone` will transfer.

Intended command (prepared, not executed):

```bash
python -m src.mepi_v1.finetune --config configs/finetune_v1_2.yaml
```

Training remains disabled in the config pending explicit execution authorization. `test_evaluations = 0`; `training_run = false`.

B_READY = TRUE
TEMPERATURE_MAPPING_READY = TRUE
PRIMARY_DATASET_READY = TRUE
ELECTRICAL_QC_READY = TRUE
LSP_DEFINITION_READY = TRUE
LSP_TARGET_READY = TRUE
SPLIT_READY = TRUE
NORMALIZATION_READY = TRUE
LEAKAGE_GUARD_READY = TRUE
TRAIN_READY = TRUE
