# MEPI v1.4 downstream training-contract provenance audit

**Status:** AUDIT COMPLETE; missing choices subsequently resolved by the experimenter; v1.4 FROZEN  
**Date:** 2026-09-24  
**Execution boundary:** no neural-network training; no test dataset/loader; no test metric  
**Authorized scope:** narrow proposed amendment to downstream optimization, losses,
waveform preprocessing, and leakage-safe PIRL training only.

## Post-audit experimenter resolution

The original audit below preserves the historical-evidence classification.
On 2026-09-24, the experimenter supplied the missing decisions. They are
`NEW_V1_4_EXPERIMENTER_AUTHORIZED`, not recovered historical settings:

- fixed learning rate with no scheduler or warmup;
- early stopping on strict validation-total-loss improvement with patience 10
  and minimum delta 0.0;
- Arrhenius target physics active but Arrhenius latent residual disabled;
- a new `Linear(256,1)` pre-PIRL P-loss auxiliary head;
- one inference-safe Steinmetz residual used identically by PIRL and
  `mean(r_St**2)` physics regularization;
- unconstrained normalized LSP mean, Softplus variance plus `1e-6`, and the
  exact full Gaussian NLL.

These decisions are frozen in MEPI-FROZEN-PROTOCOL v1.4. The fact that a
historical setting was `NOT_RECOVERABLE` remains true and is not relabeled.

## Evidence searched

- Current source, configs, notebooks, reports, artifacts, and experiment metadata.
- The complete working tree under `legacy_artifacts/` and
  `references/transformer-core/`.
- Historical downstream logs/metrics for five BiGRU depths and twelve lambda
  configurations.
- The available history of the frozen reference checkout at commit
  `8a49671de95f7849d680f707e3c7b53422487186`.
- `Manuscript_clean.docx`, including methods, Algorithm 2, and the runtime table.
- The selected xLSTM depth-8 checkpoint and its embedded preprocessing metadata.

The top-level `.git` metadata is unavailable in this workspace. The reference
checkout history contains downstream outputs and `lambda_optimize.py`, but no
committed source for the historical depth-fine-tuning scheduler or early-stop
implementation.

## Classification

Only these requested labels are used:

- `RECOVERED_FROM_EXECUTABLE_EVIDENCE`
- `RECOVERED_FROM_MANUSCRIPT_ONLY`
- `NOT_RECOVERABLE`

| Setting | Recovered value/evidence | Classification | Proposed-v1.4 disposition |
|---|---|---|---|
| Optimizer | `torch.optim.AdamW` in `references/transformer-core/lambda_optimize.py:258` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: AdamW |
| Learning rate | `5e-6` in executable code; historical depth logs begin at `5.00e-06` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: `5e-6` |
| Weight decay | `1e-4` in `lambda_optimize.py:258` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: `1e-4` |
| Scheduler policy | Lambda script has no scheduler; depth logs show run-dependent LR halvings, but scheduler source is absent | `NOT_RECOVERABLE` | Resolved by new v1.4 authorization: no scheduler, fixed `5e-6` |
| Scheduler parameters | Factor/metric/patience/threshold/cooldown/min-LR source absent | `NOT_RECOVERABLE` | Not applicable under the new no-scheduler choice |
| Batch size | Train/validation loaders use 64 in `lambda_optimize.py:245-246`; manuscript runtime table also says 64 | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: 64 |
| Maximum epochs | `range(50)` in `lambda_optimize.py:264`; every retained depth log records 50 epochs | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: 50 |
| Early-stopping patience | Manuscript says a predefined patience interval but supplies no number; retained runs complete all 50 epochs | `NOT_RECOVERABLE` | Resolved by new v1.4 authorization: 10 |
| Minimum improvement delta | No executable or manuscript value | `NOT_RECOVERABLE` | Resolved by new v1.4 authorization: 0.0, strict less-than; tie is not improvement |
| Gradient clipping | `clip_grad_norm_(..., max_norm=1.0)` in `lambda_optimize.py:284` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: 1.0 |
| Mixed precision | CUDA-conditional `autocast` and `GradScaler` in `lambda_optimize.py:259,270` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: enabled on CUDA, disabled otherwise |
| Random seed | Historical split uses `random_state=42`; current frozen configuration also specifies seed 42 | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: 42; legacy code did not fully seed every RNG |
| Historical executable checkpoint criterion | Minimum validation `MAE_norm` in `lambda_optimize.py:305-308` and retained logs | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Superseded for v1.4 by the explicitly authorized validation-total-loss rule |
| Validation-total-loss checkpoint criterion | Manuscript Algorithm 2; no matching retained executable implementation | `RECOVERED_FROM_MANUSCRIPT_ONLY` | Ready by explicit experimenter authorization: strictly lower validation total loss |
| Restore best after training | Manuscript describes final selected weights; historical scripts save best but do not demonstrate final in-memory restore | `RECOVERED_FROM_MANUSCRIPT_ONLY` | Intended, but complete early-stop contract remains blocked by missing patience/min-delta |
| Current targets | `efficiency_percent`, `P_loss`, `LSP` in the frozen current source/schema | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready; obsolete Pin/Pout outputs remain excluded |
| Target normalization | Train-only population mean/std for all three current targets in `train_normalization_v1_2.json` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready; electrical vector uses normalized efficiency and P-loss; LSP uses `LSP_z` |
| Historical executable regression loss | Legacy lambda code uses Huber + `0.1*MAE`, not the requested current three-target loss | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Historical only; not carried forward |
| `0.7*MSE + 0.3*MAE` task loss | Manuscript equations 24-25 | `RECOVERED_FROM_MANUSCRIPT_ONLY` | Ready by explicit experimenter authorization for normalized electrical vector and normalized LSP |
| Lambda 1/2 | Manuscript/config use 1.0/1.0 | `RECOVERED_FROM_MANUSCRIPT_ONLY` | Ready: 1.0/1.0 |
| Lambda 3/4 grid | Twelve combinations are executable in `lambda_optimize.py:322-326` and retained as twelve result directories | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready: `{0.1,0.3,0.5,1.0} × {0.05,0.10,0.20}` |
| Baseline lambda 3/4 | Historical BiGRU result includes 0.3/0.20 | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready only as predefined xLSTM baseline/grid member, not selected optimum |
| LSP variance floor | Current executable head uses `softplus(raw) + 1e-6` in `src/mepi_v1/models.py:170` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Numerical floor ready: `1e-6` |
| Historical Gaussian NLL | Legacy executable uses mean reduction of `0.5*exp(-logvar)*error² + 0.5*logvar` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Parameterization conflicts with the current positive-variance output; full v1.4 UQ contract not ready |
| Arrhenius normalization/residual | v1.2 protocol and `src/mepi_v1/targets.py` define shared train-only LSP scaler and residual | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Target construction/normalization retained; latent Arrhenius residual disabled by the v1.4 inference-safety correction |
| Historical Steinmetz residual | Manuscript requires core-conditioned coefficients; legacy executable instead learns/clamps per-sample coefficients | `NOT_RECOVERABLE` | Replaced by explicitly authorized core-agnostic train-only common prior |
| Common Steinmetz prior | Newly fitted exactly as authorized with `numpy.linalg.lstsq`, intercept, float64, and 687 positive train rows | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Draft prior ready; see artifact below |
| Physics loss | Legacy executable uses mean absolute learned residual | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Superseded by new v1.4 authorization: `mean(r_St²)` |
| PIRL residual projection | Current v1.3 PIRL accepts `[batch,2]` external residuals and applies `Linear(2,256) -> Tanh` | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Replaced in isolated v1.4 code by `Linear(1,256) -> Tanh` and the new pre-PIRL auxiliary head |
| Waveform preprocessing | Selected checkpoint stores MagNet-train global mean `-3.1640557780983195e-13` and scale `0.06941968146373301`; current pretraining source applies global standardization | `RECOVERED_FROM_EXECUTABLE_EVIDENCE` | Ready and representation-compatible |

## Leakage-safe common Steinmetz prior

The legacy manuscript's core-conditioned coefficient path would introduce
core identity into the physics pathway while the frozen feature contract denies
that input:

`CORE_ID_SIDE_CHANNEL_CONFLICT = TRUE`

The authorized replacement was fit using all 687 QC-valid training rows
together. No core identity was included. The exact fit is:

```text
log(P_loss_W) = log(k) + alpha*log(frequency_Hz) + beta*log(B_peak_T)
library/function = numpy.linalg.lstsq
dtype = float64
intercept = included
rcond = None
epsilon = none (non-positive/non-finite rows are rejected)
k = 1.638586240032685e-05
alpha = 1.4329982981833669
beta = 1.8648536845101025
fit rows = 687
```

One of the 72 frozen train groups has no QC-valid candidate row, so the fit
observes 71 group IDs while preserving all 72 frozen train assignments in the
serialized provenance. The artifact is
`artifacts/finetune_v1_4/steinmetz_prior_train_v1_4.json`.

Apply it without refitting:

```text
P_St_raw = k * frequency_hz**alpha * B_peak_t**beta
P_St_z   = (P_St_raw - 0.15536298551332164) / 0.09546595354711451
r_St     = P_loss_aux_z - P_St_z
```

## Waveform preprocessing recovered from the selected checkpoint

The checkpoint preprocessing is reconstructable exactly:

1. Validate a 1024-point B waveform.
2. Apply the global MagNet-training-element transform
   `(B_raw - mean) / scale`.
3. Cast to float32 and add the singleton channel dimension.

There is no clipping and no per-waveform centering/scaling. The statistic floor
used only when fitting the checkpoint's scale was `1e-12`; the stored positive
scale is applied directly downstream. This retains physical amplitude relative
to the representation-pretraining distribution.

## Resolved source-level inconsistencies

The old v1.3 classes remain unchanged. Isolated v1.4 classes now expose the
unconstrained normalized LSP mean and raw/positive variance, create
`P_loss_aux_z` directly from the pre-PIRL representation, and accept only
the inference-safe Steinmetz reference. The scheduler and stopping gaps were
closed by explicit new experimenter choices rather than historical inference.

## Audit conclusion

```text
HISTORICAL_SETTINGS_AUDITED = TRUE
CORE_ID_SIDE_CHANNEL_CONFLICT = TRUE
WAVEFORM_PREPROCESSING_READY = TRUE
STEINMETZ_PRIOR_READY = TRUE
PROTOCOL_V1_4_FROZEN = TRUE
TRAIN_READY = TRUE
TEST_ACCESSED = FALSE
SCIENTIFIC_TRAINING_RUN = FALSE
```
