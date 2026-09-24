# MEPI-FROZEN-PROTOCOL v1.4

**Status:** FROZEN  
**Date:** 2026-09-24  
**Supersedes:** MEPI-FROZEN-PROTOCOL v1.3 only for the downstream training contract and the target-derived Arrhenius latent-residual requirement  
**Purpose:** Freeze an executable, inference-safe xLSTM downstream fine-tuning contract without changing the frozen dataset, split, target construction, selected backbone, or representation checkpoint.

MEPI-FROZEN-PROTOCOL v1.3 was explicitly unfrozen only for this narrow v1.4
amendment. All unrelated v1.3 and v1.2 requirements remain unchanged,
including the dataset and QC artifacts, group-wise split, nine predictive
features, current three targets, train-only normalization, validation-only
model development, test isolation, xLSTM depth 8 architecture, latent width
256, and selected representation checkpoint.

## 1. Provenance boundary

The following settings are recovered from historical executable evidence and
remain frozen:

- optimizer: `torch.optim.AdamW`;
- learning rate: `5e-6`;
- weight decay: `1e-4`;
- batch size: 64;
- maximum epochs: 50;
- global gradient-norm clipping: 1.0;
- seed: 42;
- conditional AMP: enabled on CUDA and disabled otherwise;
- exact selected-checkpoint waveform transform;
- normalized LSP variance floor: `1e-6`.

The following are new experimenter-authorized v1.4 decisions. They are not
historically recovered settings:

- no learning-rate scheduler and no warmup;
- validation-total-loss checkpoint selection and early stopping with patience
  10, minimum delta 0.0, and strict improvement;
- disabling target-derived Arrhenius information in the latent forward path;
- a pre-PIRL auxiliary normalized P-loss head;
- Steinmetz-only latent PIRL and squared residual physics loss;
- normalized-space LSP mean/variance heads and full Gaussian NLL;
- the exact task-loss formulas, total-loss formula, baseline weights, and
  validation-only loss-weight grid.

The detailed evidence and classification record is
`reports/MEPI_V1_4_TRAINING_CONTRACT_PROVENANCE.md`.

## 2. Frozen inherited data and transfer contract

The model-development data remain the frozen MEPI v1.2 train and validation
subsets. The test subset must not be instantiated, loaded, inspected, used for
stopping, or used for checkpoint selection.

Predictive inputs remain exactly the nine frozen tabular features plus
`B(t)_1024`. The model must not receive `core_id`, measured `T_core`,
`LSP` targets, `P_loss` targets, or any target-derived quantity as a
forward input. Targets affect weights only through losses.

The downstream backbone remains xLSTM depth 8 with latent width 256. Transfer
exactly and strictly the selected checkpoint's `waveform_encoder` and
`backbone`:

```text
checkpoint = experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt
SHA256     = fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c
```

Initialize the downstream tabular encoder, fusion, P-loss auxiliary head,
Steinmetz residual projection, and multi-task heads from scratch. Obsolete
Pin/Pout prediction heads are prohibited.

## 3. Frozen waveform preprocessing

Apply the exact preprocessing embedded in the selected checkpoint:

```text
B_model = (B_raw - mean) / scale
mean    = -3.1640557780983195e-13
scale   = 0.06941968146373301
```

The statistics cover all waveform elements of the MagNet training subset.
They are global, not per-waveform. There is no clipping or per-waveform
centering. Compute the transform using the stored constants, cast the result
to float32, and add the singleton channel dimension. Do not refit these
statistics downstream.

## 4. Inference-safe PIRL correction

Arrhenius physics remains authoritative for LSP target construction,
train-only LSP normalization, LSP prediction loss, and LSP uncertainty
modeling. Measured `T_core` is unavailable during normal inference and is
therefore prohibited from the latent forward path.

```text
ARRHENIUS_TARGET_PHYSICS = ACTIVE
ARRHENIUS_LATENT_RESIDUAL = DISABLED
STEINMETZ_LATENT_RESIDUAL = ACTIVE
```

This explicitly supersedes the v1.2/v1.3 requirement to inject a
target-derived Arrhenius residual into the PIRL latent path.

Let `s` be the pooled pre-PIRL representation with width 256. Define a newly
initialized internal auxiliary head:

```text
P_loss_aux_z = Linear(256, 1)(s)
```

It has no activation. It receives only `s`; it must not receive core
identity, measured core temperature, targets, or target-derived fields.

Use the already fitted common train-only Steinmetz prior without refitting:

```text
artifact = artifacts/finetune_v1_4/steinmetz_prior_train_v1_4.json
SHA256   = a5e110924093455cf6e0912ed4fcf82c046fec82326f0480d56358532662d5f8

k     = 1.638586240032685e-05
alpha = 1.4329982981833669
beta  = 1.8648536845101025

P_St_raw = k * frequency_hz**alpha * B_peak_t**beta
P_St_z   = (P_St_raw - 0.15536298551332164) / 0.09546595354711451
r_St     = P_loss_aux_z - P_St_z
```

The prior was fit on 687 QC-valid training rows without `core_id`. Validation
must not refit it. The PIRL is:

```text
s_phys = s + Tanh(Linear(1, 256)(r_St))
L_physics = mean(r_St ** 2)
```

The same `r_St` tensor must be used for latent projection and physics
regularization. Differently normalized copies are prohibited.

## 5. Frozen normalized-space prediction heads

The electrical prediction vector contains only normalized efficiency and
normalized P-loss:

```text
yhat_electrical_z = [efficiency_percent_z_hat, P_loss_z_hat]
```

The two electrical heads retain the existing downstream hidden-head form,
`Linear(256,64) -> GELU -> Linear(64,1)`, but their normalized outputs are
unconstrained; no Softplus is applied.

From `s_phys`, define two independent LSP outputs:

```text
mu_LSP_z      = Linear(256, 1)(s_phys)
raw_var_LSP_z = Linear(256, 1)(s_phys)
var_LSP_z     = softplus(raw_var_LSP_z) + 1e-6
```

No positivity transform or other activation may be applied to `mu_LSP_z`.
The variance floor is exactly `1e-6`.

At inference, invert the train-only LSP normalization as:

```text
LSP_raw_pred = mu_LSP_z * sigma_LSP_train + mu_LSP_train
var_LSP_raw  = var_LSP_z * sigma_LSP_train**2
std_LSP_raw  = sqrt(var_LSP_raw)
```

Normalized variance must not be reported as raw-space variance.

## 6. Frozen losses

Targets are exactly `efficiency_percent`, `P_loss`, and `LSP`. Use the
frozen train-only population mean and standard deviation for every target;
validation uses those unchanged training statistics.

```text
L_electrical =
    0.7 * MSE(yhat_electrical_z, y_electrical_z)
  + 0.3 * MAE(yhat_electrical_z, y_electrical_z)

L_LSP =
    0.7 * MSE(mu_LSP_z, LSP_z)
  + 0.3 * MAE(mu_LSP_z, LSP_z)

L_UQ =
    mean(
        0.5 * (
            log(2*pi*var_LSP_z)
            + (LSP_z - mu_LSP_z)**2 / var_LSP_z
        )
    )
```

`L_UQ` is full Gaussian NLL with mean reduction, entirely in normalized LSP
space. No physical-space value enters this NLL.

For the predefined baseline only:

```text
lambda1 = 1.0
lambda2 = 1.0
lambda3 = 0.3
lambda4 = 0.20

L_total =
    L_electrical
  + L_LSP
  + lambda3 * L_physics
  + lambda4 * L_UQ
```

The baseline pair `(0.3, 0.20)` is not claimed to be the optimal xLSTM
setting. A later validation-only experiment may select from:

```text
lambda3 = [0.1, 0.3, 0.5, 1.0]
lambda4 = [0.05, 0.10, 0.20]
lambda1 = lambda2 = 1.0
```

No loss-weight experiment is authorized by this freeze itself, and the test
subset remains inaccessible throughout model development.

## 7. Frozen optimizer and stopping contract

```text
optimizer          = AdamW
learning_rate      = 5e-6, fixed for the complete run
weight_decay       = 1e-4
scheduler          = NONE
warmup             = NONE
batch_size         = 64
max_epochs         = 50
gradient_clip_norm = 1.0
seed               = 42
AMP                = enabled only when CUDA is active
```

There is no scheduler object, `scheduler.step()`, scheduler monitor,
ReduceLROnPlateau, cosine decay, OneCycle, or inherited MagNet scheduler.

After every epoch, evaluate `L_total` on validation only. Improvement is
strictly:

```text
validation_total_loss < best_validation_total_loss
```

An exact tie is not an improvement. Set `patience=10` and `min_delta=0.0`.
After ten consecutive epochs without strict improvement, stop and restore the
checkpoint with the lowest validation total loss. Never use test data for
stopping or checkpoint selection.

## 8. Crash-safe resume contract

The rolling checkpoint must be replaced atomically and contain at least:

- next epoch;
- complete model state;
- AdamW optimizer state;
- best validation total loss;
- best epoch;
- patience counter;
- AMP scaler state when active;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- PyTorch CUDA RNG state when available;
- compatibility metadata and training history.

No scheduler state exists or is required. A compatibility mismatch must stop
resume rather than reconstruct state or silently start from weights only.
A completed run must not be silently retrained; explicit `FORCE_RETRAIN=True`
is required to discard its local rolling/best outputs.

## 9. Run-All notebook contract

The authoritative interface is
`notebooks/32_xlstm_v1_4_finetune.ipynb`, intended for:

```text
Restart Kernel -> Run All
```

Its default is `RUN_TRAINING = False`. Under that default, all protocol,
artifact, architecture, leakage, loss, and resume software checks pass and the
training call skips cleanly. Setting `RUN_TRAINING = True` is the deliberate
manual authorization to run the frozen baseline on train plus validation only.
The notebook must never instantiate a test dataset or test loader.

## 10. Frozen status and change control

```text
SCHEDULER_READY = TRUE
PATIENCE_READY = TRUE
PIRL_INFERENCE_SAFE = TRUE
STEINMETZ_PRIOR_READY = TRUE
ARRHENIUS_TARGET_PHYSICS_READY = TRUE
ARRHENIUS_LATENT_RESIDUAL_DISABLED = TRUE
UQ_LOSS_READY = TRUE
RESUME_READY = TRUE
NOTEBOOK_V1_4_READY = TRUE
TRAIN_READY = TRUE
SCIENTIFIC_TRAINING_RUN = FALSE
TEST_ACCESSED = FALSE
```

MEPI-FROZEN-PROTOCOL v1.4 is frozen. Any later scientific change requires an
explicit instruction containing:

`UNFREEZE MEPI-FROZEN-PROTOCOL v1.4`

