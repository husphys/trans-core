# MEPI-FROZEN-PROTOCOL v1.3

**Status:** FROZEN  
**Date:** 2026-09-24  
**Supersedes:** MEPI-FROZEN-PROTOCOL v1.2 (2026-09-24)  
**Purpose:** Narrow amendment correcting the frozen MEPI backbone and transfer-checkpoint provenance to match the actual selected xLSTM representation experiment.

MEPI-FROZEN-PROTOCOL v1.2 was explicitly unfrozen only for this architecture
and checkpoint-provenance amendment. All v1.2 requirements not explicitly
named below remain unchanged, including the frozen dataset, B processing,
temperature mapping, LSP definition, train-only normalization, split, QC,
leakage controls, validation-only model development, and final-test isolation.

## v1.3 Amendment from v1.2

### 1. Frozen downstream sequence backbone

```text
BACKBONE_FAMILY = xLSTM
BACKBONE_DEPTH  = 8
LATENT_WIDTH    = 256
```

Use the exact `XLSTMBackbone` and `XLSTMBlock` implementation in
`src/mepi_v1/backbones.py`. Each of the eight blocks contains:

- `LayerNorm(256)` before the memory path;
- one unidirectional, single-layer `torch.nn.LSTM` with input size 256,
  hidden size 256, `batch_first=True`, and dropout 0.0;
- a gate `Linear(256,256)` followed by `Sigmoid`;
- a projection `Linear(256,256)`;
- the residual update
  `x + gate(LayerNorm(x)) * projection(LSTM(LayerNorm(x)))`.

No hidden size, block type, normalization, dropout, projection, or residual
form may be silently changed. After the backbone, downstream MEPI uses mean
pooling over the sequence dimension, exactly as implemented in
`DownstreamMEPI.forward`.

The unchanged waveform branch uses Conv1d channels 1→64→128→128 with kernel
size 7 and padding 3, GELU activations, max-pooling after the first two
convolutions, an FFT-magnitude branch adaptively pooled to the temporal token
length, concatenated token width 129, and `Linear(129,256)` projection. The
downstream nine-feature tabular encoder, fusion, PIRL, and multi-task heads
remain the existing implementations; this amendment does not redesign them.

### 2. Frozen representation-pretraining checkpoint

```text
path   = experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt
SHA256 = fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c
origin = completed MEPI xLSTM depth sensitivity on MagNet
```

The checkpoint was selected at epoch 10 by minimum normalized MAE on the
MagNet validation subset (`validation MAE_norm = 0.020495040982892985`). The
selection evidence records `test_split_accessed = false` and
`test_evaluations = 0`. Its pretraining model has 5,969,930 trainable
parameters.

MagNet representation pretraining contained:

- waveform encoder;
- two-feature MagNet operating-condition encoder;
- multimodal fusion;
- eight-block xLSTM backbone;
- post-backbone `LayerNorm(256)` and temporal mean pooling;
- ten material-routed temporary core-loss heads, each
  `Linear(256,64) → GELU → Linear(64,1)`.

PIRL and downstream MTPH were not present during MagNet pretraining.

### 3. Frozen transfer boundary

Transfer exactly and strictly:

- the complete pretrained `waveform_encoder` state;
- the complete pretrained eight-block xLSTM `backbone` state.

The checkpoint supplies 8 waveform-encoder state keys and 80 backbone state
keys. All 88 keys must match the downstream modules exactly, including tensor
shapes. A pretrained waveform/backbone missing key, unexpected key, or shape
mismatch fails readiness; arbitrary `strict=False` loading is prohibited.

Do not transfer the MagNet operating encoder, pretraining fusion,
post-backbone pretraining norm, or temporary material heads. Initialize the
downstream nine-feature tabular encoder, downstream fusion, PIRL, and MTPH as
downstream-only modules. Downstream MEPI therefore consists of the pretrained
waveform encoder plus pretrained xLSTM backbone, followed by the existing
downstream PIRL and multi-task heads.

### 4. Superseded BiGRU authority and result boundary

Any inherited v1.1/v1.2 or manuscript text identifying BiGRU as the final
frozen MEPI backbone is superseded. BiGRU remains a historical candidate and
historical experiment only.

Old BiGRU depth metrics, downstream prediction metrics, parameter counts,
checkpoint sizes, training times, architecture figures, and BiGRU-attributed
conclusions are marked `LEGACY_RESULT_REQUIRES_RETRAINING`. They are not xLSTM
results and must not be carried forward. New downstream xLSTM training and
validation must generate replacement evidence before manuscript numerical
claims are revised. The frozen test subset remains inaccessible until model
development and checkpoint selection are complete.

## Frozen change control

MEPI-FROZEN-PROTOCOL v1.3 is now frozen. Any later scientific change requires
an explicit instruction containing:

`UNFREEZE MEPI-FROZEN-PROTOCOL v1.3`

Consistency fixes required to implement this text do not change the protocol.
