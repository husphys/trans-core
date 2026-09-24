# MEPI v1.3 xLSTM Train Readiness

The v1.3 architecture amendment freezes the validation-selected MagNet representation backbone and checkpoint. The inherited v1.2 scientific dataset and all hashes remain unchanged. The transfer dry run instantiated the downstream model without training and used strict module loads; no test loader, test metric, or test prediction was created.

## Validation

Complete regression suite: `122 passed`. The v1.2 scientific-artifact checksum manifest and v1.3 architecture-metadata checksum manifest both pass.

PROTOCOL_VERSION = MEPI-FROZEN-PROTOCOL v1.3
PROTOCOL_SHA256 = 99b31c82a3a92b6d5446477cbb9a6ae5fe40afae0476a6533324b2b14898ccfa

BACKBONE_FAMILY = xLSTM
BACKBONE_DEPTH = 8

PRETRAIN_CHECKPOINT = experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt
PRETRAIN_CHECKPOINT_SHA256 = fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c

CHECKPOINT_PROVENANCE_VERIFIED = TRUE
CHECKPOINT_ARCHITECTURE_MATCH = TRUE
PRETRAINED_BACKBONE_LOAD_READY = TRUE

DATASET_PROTOCOL = MEPI-FROZEN-PROTOCOL v1.2
PRIMARY_MEASURED_ROWS = 900
QC_VALID_ROWS = 862

TRAIN_GROUPS = 72
VALIDATION_GROUPS = 9
TEST_GROUPS = 9

TRAIN_ROWS = 687
VALIDATION_ROWS = 85
TEST_ROWS = 90

TEST_ACCESSED = FALSE

LSP_DEFINITION_READY = TRUE
NORMALIZATION_READY = TRUE
LEAKAGE_GUARD_READY = TRUE

MANUSCRIPT_BIGRU_RESULTS_LEGACY = TRUE

TRAIN_READY = TRUE
allow_training = false

Exact later command, not executed:

```bash
python -m src.mepi_v1.finetune --config configs/finetune_v1_3.yaml
```
