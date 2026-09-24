# MEPI Backbone Provenance Audit

## Evidence examined

The audit compared frozen protocols v1.1/v1.2, `Manuscript_clean.docx`, MagNet/xLSTM configs, all five depth summaries, depth comparison and selection files, source implementations, checkpoint metadata/state dictionaries, training metrics, transfer tests, and current fine-tuning configuration. Git history is unavailable because this workspace is not a Git working tree.

## Selection conclusion

The clean measured backbone comparison selected xLSTM on validation MAE_norm. The subsequent completed MagNet representation depth study evaluated depths 2/4/6/8/10 and explicitly selected depth 8 by minimum validation MAE_norm `0.020495040982892985`. Its checkpoint metadata independently records `backbone=xLSTM`, `latent_dim=256`, and `backbone_layers=8`. The bound checkpoint hash is `fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c`. The study constructed only train and validation datasets; summary evidence reports `test_split_accessed=false` and `test_evaluations=0`.

This proves xLSTM depth 8 as the selected **representation-pretraining** backbone/checkpoint. It does not validate old downstream BiGRU metrics as xLSTM metrics; downstream xLSTM fine-tuning must produce new validation results.

The depth-study summary retains its historical `MEPI-FROZEN-PROTOCOL v1.0` provenance label. It is not silently relabeled. The later v1.1/v1.2 amendments changed acquisition/data and LSP definitions rather than the MagNet representation architecture, and v1.3 explicitly adopts this exact hash-fixed checkpoint.

## Exact architecture recovered

- Family/depth/width: xLSTM / 8 blocks / latent width 256.
- Pretraining trainable parameters: 5,969,930.
- Waveform encoder: Conv1d 1→64, kernel 7/padding 3, GELU, max-pool 2; Conv1d 64→128, kernel 7/padding 3, GELU, max-pool 2; Conv1d 128→128, kernel 7/padding 3, GELU; FFT magnitude pooled to the temporal length; concatenated width 129 projected by Linear(129,256).
- MagNet operating encoder: Linear(2,256), GELU, LayerNorm(256), Linear(256,256).
- Fusion: MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.0, batch_first=True) plus residual LayerNorm(256).
- Each xLSTM block: LayerNorm(256); single-layer unidirectional torch.nn.LSTM(256, 256, batch_first=True); Linear(256,256) then Sigmoid; Linear(256,256); residual `x + gate(LayerNorm(x)) * projection(LSTM(LayerNorm(x)))`; dropout 0.0.
- Post-backbone normalization/pooling: LayerNorm(256), then mean over the sequence dimension.
- Temporary head: 10 x [Linear(256,64), GELU, Linear(64,1)] routed by material metadata.
- PIRL during MagNet pretraining: NO.
- Downstream MTPH during MagNet pretraining: NO.

## Checkpoint and dry-run transfer

- Checkpoint full `model_state` keys: 142.
- Intended strict matches: 88 (`waveform_encoder` 8 + `backbone` 80).
- Expected newly initialized downstream keys: 26.
- Pretraining-only full-model keys not transferred: 54.
- Unexpected keys in intended strict module loads: 0.
- Shape mismatches: 0.
- Arbitrary `strict=False`: NOT USED.

### Exact matched keys

- `waveform_encoder.temporal.0.weight`
- `waveform_encoder.temporal.0.bias`
- `waveform_encoder.temporal.3.weight`
- `waveform_encoder.temporal.3.bias`
- `waveform_encoder.temporal.6.weight`
- `waveform_encoder.temporal.6.bias`
- `waveform_encoder.projection.weight`
- `waveform_encoder.projection.bias`
- `backbone.blocks.0.norm.weight`
- `backbone.blocks.0.norm.bias`
- `backbone.blocks.0.memory.weight_ih_l0`
- `backbone.blocks.0.memory.weight_hh_l0`
- `backbone.blocks.0.memory.bias_ih_l0`
- `backbone.blocks.0.memory.bias_hh_l0`
- `backbone.blocks.0.gate.0.weight`
- `backbone.blocks.0.gate.0.bias`
- `backbone.blocks.0.projection.weight`
- `backbone.blocks.0.projection.bias`
- `backbone.blocks.1.norm.weight`
- `backbone.blocks.1.norm.bias`
- `backbone.blocks.1.memory.weight_ih_l0`
- `backbone.blocks.1.memory.weight_hh_l0`
- `backbone.blocks.1.memory.bias_ih_l0`
- `backbone.blocks.1.memory.bias_hh_l0`
- `backbone.blocks.1.gate.0.weight`
- `backbone.blocks.1.gate.0.bias`
- `backbone.blocks.1.projection.weight`
- `backbone.blocks.1.projection.bias`
- `backbone.blocks.2.norm.weight`
- `backbone.blocks.2.norm.bias`
- `backbone.blocks.2.memory.weight_ih_l0`
- `backbone.blocks.2.memory.weight_hh_l0`
- `backbone.blocks.2.memory.bias_ih_l0`
- `backbone.blocks.2.memory.bias_hh_l0`
- `backbone.blocks.2.gate.0.weight`
- `backbone.blocks.2.gate.0.bias`
- `backbone.blocks.2.projection.weight`
- `backbone.blocks.2.projection.bias`
- `backbone.blocks.3.norm.weight`
- `backbone.blocks.3.norm.bias`
- `backbone.blocks.3.memory.weight_ih_l0`
- `backbone.blocks.3.memory.weight_hh_l0`
- `backbone.blocks.3.memory.bias_ih_l0`
- `backbone.blocks.3.memory.bias_hh_l0`
- `backbone.blocks.3.gate.0.weight`
- `backbone.blocks.3.gate.0.bias`
- `backbone.blocks.3.projection.weight`
- `backbone.blocks.3.projection.bias`
- `backbone.blocks.4.norm.weight`
- `backbone.blocks.4.norm.bias`
- `backbone.blocks.4.memory.weight_ih_l0`
- `backbone.blocks.4.memory.weight_hh_l0`
- `backbone.blocks.4.memory.bias_ih_l0`
- `backbone.blocks.4.memory.bias_hh_l0`
- `backbone.blocks.4.gate.0.weight`
- `backbone.blocks.4.gate.0.bias`
- `backbone.blocks.4.projection.weight`
- `backbone.blocks.4.projection.bias`
- `backbone.blocks.5.norm.weight`
- `backbone.blocks.5.norm.bias`
- `backbone.blocks.5.memory.weight_ih_l0`
- `backbone.blocks.5.memory.weight_hh_l0`
- `backbone.blocks.5.memory.bias_ih_l0`
- `backbone.blocks.5.memory.bias_hh_l0`
- `backbone.blocks.5.gate.0.weight`
- `backbone.blocks.5.gate.0.bias`
- `backbone.blocks.5.projection.weight`
- `backbone.blocks.5.projection.bias`
- `backbone.blocks.6.norm.weight`
- `backbone.blocks.6.norm.bias`
- `backbone.blocks.6.memory.weight_ih_l0`
- `backbone.blocks.6.memory.weight_hh_l0`
- `backbone.blocks.6.memory.bias_ih_l0`
- `backbone.blocks.6.memory.bias_hh_l0`
- `backbone.blocks.6.gate.0.weight`
- `backbone.blocks.6.gate.0.bias`
- `backbone.blocks.6.projection.weight`
- `backbone.blocks.6.projection.bias`
- `backbone.blocks.7.norm.weight`
- `backbone.blocks.7.norm.bias`
- `backbone.blocks.7.memory.weight_ih_l0`
- `backbone.blocks.7.memory.weight_hh_l0`
- `backbone.blocks.7.memory.bias_ih_l0`
- `backbone.blocks.7.memory.bias_hh_l0`
- `backbone.blocks.7.gate.0.weight`
- `backbone.blocks.7.gate.0.bias`
- `backbone.blocks.7.projection.weight`
- `backbone.blocks.7.projection.bias`

### Expected downstream-only keys

- `fusion.cross_attention.in_proj_bias` — newly initialized downstream module
- `fusion.cross_attention.in_proj_weight` — newly initialized downstream module
- `fusion.cross_attention.out_proj.bias` — newly initialized downstream module
- `fusion.cross_attention.out_proj.weight` — newly initialized downstream module
- `fusion.norm.bias` — newly initialized downstream module
- `fusion.norm.weight` — newly initialized downstream module
- `mtph.efficiency.0.bias` — newly initialized downstream module
- `mtph.efficiency.0.weight` — newly initialized downstream module
- `mtph.efficiency.2.bias` — newly initialized downstream module
- `mtph.efficiency.2.weight` — newly initialized downstream module
- `mtph.loss_related.0.bias` — newly initialized downstream module
- `mtph.loss_related.0.weight` — newly initialized downstream module
- `mtph.loss_related.2.bias` — newly initialized downstream module
- `mtph.loss_related.2.weight` — newly initialized downstream module
- `mtph.lsp.0.bias` — newly initialized downstream module
- `mtph.lsp.0.weight` — newly initialized downstream module
- `mtph.lsp.2.bias` — newly initialized downstream module
- `mtph.lsp.2.weight` — newly initialized downstream module
- `pirl.projection.0.bias` — newly initialized downstream module
- `pirl.projection.0.weight` — newly initialized downstream module
- `tabular_encoder_finetune.network.0.bias` — newly initialized downstream module
- `tabular_encoder_finetune.network.0.weight` — newly initialized downstream module
- `tabular_encoder_finetune.network.2.bias` — newly initialized downstream module
- `tabular_encoder_finetune.network.2.weight` — newly initialized downstream module
- `tabular_encoder_finetune.network.3.bias` — newly initialized downstream module
- `tabular_encoder_finetune.network.3.weight` — newly initialized downstream module

### Pretraining-only keys intentionally not transferred

- `fusion.cross_attention.in_proj_bias` — pretraining-only, intentionally not transferred
- `fusion.cross_attention.in_proj_weight` — pretraining-only, intentionally not transferred
- `fusion.cross_attention.out_proj.bias` — pretraining-only, intentionally not transferred
- `fusion.cross_attention.out_proj.weight` — pretraining-only, intentionally not transferred
- `fusion.norm.bias` — pretraining-only, intentionally not transferred
- `fusion.norm.weight` — pretraining-only, intentionally not transferred
- `norm.bias` — pretraining-only, intentionally not transferred
- `norm.weight` — pretraining-only, intentionally not transferred
- `operating_encoder_pretrain.network.0.bias` — pretraining-only, intentionally not transferred
- `operating_encoder_pretrain.network.0.weight` — pretraining-only, intentionally not transferred
- `operating_encoder_pretrain.network.2.bias` — pretraining-only, intentionally not transferred
- `operating_encoder_pretrain.network.2.weight` — pretraining-only, intentionally not transferred
- `operating_encoder_pretrain.network.3.bias` — pretraining-only, intentionally not transferred
- `operating_encoder_pretrain.network.3.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.0.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.0.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.0.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.0.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.1.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.1.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.1.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.1.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.2.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.2.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.2.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.2.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.3.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.3.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.3.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.3.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.4.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.4.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.4.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.4.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.5.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.5.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.5.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.5.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.6.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.6.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.6.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.6.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.7.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.7.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.7.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.7.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.8.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.8.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.8.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.8.2.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.9.0.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.9.0.weight` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.9.2.bias` — pretraining-only, intentionally not transferred
- `temporary_heads.heads.9.2.weight` — pretraining-only, intentionally not transferred

## BiGRU conflict classification

`Manuscript_clean.docx` and inherited v1.1 text still describe BiGRU and old BiGRU downstream results. Those statements conflict with the completed clean selection evidence above. They are historical manuscript claims, not evidence for this selected checkpoint, and must be marked `LEGACY_RESULT_REQUIRES_RETRAINING`; their metrics, parameter counts, checkpoint sizes, and training times cannot be transferred to xLSTM.

ACTUAL_SELECTED_BACKBONE = xLSTM
ACTUAL_SELECTED_DEPTH = 8
CHECKPOINT_SHA256 = fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c
CHECKPOINT_PROVENANCE_VERIFIED = TRUE
CHECKPOINT_SELECTED_ON_VALIDATION_ONLY = TRUE
LEGACY_BIGRU_TEXT_CONFLICT = TRUE
ARCHITECTURE_AMENDMENT_JUSTIFIED = TRUE
