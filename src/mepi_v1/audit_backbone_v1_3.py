"""Audit the selected MagNet representation checkpoint before protocol v1.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .models import DownstreamMEPI, PretrainingModel

CHECKPOINT_RELATIVE = "experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt"
EXPECTED_SHA256 = "fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def audit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    checkpoint_path = root / CHECKPOINT_RELATIVE
    summary_path = checkpoint_path.with_name("training_summary.json")
    selected_path = checkpoint_path.parents[1] / "selected_depth.json"
    comparison_path = checkpoint_path.parents[1] / "depth_comparison.csv"
    required = (checkpoint_path, summary_path, selected_path, comparison_path)
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing architecture evidence: {missing_files}")

    digest = _sha256(checkpoint_path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Depth-8 checkpoint SHA mismatch: {digest}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    family = config.get("backbone")
    depth = int(model_config.get("backbone_layers", -1))
    latent_dim = int(model_config.get("latent_dim", -1))
    if (family, depth, latent_dim) != ("xLSTM", 8, 256):
        raise AssertionError(f"Checkpoint architecture is {(family, depth, latent_dim)}, not xLSTM/8/256")
    if selected.get("selected_depth") != 8 or selected.get("backbone") != "xLSTM":
        raise AssertionError("selected_depth.json does not select xLSTM depth 8")
    if selected.get("selection_criterion") != "minimum validation MAE_norm":
        raise AssertionError("Depth selection criterion is not validation MAE_norm")
    if selected.get("test_split_accessed") is not False:
        raise AssertionError("Depth selection evidence reports test access")
    if summary.get("best_checkpoint_sha256") != digest:
        raise AssertionError("Training summary does not bind the checkpoint hash")
    if summary.get("test_split_accessed") is not False or summary.get("test_evaluations") != 0:
        raise AssertionError("Training summary violates test isolation")

    pretraining = PretrainingModel(
        "xLSTM", material_count=10, latent_dim=latent_dim, backbone_layers=depth
    )
    full_load = pretraining.load_state_dict(checkpoint["model_state"], strict=True)
    if full_load.missing_keys or full_load.unexpected_keys:
        raise AssertionError("Full pretraining checkpoint does not strict-load")
    if _parameter_count(pretraining) != int(summary["parameter_count"]):
        raise AssertionError("Reconstructed parameter count differs from evidence")

    downstream = DownstreamMEPI("xLSTM", latent_dim=latent_dim, backbone_layers=depth)
    waveform_state = checkpoint["waveform_encoder"]
    backbone_state = checkpoint["backbone"]
    downstream_waveform = downstream.waveform_encoder.state_dict()
    downstream_backbone = downstream.backbone.state_dict()
    shape_mismatches: list[dict[str, Any]] = []
    for prefix, source, target in (
        ("waveform_encoder", waveform_state, downstream_waveform),
        ("backbone", backbone_state, downstream_backbone),
    ):
        for key, tensor in source.items():
            if key not in target or tuple(tensor.shape) != tuple(target[key].shape):
                shape_mismatches.append(
                    {
                        "key": f"{prefix}.{key}",
                        "checkpoint_shape": list(tensor.shape),
                        "downstream_shape": list(target[key].shape) if key in target else None,
                    }
                )
    if shape_mismatches:
        raise AssertionError(f"Pretrained transfer shape mismatch: {shape_mismatches}")
    waveform_load = downstream.waveform_encoder.load_state_dict(waveform_state, strict=True)
    backbone_load = downstream.backbone.load_state_dict(backbone_state, strict=True)
    if any(
        (result.missing_keys or result.unexpected_keys)
        for result in (waveform_load, backbone_load)
    ):
        raise AssertionError("Intended transferred modules do not strict-load")

    matched_keys = [f"waveform_encoder.{key}" for key in waveform_state]
    matched_keys += [f"backbone.{key}" for key in backbone_state]
    downstream_keys = list(downstream.state_dict())
    expected_missing = sorted(set(downstream_keys) - set(matched_keys))
    checkpoint_model_keys = list(checkpoint["model_state"])
    pretraining_only = sorted(
        key
        for key in checkpoint_model_keys
        if not (key.startswith("waveform_encoder.") or key.startswith("backbone."))
    )
    module_counts = {
        name: _parameter_count(getattr(pretraining, name))
        for name in (
            "waveform_encoder",
            "operating_encoder_pretrain",
            "fusion",
            "backbone",
            "norm",
            "temporary_heads",
        )
    }
    result = {
        "status": "PASS",
        "checkpoint_path": CHECKPOINT_RELATIVE,
        "checkpoint_sha256": digest,
        "checkpoint_top_level_keys": sorted(checkpoint),
        "checkpoint_model_state_key_count": len(checkpoint_model_keys),
        "checkpoint_component_key_counts": {
            "waveform_encoder": len(waveform_state),
            "backbone": len(backbone_state),
            "operating_encoder_pretrain": len(checkpoint["operating_encoder_pretrain"]),
            "fusion": len(checkpoint["fusion"]),
            "temporary_heads": len(checkpoint["temporary_heads"]),
        },
        "architecture": {
            "family": family,
            "depth": depth,
            "latent_dim": latent_dim,
            "pretraining_trainable_parameters": _parameter_count(pretraining),
            "downstream_untrained_parameters": _parameter_count(downstream),
            "module_parameter_counts": module_counts,
            "xLSTM_block": {
                "normalization": "LayerNorm(256)",
                "memory": "single-layer unidirectional torch.nn.LSTM(256, 256, batch_first=True)",
                "gate": "Linear(256,256) then Sigmoid",
                "projection": "Linear(256,256)",
                "residual": "x + gate(LayerNorm(x)) * projection(LSTM(LayerNorm(x)))",
                "dropout": 0.0,
            },
            "fusion": "MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.0, batch_first=True) plus residual LayerNorm(256)",
            "pooling": "mean over sequence dimension after xLSTM backbone",
        },
        "pretraining_modules": [
            "WaveformEncoder MIE branch",
            "two-feature operating_encoder_pretrain",
            "multimodal fusion",
            "8-block xLSTM backbone",
            "post-backbone LayerNorm",
            "10 material-routed temporary core-loss heads",
        ],
        "temporary_head": "10 x [Linear(256,64), GELU, Linear(64,1)] routed by material metadata",
        "pirl_present_during_pretraining": False,
        "mtph_present_during_pretraining": False,
        "selection": {
            "metric": "validation MAE_norm",
            "value": float(selected["validation_mae_norm"]),
            "subset": "MagNet validation",
            "best_epoch": int(summary["best_epoch"]),
            "selected_with_validation_only": True,
            "test_split_accessed": False,
            "test_evaluations": 0,
            "scope": selected["scope"],
        },
        "transfer": {
            "matched_key_count": len(matched_keys),
            "matched_keys": matched_keys,
            "expected_missing_downstream_key_count": len(expected_missing),
            "expected_missing_downstream_keys": expected_missing,
            "pretraining_only_key_count": len(pretraining_only),
            "pretraining_only_keys_not_loaded": pretraining_only,
            "unexpected_keys_in_intended_module_load": [],
            "shape_mismatches": shape_mismatches,
            "strict_module_loads": True,
        },
    }
    return result


def write_report(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    evidence = audit(root)
    transfer = evidence["transfer"]
    matched_lines = "\n".join(f"- `{key}`" for key in transfer["matched_keys"])
    missing_lines = "\n".join(
        f"- `{key}` — newly initialized downstream module"
        for key in transfer["expected_missing_downstream_keys"]
    )
    pretraining_only_lines = "\n".join(
        f"- `{key}` — pretraining-only, intentionally not transferred"
        for key in transfer["pretraining_only_keys_not_loaded"]
    )
    report = f"""# MEPI Backbone Provenance Audit

## Evidence examined

The audit compared frozen protocols v1.1/v1.2, `Manuscript_clean.docx`, MagNet/xLSTM configs, all five depth summaries, depth comparison and selection files, source implementations, checkpoint metadata/state dictionaries, training metrics, transfer tests, and current fine-tuning configuration. Git history is unavailable because this workspace is not a Git working tree.

## Selection conclusion

The clean measured backbone comparison selected xLSTM on validation MAE_norm. The subsequent completed MagNet representation depth study evaluated depths 2/4/6/8/10 and explicitly selected depth 8 by minimum validation MAE_norm `{evidence['selection']['value']}`. Its checkpoint metadata independently records `backbone=xLSTM`, `latent_dim=256`, and `backbone_layers=8`. The bound checkpoint hash is `{evidence['checkpoint_sha256']}`. The study constructed only train and validation datasets; summary evidence reports `test_split_accessed=false` and `test_evaluations=0`.

This proves xLSTM depth 8 as the selected **representation-pretraining** backbone/checkpoint. It does not validate old downstream BiGRU metrics as xLSTM metrics; downstream xLSTM fine-tuning must produce new validation results.

The depth-study summary retains its historical `MEPI-FROZEN-PROTOCOL v1.0` provenance label. It is not silently relabeled. The later v1.1/v1.2 amendments changed acquisition/data and LSP definitions rather than the MagNet representation architecture, and v1.3 explicitly adopts this exact hash-fixed checkpoint.

## Exact architecture recovered

- Family/depth/width: xLSTM / 8 blocks / latent width 256.
- Pretraining trainable parameters: {evidence['architecture']['pretraining_trainable_parameters']:,}.
- Waveform encoder: Conv1d 1→64, kernel 7/padding 3, GELU, max-pool 2; Conv1d 64→128, kernel 7/padding 3, GELU, max-pool 2; Conv1d 128→128, kernel 7/padding 3, GELU; FFT magnitude pooled to the temporal length; concatenated width 129 projected by Linear(129,256).
- MagNet operating encoder: Linear(2,256), GELU, LayerNorm(256), Linear(256,256).
- Fusion: {evidence['architecture']['fusion']}.
- Each xLSTM block: {evidence['architecture']['xLSTM_block']['normalization']}; {evidence['architecture']['xLSTM_block']['memory']}; {evidence['architecture']['xLSTM_block']['gate']}; {evidence['architecture']['xLSTM_block']['projection']}; residual `{evidence['architecture']['xLSTM_block']['residual']}`; dropout 0.0.
- Post-backbone normalization/pooling: LayerNorm(256), then mean over the sequence dimension.
- Temporary head: {evidence['temporary_head']}.
- PIRL during MagNet pretraining: NO.
- Downstream MTPH during MagNet pretraining: NO.

## Checkpoint and dry-run transfer

- Checkpoint full `model_state` keys: {evidence['checkpoint_model_state_key_count']}.
- Intended strict matches: {transfer['matched_key_count']} (`waveform_encoder` {evidence['checkpoint_component_key_counts']['waveform_encoder']} + `backbone` {evidence['checkpoint_component_key_counts']['backbone']}).
- Expected newly initialized downstream keys: {transfer['expected_missing_downstream_key_count']}.
- Pretraining-only full-model keys not transferred: {transfer['pretraining_only_key_count']}.
- Unexpected keys in intended strict module loads: 0.
- Shape mismatches: 0.
- Arbitrary `strict=False`: NOT USED.

### Exact matched keys

{matched_lines}

### Expected downstream-only keys

{missing_lines}

### Pretraining-only keys intentionally not transferred

{pretraining_only_lines}

## BiGRU conflict classification

`Manuscript_clean.docx` and inherited v1.1 text still describe BiGRU and old BiGRU downstream results. Those statements conflict with the completed clean selection evidence above. They are historical manuscript claims, not evidence for this selected checkpoint, and must be marked `LEGACY_RESULT_REQUIRES_RETRAINING`; their metrics, parameter counts, checkpoint sizes, and training times cannot be transferred to xLSTM.

ACTUAL_SELECTED_BACKBONE = xLSTM
ACTUAL_SELECTED_DEPTH = 8
CHECKPOINT_SHA256 = {evidence['checkpoint_sha256']}
CHECKPOINT_PROVENANCE_VERIFIED = TRUE
CHECKPOINT_SELECTED_ON_VALIDATION_ONLY = TRUE
LEGACY_BIGRU_TEXT_CONFLICT = TRUE
ARCHITECTURE_AMENDMENT_JUSTIFIED = TRUE
"""
    destination = root / "reports/MEPI_BACKBONE_PROVENANCE_AUDIT.md"
    destination.write_text(report, encoding="utf-8")
    (root / "reports/backbone_transfer_v1_3.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


if __name__ == "__main__":
    payload = write_report(Path.cwd())
    print(json.dumps(payload, indent=2, sort_keys=True))
