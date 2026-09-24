#!/usr/bin/env python3
"""Generate downstream preparation notebooks without executing scientific work."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def md(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value.splitlines(keepends=True),
    }


def intro(title: str, purpose: str, status: str, prerequisites: str, paths: str) -> list[dict[str, object]]:
    return [
        md(
            f"# {title}\n\n"
            f"**Purpose:** {purpose}\n\n"
            f"**Scientific status:** {status}\n\n"
            f"**Prerequisites:** {prerequisites}\n\n"
            f"**Dataset/evidence paths:** {paths}\n"
        ),
        code(
            "from pathlib import Path\nimport os\nimport sys\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if PROJECT_ROOT.name == 'notebooks':\n    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "FORCE_RETRAIN = False\nAUTO_RESUME = True\n"
            "REQUIRED_INTERPRETER = os.environ.get('MEPI_REQUIRED_INTERPRETER')\n"
            "print(f'sys.executable: {sys.executable}')\n"
            "print('Required kernel: Python (trans-core)')\n"
            "print(f'FORCE_RETRAIN={FORCE_RETRAIN}; AUTO_RESUME={AUTO_RESUME}')\n"
            "if REQUIRED_INTERPRETER and str(Path(sys.executable).resolve()) != str(Path(REQUIRED_INTERPRETER).resolve()):\n"
            "    raise RuntimeError('Select Python (trans-core), Restart Kernel, then Run All')\n"
            "print('ENVIRONMENT CHECK: PASS')\n"
        ),
    ]


def payload(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (trans-core)", "language": "python", "name": "trans-core"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS: dict[str, list[dict[str, object]]] = {}

NOTEBOOKS["22_xlstm_transfer_audit.ipynb"] = intro(
    "xLSTM depth-checkpoint transfer audit",
    "Verify all five MagNet xLSTM depth checkpoints transfer only the waveform encoder and xLSTM backbone.",
    "AUDIT ONLY; no scientific fine-tuning.",
    "Completed Notebook 21 depth artifacts for depths 2, 4, 6, 8, and 10.",
    "experiments/xlstm_depth_v1/depth_{2,4,6,8,10}/best_checkpoint.pt",
) + [
    code(
        "from src.mepi_v1.checkpointing import audit_xlstm_transfer_checkpoint\n"
        "DEPTHS = (2, 4, 6, 8, 10)\nreports = []\n"
        "for depth in DEPTHS:\n"
        "    try:\n"
        "        report = audit_xlstm_transfer_checkpoint(PROJECT_ROOT, depth)\n"
        "        reports.append(report)\n"
        "        print(f'Depth {depth} checkpoint: {report[\"checkpoint_path\"]}')\n"
        "        print(f'SHA256: {report[\"checkpoint_sha256\"]}')\n"
        "        print('Waveform encoder keys:', report['waveform_encoder_keys'])\n"
        "        print('xLSTM backbone keys:', report['backbone_keys'])\n"
        "        print('Temporary material-head keys (discarded):', report['temporary_material_head_keys'])\n"
        "        print('MagNet 2-feature encoder keys (discarded):', report['magnet_operating_encoder_keys'])\n"
        "        print('Transferred tensor shapes:', report['transferred_shapes'])\n"
        "        print('Discarded tensor shapes:', report['discarded_shapes'])\n"
        "        print(f'TRANSFER AUDIT: PASS — depth {depth}')\n"
        "    except Exception as error:\n"
        "        print(f'TRANSFER AUDIT: FAIL — depth {depth}: {error}')\n"
        "        raise\n"
        "if len(reports) != 5:\n    raise RuntimeError('Transfer audit did not pass for all five depths')\n"
        "print('5/5 DEPTH CHECKPOINTS TRANSFER-READY')\n"
    )
]

NOTEBOOKS["30_finetune_dataset_audit.ipynb"] = intro(
    "Real fine-tuning dataset audit",
    "Audit the physical two-core dataset, targets, raw waveforms, leakage, group split, and train-only preparation before training.",
    "WAITING FOR REAL FINETUNE DATA.",
    "Set dataset.path and authoritative physical constants in configs/finetune_v1.yaml.",
    "configs/finetune_v1.yaml; experiments/downstream_v1/manifests; experiments/downstream_v1/status/30_dataset_audit.json",
) + [
    code(
        "from src.mepi_v1.downstream_audit import audit_finetune_dataset\n"
        "from src.mepi_v1.workflow_guards import write_status\n"
        "CONFIG_PATH = PROJECT_ROOT / 'configs/finetune_v1.yaml'\n"
        "audit = audit_finetune_dataset(CONFIG_PATH)\n"
        "if audit['status'] == 'WAITING_FOR_REAL_FINETUNE_DATA':\n"
        "    print('DATASET STATUS:')\n    print('WAITING FOR REAL FINETUNE DATA')\n"
        "    print('No manifests, processed scientific values, or synthetic data were created.')\n"
        "    write_status(PROJECT_ROOT, '30_dataset_audit', {'status': 'WAITING_FOR_REAL_FINETUNE_DATA', 'blockers': audit['blockers']})\n"
        "elif audit['status'] == 'PASS':\n"
        "    write_status(PROJECT_ROOT, '30_dataset_audit', audit)\n"
        "    print('DATASET STATUS: PASS')\n    print(audit)\n"
        "else:\n"
        "    write_status(PROJECT_ROOT, '30_dataset_audit', audit)\n"
        "    print('DATASET STATUS: FAIL')\n    print(audit)\n"
        "    raise RuntimeError('Fine-tuning dataset audit failed closed')\n"
    )
]

NOTEBOOKS["31_xlstm_downstream_depth_experiment.ipynb"] = intro(
    "Downstream xLSTM depth experiment",
    "Run the manuscript-style downstream comparison at depths 2, 4, 6, 8, and 10, each from its matching MagNet checkpoint.",
    "PREPARED; scientific execution blocked until Notebook 30 is PASS.",
    "30_finetune_dataset_audit = PASS; real data and constants; transfer audit PASS.",
    "experiments/downstream_v1/manifests; experiments/xlstm_depth_v1; experiments/downstream_v1/depth_*",
) + [
    code(
        "from src.mepi_v1.workflow_guards import require_prerequisites\n"
        "checks = require_prerequisites(PROJECT_ROOT, '31_downstream_depth')\n"
        "print('DEPENDENCY GUARD: PASS', checks)\n"
    ),
    code(
        "from src.mepi_v1.checkpointing import audit_xlstm_transfer_checkpoint\n"
        "DEPTHS = (2, 4, 6, 8, 10)\nEPOCHS_PER_DEPTH = 50\n"
        "checkpoint_map = {depth: PROJECT_ROOT / f'experiments/xlstm_depth_v1/depth_{depth}/best_checkpoint.pt' for depth in DEPTHS}\n"
        "for depth, path in checkpoint_map.items():\n"
        "    audit_xlstm_transfer_checkpoint(PROJECT_ROOT, depth)\n"
        "    print(f'MagNet depth {depth} -> downstream depth {depth}: {path}')\n"
        "print('PASS — five one-to-one depth/checkpoint mappings; equal 50-epoch maximum budgets')\n"
        "print('TEST SPLIT ACCESS: FORBIDDEN')\n"
    ),
    code(
        "def report_epoch(depth, epoch, train_loss, validation_metrics, learning_rate, elapsed_seconds, gpu_memory_gib):\n"
        "    print(f'Depth {depth} | Epoch {epoch}/50 | train_loss={train_loss:.6g} | validation={validation_metrics} | lr={learning_rate:.3g} | elapsed_s={elapsed_seconds:.1f} | gpu_GiB={gpu_memory_gib:.3f}')\n"
        "print('VISIBLE EPOCH PROGRESS: READY')\n"
        "print('CRASH-SAFE RESUME: last_checkpoint.pt must contain model/optimizer/scheduler/AMP/RNG state')\n"
    ),
    md(
        "## Manual training cells\n\n"
        "Each candidate must use the audited train/validation manifests, train-fitted preprocessing, a new nine-feature encoder/fusion/PIRL/MTPH, and its mapped pretrained waveform encoder/backbone. "
        "The implementation remains fail-closed until Notebook 30 freezes the real dataset interface and required physics metadata. Per-epoch output must show epoch/50, training loss, validation manuscript metrics, learning rate, elapsed time, and GPU memory. Crash-safe `last_checkpoint.pt` resume is mandatory."
    ),
    code(
        "print('BEST MAGNET DEPTH: 8')\n"
        "print('FINAL DOWNSTREAM xLSTM DEPTH: NOT YET SELECTED')\n"
        "print('DEPTH SELECTION RULE STATUS: AMBIGUOUS_MANUSCRIPT_RULE')\n"
        "raise RuntimeError('Stop after candidate metrics: the manuscript does not define one deterministic multi-metric depth-selection rule; do not invent a scalar score')\n"
    ),
]

NOTEBOOKS["33_loss_weight_sensitivity.ipynb"] = intro(
    "Loss-weight sensitivity",
    "Reproduce only the manuscript's 12 lambda configurations using validation for selection.",
    "PREPARATION ONLY; no test access.",
    "Notebook 31 COMPLETE and final downstream xLSTM depth frozen.",
    "configs/finetune_v1.yaml; experiments/downstream_v1/status/31_downstream_depth.json",
) + [
    code(
        "from src.mepi_v1.workflow_guards import require_prerequisites\n"
        "require_prerequisites(PROJECT_ROOT, '33_loss_weight')\n"
        "LAMBDA_1 = (1.0,)\nLAMBDA_2 = (1.0,)\nLAMBDA_3 = (0.1, 0.3, 0.5, 1.0)\nLAMBDA_4 = (0.05, 0.10, 0.20)\n"
        "grid = [(a,b,c,d) for a in LAMBDA_1 for b in LAMBDA_2 for c in LAMBDA_3 for d in LAMBDA_4]\n"
        "assert len(grid) == 12\nprint('LOSS-WEIGHT WORKFLOW READY: 12 MANUSCRIPT CONFIGURATIONS')\n"
        "print('VALIDATION ONLY; TEST ACCESS: FORBIDDEN')\n"
    ),
    code(
        "def report_epoch(lambda_values, epoch, train_loss, validation_metrics):\n"
        "    print(f'Lambdas={lambda_values} | Epoch {epoch}/50 | train_loss={train_loss:.6g} | validation={validation_metrics}')\n"
        "print('VISIBLE EPOCH PROGRESS AND CRASH-SAFE RESUME: REQUIRED FOR EACH OF 12 RUNS')\n"
    ),
]

NOTEBOOKS["34_final_training_and_test.ipynb"] = intro(
    "Final training and exactly-once test evaluation",
    "Train the fully frozen final xLSTM MEPI configuration and access test only for final evaluation.",
    "PREPARATION ONLY.",
    "Notebook 33 COMPLETE; dataset, split, depth, weights, and architecture all frozen.",
    "experiments/downstream_v1/status/33_loss_weight.json; experiments/downstream_v1/final",
) + [
    code(
        "def report_epoch(epoch, train_loss, validation_metrics):\n"
        "    print(f'Final model | Epoch {epoch}/50 | train_loss={train_loss:.6g} | validation={validation_metrics}')\n"
        "print('VISIBLE EPOCH PROGRESS AND CRASH-SAFE RESUME: READY')\n"
    ),
    code(
        "from src.mepi_v1.workflow_guards import require_prerequisites, require_test_access\n"
        "require_prerequisites(PROJECT_ROOT, '34_final_model')\n"
        "FINAL_CONFIGURATION_FROZEN = False\n"
        "print(f'FINAL CONFIGURATION FROZEN: {\"YES\" if FINAL_CONFIGURATION_FROZEN else \"NO\"}')\n"
        "require_test_access(stage='34_final_model', final_configuration_frozen=FINAL_CONFIGURATION_FROZEN)\n"
        "print('TEST ACCESS AUTHORIZED FOR EXACTLY-ONCE FINAL EVALUATION')\n"
    )
]

NOTEBOOKS["35_frequency_screening.ipynb"] = intro(
    "MEPI-assisted frequency screening",
    "Preserve the manuscript application experiment using the final xLSTM-based MEPI model.",
    "PREPARATION ONLY.",
    "Notebook 34 final model COMPLETE.",
    "experiments/downstream_v1/status/34_final_model.json; experiments/downstream_v1/frequency_screening",
) + [
    code(
        "from src.mepi_v1.workflow_guards import require_prerequisites\n"
        "require_prerequisites(PROJECT_ROOT, '35_frequency_screening')\n"
        "print('FREQUENCY SCREENING MODEL FAMILY: xLSTM-based MEPI')\n"
        "print('APPLICATION LOGIC: manuscript-preserved; no redesign or new scientific claim')\n"
    )
]

NOTEBOOKS["36_tables_and_figures.ipynb"] = intro(
    "Tables and figures",
    "Gather measured evidence from screening, both depth stages, loss weights, final evaluation, and frequency screening.",
    "PARTIAL REPORTING IS ALLOWED; unavailable downstream sections remain explicit.",
    "None for immutable upstream evidence; downstream sections require their machine-readable COMPLETE statuses.",
    "experiments/pretrain_v1; experiments/xlstm_depth_v1; experiments/downstream_v1",
) + [
    code(
        "from src.mepi_v1.workflow_guards import read_status\n"
        "sections = {\n"
        " 'official_8_backbone_screening': 'AVAILABLE',\n"
        " 'xlstm_magnet_depth_study': 'AVAILABLE',\n"
        " 'downstream_xlstm_depth': read_status(PROJECT_ROOT, '31_downstream_depth')['status'],\n"
        " 'loss_weight_sensitivity': read_status(PROJECT_ROOT, '33_loss_weight')['status'],\n"
        " 'final_evaluation': read_status(PROJECT_ROOT, '34_final_model')['status'],\n"
        " 'frequency_screening': 'AVAILABLE' if read_status(PROJECT_ROOT, '34_final_model')['status'] == 'COMPLETE' else 'MISSING',\n"
        "}\n"
        "for name, status in sections.items():\n"
        "    print(f'{name}: {status if status in {\"AVAILABLE\", \"COMPLETE\"} else \"WAITING_FOR_REAL_DATA\"}')\n"
        "print('No legacy manuscript value is promoted as a new experimental result.')\n"
    )
]


def main() -> None:
    destination = ROOT / "notebooks"
    for name, cells in NOTEBOOKS.items():
        path = destination / name
        path.write_text(json.dumps(payload(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
