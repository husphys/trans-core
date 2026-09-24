#!/usr/bin/env python3
"""Generate the single manual xLSTM depth notebook; never execute training."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (trans-core)",
                "language": "python",
                "name": "trans-core",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


CELLS = [
    markdown(
        "# A. Purpose of experiment\n\n"
        "Manual **Run All** xLSTM depth study replacing the scientific role of the legacy BiGRU depth study. "
        "Only the MagNet representation-pretraining stage is executable now. The test split is never accessed. "
        "The downstream confirmation remains blocked until real fine-tuning data are available.\n"
    ),
    code(
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().resolve()\n"
        "RUN_DEPTHS = [2, 4, 6, 8, 10]\n"
        "FORCE_RETRAIN = False\n"
        "AUTO_RESUME = True\n"
        "print('PASS — manual controls loaded')\n"
        "print(f'RUN_DEPTHS={RUN_DEPTHS}')\n"
        "print(f'FORCE_RETRAIN={FORCE_RETRAIN} (applies only to depths explicitly listed in RUN_DEPTHS)')\n"
        "print(f'AUTO_RESUME={AUTO_RESUME}')\n"
        "output_root = PROJECT_ROOT / 'experiments/xlstm_depth_v1'\n"
        "has_results = output_root.exists() and any(output_root.rglob('training_summary.json'))\n"
        "print(f\"DEPTH EXPERIMENT STATUS: {'RECOVERY/COMPLETED ARTIFACTS FOUND' if has_results else 'NEW RUN'}\")\n"
    ),
    markdown("# B. Environment verification\n"),
    code(
        "import os\nimport sys\nimport numpy as np\nimport torch\n"
        "REQUIRED_INTERPRETER = os.environ.get('MEPI_REQUIRED_INTERPRETER')\n"
        "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'UNAVAILABLE'\n"
        "print(f'sys.executable: {sys.executable}')\n"
        "print(f'Python version: {sys.version.replace(chr(10), chr(32))}')\n"
        "print(f'PyTorch version: {torch.__version__}')\n"
        "print(f'CUDA build: {torch.version.cuda}')\n"
        "print(f'CUDA available: {torch.cuda.is_available()}')\n"
        "print(f'GPU name: {gpu_name}')\n"
        "print(f'NumPy version: {np.__version__}')\n"
        "if REQUIRED_INTERPRETER and str(Path(sys.executable).resolve()) != str(Path(REQUIRED_INTERPRETER).resolve()):\n"
        "    print('FAIL — wrong interpreter')\n"
        "    raise RuntimeError('Select Python (trans-core), restart the kernel, then Run All')\n"
        "if not torch.cuda.is_available():\n"
        "    print('FAIL — CUDA is required for the scientific run')\n"
        "    raise RuntimeError('CUDA is unavailable in Python (trans-core); stop before training')\n"
        "print('PASS — required Python (trans-core) interpreter and CUDA are active')\n"
    ),
    markdown("# C. Frozen-protocol verification\n"),
    code(
        "from src.mepi_v1.reproducibility import sha256_file\n"
        "EXPECTED_PROTOCOL_SHA256 = 'a515052cd2f8cf2970b731cfb48dee055c316a5d4acc0f0ca90313436c344c0f'\n"
        "protocol_paths = [PROJECT_ROOT / 'MEPI-FROZEN-PROTOCOL v1.1.md', PROJECT_ROOT / 'docs/MEPI-FROZEN-PROTOCOL v1.1.md']\n"
        "protocol_hashes = [sha256_file(path) for path in protocol_paths]\n"
        "passed = protocol_hashes == [EXPECTED_PROTOCOL_SHA256, EXPECTED_PROTOCOL_SHA256]\n"
        "print(f\"{'PASS' if passed else 'FAIL'} — frozen protocol hashes: {protocol_hashes}\")\n"
        "if not passed:\n    raise RuntimeError('Frozen protocol is missing or changed; training is blocked')\n"
    ),
    markdown("# D. Dataset and split verification\n"),
    code(
        "from src.mepi_v1.xlstm_depth import XLSTMDepthRun\n"
        "probe = XLSTMDepthRun(PROJECT_ROOT, 2)\n"
        "dataset_check = probe.validate_dataset()\n"
        "assert dataset_check['waveform_shape'][1] == 1024\n"
        "assert dataset_check['test_split_accessed'] is False\n"
        "print('PASS — B(t) has 1024 points; only train and validation manifests were inspected')\n"
    ),
    markdown("# E. Dataset and split fingerprints\n"),
    code(
        "EXPECTED_DATASET_SHA256 = '0aca43184e9bd6cc90eec736a89fe1ea32f45b126527cb97918cd92684ac2c19'\n"
        "EXPECTED_TRAIN_SHA256 = '2e06050bbe1cff8b6f4f7ae3a99788d6914e214353a5e6a2f2ece7463b6377b0'\n"
        "EXPECTED_VALIDATION_SHA256 = 'aa7d1ee30ca3f3967eadee81d2a31e0a9bcb7e0bc5421f36cb849d6da1dc9749'\n"
        "prepared = probe._prepare()\n"
        "actual = {\n"
        "    'dataset': prepared['compatibility']['dataset_fingerprint'],\n"
        "    'train': sha256_file(prepared['manifests']['train']),\n"
        "    'validation': sha256_file(prepared['manifests']['validation']),\n"
        "}\n"
        "expected = {'dataset': EXPECTED_DATASET_SHA256, 'train': EXPECTED_TRAIN_SHA256, 'validation': EXPECTED_VALIDATION_SHA256}\n"
        "print(f\"{'PASS' if actual == expected else 'FAIL'} — fingerprints: {actual}\")\n"
        "if actual != expected:\n    raise RuntimeError('Frozen dataset or train/validation manifests changed')\n"
        "print('PASS — MagNet test manifest was not read or fingerprinted')\n"
    ),
    markdown("# F. xLSTM baseline architecture\n"),
    code(
        "from src.mepi_v1.backbones import XLSTMBackbone\n"
        "from src.mepi_v1.models import PretrainingModel\n"
        "baseline = PretrainingModel('xLSTM', material_count=10, latent_dim=256, backbone_layers=2)\n"
        "assert isinstance(baseline.backbone, XLSTMBackbone)\n"
        "print('PASS — waveform encoder -> operating encoder -> fusion -> stacked xLSTM blocks -> temporary material-routed heads')\n"
        "print('PASS — material identity is not a predictive input; PIRL and MTPH are absent')\n"
    ),
    markdown("# G. Controlled variables\n"),
    code(
        "fixed = {\n"
        " 'waveform_length': 1024, 'predictive_tabular': ['frequency', 'temperature'],\n"
        " 'latent_dim': 256, 'epochs': 10, 'batch_size': probe.config['training']['batch_size'],\n"
        " 'optimizer': 'AdamW', 'learning_rate': probe.config['training']['learning_rate'],\n"
        " 'scheduler': 'constant LambdaLR', 'dataloader': {k: probe.config['training'].get(k) for k in ['num_workers','pin_memory','persistent_workers','prefetch_factor','cache_mode']},\n"
        " 'amp': probe.config['training']['amp'], 'tf32': probe.config['runtime']['allow_tf32'],\n"
        " 'seed': probe.seed, 'target_transform': prepared['preprocessing']['target'],\n"
        "}\n"
        "print('PASS — depth is the only varied model/configuration value')\nfixed\n"
    ),
    markdown("# H. Depth candidates [2, 4, 6, 8, 10]\n"),
    code(
        "from src.mepi_v1.xlstm_depth import DEPTHS\n"
        "invalid = sorted(set(RUN_DEPTHS) - set(DEPTHS))\n"
        "print(f\"{'PASS' if not invalid else 'FAIL'} — canonical DEPTHS={list(DEPTHS)}; requested RUN_DEPTHS={RUN_DEPTHS}\")\n"
        "if invalid:\n    raise ValueError(f'Unsupported depth values: {invalid}')\n"
    ),
    markdown("# I. Parameter-count preview\n"),
    code(
        "import pandas as pd\n"
        "preview = []\n"
        "for depth in DEPTHS:\n"
        "    model = PretrainingModel('xLSTM', material_count=10, latent_dim=256, backbone_layers=depth)\n"
        "    preview.append({'depth': depth, 'parameter_count': sum(p.numel() for p in model.parameters()), 'block_count': len(model.backbone.blocks)})\n"
        "preview_df = pd.DataFrame(preview)\n"
        "assert preview_df['parameter_count'].is_monotonic_increasing\n"
        "print('PASS — block count equals depth and parameter count increases with depth')\npreview_df\n"
    ),
    markdown("# J. Training setup\n\nEach candidate has its own visible cell below. Compatible completed candidates are skipped; interrupted candidates resume automatically from `last_checkpoint.pt`.\n"),
    code(
        "import torch\n"
        "from src.mepi_v1.xlstm_depth import official_pretraining_snapshot\n"
        "official_before = official_pretraining_snapshot(PROJECT_ROOT)\n"
        "depth_results = {}\n"
        "def run_manual_depth(depth):\n"
        "    if depth not in RUN_DEPTHS:\n"
        "        print(f'Depth {depth}/10: SKIPPED BY RUN_DEPTHS')\n"
        "        return None\n"
        "    run = XLSTMDepthRun(PROJECT_ROOT, depth, force_retrain=FORCE_RETRAIN, auto_resume=AUTO_RESUME)\n"
        "    status = run.inspect_status()\n"
        "    if status['status'] == 'COMPLETE':\n"
        "        summary = run.validate_completed_compatibility()\n"
        "        print(f'Depth {depth}/10: COMPLETE — skipped safely')\n"
        "        depth_results[depth] = summary\n"
        "        return summary\n"
        "    print(f\"Depth {depth}/10: {'RESUME' if status['status'] == 'INCOMPLETE' else 'FROM SCRATCH'}\")\n"
        "    run.check_environment(require_cuda=True)\n"
        "    run.validate_source_protocol()\n"
        "    run.validate_dataset()\n"
        "    run.construct_model_smoke()\n"
        "    run.inspect_checkpoint()\n"
        "    torch.cuda.reset_peak_memory_stats()\n"
        "    run.train_or_resume()\n"
        "    summary = run.finalize_validation_only(peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()))\n"
        "    run.write_depth_evidence(summary, official_before)\n"
        "    if official_pretraining_snapshot(PROJECT_ROOT) != official_before:\n"
        "        raise RuntimeError('Official pretraining evidence changed during the depth run')\n"
        "    print(f\"PASS — Depth {depth} complete; validation MAE_norm={summary['validation_metrics']['normalized_mae']}\")\n"
        "    print(f\"Checkpoint saved: {summary['best_checkpoint']}\")\n"
        "    depth_results[depth] = summary\n"
        "    return summary\n"
        "print('PASS — manual training helper ready; no candidate has been launched by this setup cell')\n"
    ),
    markdown("# K. Depth-2 training\n"),
    code("depth_2_summary = run_manual_depth(2)\n"),
    markdown("# L. Depth-4 training\n"),
    code("depth_4_summary = run_manual_depth(4)\n"),
    markdown("# M. Depth-6 training\n"),
    code("depth_6_summary = run_manual_depth(6)\n"),
    markdown("# N. Depth-8 training\n"),
    code("depth_8_summary = run_manual_depth(8)\n"),
    markdown("# O. Depth-10 training\n"),
    code("depth_10_summary = run_manual_depth(10)\n"),
    markdown("# P. Validation comparison\n"),
    code(
        "from src.mepi_v1.xlstm_depth import summarize_depth_experiment\n"
        "comparison = summarize_depth_experiment(PROJECT_ROOT)\n"
        "comparison_df = pd.DataFrame(comparison).rename(columns={'training_runtime_seconds': 'runtime', 'best_checkpoint_sha256': 'checkpoint_sha256'})\n"
        "display_columns = ['depth','parameter_count','best_epoch','validation_mae_norm','validation_rmse','validation_r2','runtime','peak_gpu_memory_gib','checkpoint_sha256','training_horizon_boundary_reached']\n"
        "completed_df = comparison_df[comparison_df['status'] == 'COMPLETE'][display_columns].sort_values('validation_mae_norm')\n"
        "print('PASS — comparison uses validation metrics only and is sorted by validation MAE_norm')\ncompleted_df\n"
    ),
    markdown("# Q. Selected depth\n"),
    code(
        "if len(completed_df) == len(DEPTHS):\n"
        "    selected_depth = int(completed_df.iloc[0]['depth'])\n"
        "    print(f'SELECTED xLSTM DEPTH = {selected_depth}')\n"
        "    print('SELECTION CRITERION =\\nminimum validation MAE_norm')\n"
        "    print('TEST DATA ACCESSED =\\nNO')\n"
        "else:\n"
        "    selected_depth = None\n"
        "    print(f'PENDING — {len(completed_df)}/{len(DEPTHS)} candidates are complete; no depth selected')\n"
    ),
    markdown("# R. Reproducibility and evidence summary\n"),
    code(
        "print('PASS — reproducibility/evidence summary')\n"
        "print(f'Official pretraining tree SHA-256: {official_before[\"tree_sha256\"]}')\n"
        "print(f'Dataset SHA-256: {EXPECTED_DATASET_SHA256}')\n"
        "print(f'Train manifest SHA-256: {EXPECTED_TRAIN_SHA256}')\n"
        "print(f'Validation manifest SHA-256: {EXPECTED_VALIDATION_SHA256}')\n"
        "print('Test manifest/dataset access: NO')\n"
        "print('A best_epoch equal to 10 is reported as TRAINING_HORIZON_BOUNDARY_REACHED=True; no run is extended.')\n"
        "completed_df\n"
    ),
    markdown("# S. Downstream status\n"),
    code(
        "if selected_depth is not None:\n    print('MAGNET DEPTH STUDY: COMPLETE')\n"
        "else:\n    print('MAGNET DEPTH STUDY: INCOMPLETE')\n"
        "print('DOWNSTREAM DEPTH CONFIRMATION:')\n"
        "print('WAITING FOR REAL FINETUNE DATA')\n"
        "print('No downstream fine-tuning, loss-weight, final-test, or frequency-screening result was generated here.')\n"
    ),
]


def main() -> None:
    destination = ROOT / "notebooks/21_xlstm_depth_experiment.ipynb"
    destination.write_text(
        json.dumps(notebook(CELLS), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
