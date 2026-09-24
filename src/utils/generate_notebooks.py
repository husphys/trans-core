from __future__ import annotations

import json
from pathlib import Path


HEADER = """from pathlib import Path
import os, sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'configs' / 'paths.yaml').exists():
    raise RuntimeError('Open this notebook with the project root as the working directory.')
sys.path.insert(0, str(PROJECT_ROOT))
"""


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


NOTEBOOKS = {
    "00_environment_and_download.ipynb": [
        markdown("# 00 — Environment and source download\n\nRecords the runtime and downloads only missing/partial source files."),
        code(HEADER),
        code(
            """import json, shutil, subprocess
from src.utils.provenance import runtime_record

print(json.dumps(runtime_record(PROJECT_ROOT), indent=2))
print('nvidia-smi:', shutil.which('nvidia-smi'))
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi'], check=False)
"""
        ),
        code(
            """from src.data.download import download_sources

DOWNLOAD_IF_MISSING = True
if DOWNLOAD_IF_MISSING:
    download_sources('configs/paths.yaml')
"""
        ),
    ],
    "01_data_audit.ipynb": [
        markdown("# 01 — Immutable data audit\n\nCreates SHA-256 and schema/target-quality evidence without modifying raw files."),
        code(HEADER),
        code(
            """from src.data.manifest import build_manifest
from src.data.audit import run_audit

print(build_manifest('configs/paths.yaml'))
audit = run_audit('configs/paths.yaml', deep_duplicates=True)
print({'magnet_files': len(audit['magnet']), 'csv_files': len(audit['csv'])})
"""
        ),
    ],
    "02_create_splits.ipynb": [
        markdown("# 02 — Leakage-controlled splits\n\nCreates group-wise 70:15:15 manifests with seed 42 and asserts zero group overlap."),
        code(HEADER),
        code("""import runpy, sys
original_argv = sys.argv[:]
try:
    sys.argv = ['src.data.splits']
    runpy.run_module('src.data.splits', run_name='__main__')
finally:
    sys.argv = original_argv
"""),
    ],
    "03_pretrain_backbones.ipynb": [
        markdown("# 03 — Eight-backbone pretraining\n\nPhase B. Full training stays closed until Phase A gates pass."),
        code(HEADER),
        code(
            """from src.utils.config import load_yaml, require_full_training_enabled
config = load_yaml('configs/pretrain.yaml')
require_full_training_enabled(config)
raise NotImplementedError('Phase B runner is intentionally unavailable until Phase A is accepted.')
"""
        ),
    ],
    "04_bigru_depth_experiment.ipynb": [
        markdown("# 04 — BiGRU depth experiment\n\nPhase C: 2, 4, 6, 8, and 10 layers; validation-only selection."),
        code(HEADER),
        code(
            """from src.utils.config import load_yaml, require_full_training_enabled
config = load_yaml('configs/finetune.yaml')
require_full_training_enabled(config)
raise NotImplementedError('Depth training is gated by the unresolved fine-tune target audit.')
"""
        ),
    ],
    "05_lambda_sensitivity.ipynb": [
        markdown("# 05 — Lambda sensitivity\n\nPhase C: the fixed 4 x 3 grid on the validation-selected depth."),
        code(HEADER),
        code(
            """from src.utils.config import load_yaml, require_full_training_enabled
config = load_yaml('configs/finetune.yaml')
require_full_training_enabled(config)
raise NotImplementedError('Lambda training is gated by the unresolved fine-tune target audit.')
"""
        ),
    ],
    "06_final_training_and_test.ipynb": [
        markdown("# 06 — Frozen final configuration and single test evaluation\n\nThe test manifest is not loaded until every selection is frozen."),
        code(HEADER),
        code(
            """from src.utils.config import load_yaml, require_full_training_enabled
config = load_yaml('configs/finetune.yaml')
require_full_training_enabled(config)
raise NotImplementedError('Final test evaluation is gated until selection artifacts are frozen.')
"""
        ),
    ],
    "07_tables_and_figures.ipynb": [
        markdown("# 07 — Tables, figures, and traceability\n\nConsumes completed result artifacts only; it never trains a model."),
        code(HEADER),
        code(
            """required = [
    PROJECT_ROOT / 'results' / 'tables',
    PROJECT_ROOT / 'results' / 'figures',
    PROJECT_ROOT / 'results' / 'reports' / 'result_traceability.md',
]
for path in required:
    print(path, 'present' if path.exists() else 'MISSING')
print('No legacy number is copied into a manuscript output before rerun evidence exists.')
"""
        ),
    ],
}


def main() -> None:
    output = Path("notebooks")
    output.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        payload = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python (transformer-repro)",
                    "language": "python",
                    "name": "transformer-repro",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (output / filename).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
