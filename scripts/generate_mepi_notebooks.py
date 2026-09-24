"""Generate the thin, uniform MEPI Run-All notebooks with stdlib JSON only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
PROTOCOL = "MEPI-FROZEN-PROTOCOL v1.1"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def config_cell() -> dict:
    return code(
        "from pathlib import Path\n\n"
        "# MEPI Run-All configuration. Safe recovery is the default.\n"
        "PROJECT_ROOT = Path.cwd().resolve()\n"
        "if PROJECT_ROOT.name == 'notebooks':\n"
        "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
        "if not (PROJECT_ROOT / 'src' / 'mepi_v1').is_dir():\n"
        "    raise RuntimeError('Run this notebook from the repository root or notebooks directory')\n"
        "SEED = 42\n"
        "FORCE_RETRAIN = False\n"
        "AUTO_RESUME = True\n"
        "CHECKPOINT_EVERY_N_STEPS = 1000\n\n"
        'print(f"[PASS] Configuration loaded: PROJECT_ROOT={PROJECT_ROOT}, SEED={SEED}")\n'
        'print(f"FORCE_RETRAIN={FORCE_RETRAIN}; AUTO_RESUME={AUTO_RESUME}; checkpoint interval={CHECKPOINT_EVERY_N_STEPS} steps")\n'
    )


def setup_cell(imports: str) -> dict:
    return code(
        "import sys\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n"
        f"{imports}\n"
        'print("[PASS] Project source imports completed")\n'
    )


def write(name: str, cells: list[dict]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")


def training_notebook(backbone: str) -> list[dict]:
    output = f"experiments/pretrain_v1/{backbone}"
    return [
        markdown(
            f"# MEPI v1 — {backbone} pretraining\n\n"
            f"**Purpose:** train the `{backbone}` candidate using the frozen common MagNet protocol.  \n"
            "**Inputs:** audited MagNet HDF5 plus deterministic train/validation/test manifests.  \n"
            f"**Outputs:** `{output}/` checkpoints, metrics, environment, summary, and measured evidence.  \n"
            f"**Frozen protocol:** {PROTOCOL}.  \n"
            "**Trains a model:** yes. The test split is opened only after the validation-selected best checkpoint is frozen.  \n"
            "**Restart behavior:** a compatible rolling checkpoint resumes automatically; a COMPLETE run skips training."
        ),
        config_cell(),
        markdown("## Setup\n\nImport the shared source workflow. Notebook cells contain orchestration only."),
        setup_cell("from src.mepi_v1.notebook_workflow import PretrainingNotebookRun"),
        markdown("## A — Notebook identity\n\nConfirm the candidate, protocol, output directory, and recovery behavior."),
        code(
            f'run = PretrainingNotebookRun(PROJECT_ROOT, "{backbone}", seed=SEED, force_retrain=FORCE_RETRAIN, auto_resume=AUTO_RESUME, checkpoint_every_n_steps=CHECKPOINT_EVERY_N_STEPS)\n'
            "identity = run.print_identity()\n"
        ),
        markdown("## B — Environment\n\nDisplay Python/PyTorch/CUDA/GPU/project details. This training notebook fails closed without CUDA."),
        code(
            "environment_rows = run.check_environment(require_cuda=True)\n"
            "import pandas as pd\n"
            "display(pd.DataFrame(environment_rows))\n"
        ),
        markdown("## C — Source and frozen protocol\n\nRequire the authoritative protocol and fingerprint the clean source tree."),
        code("source_state = run.validate_source_protocol()\ndisplay(pd.DataFrame([source_state]))\n"),
        markdown("## D — Dataset contract\n\nVerify counts, fingerprints, frozen feature shapes, and exclusion of material/CoreID from predictive inputs."),
        code("dataset_state = run.validate_dataset()\ndisplay(pd.DataFrame([dataset_state]))\n"),
        markdown("## E — Experiment state\n\nChoose NEW, AUTO-RESUME, or COMPLETE/SKIP without prompting."),
        code("experiment_state = run.inspect_status()\ndisplay(pd.DataFrame([{k: v for k, v in experiment_state.items() if k != 'summary'}]))\n"),
        markdown("## F — Recovery checkpoint\n\nValidate every experiment identity field before any state is loaded."),
        code("checkpoint_rows = run.inspect_checkpoint()\ndisplay(pd.DataFrame(checkpoint_rows))\n"),
        markdown("## G — Model and real-batch smoke check\n\nConstruct the candidate and run one actual sample through the frozen input contract."),
        code("model_state = run.construct_model_smoke()\ndisplay(pd.DataFrame([model_state]))\n"),
        markdown("## H — Train or resume\n\nRun the shared recoverable loop. Rolling writes are atomic and only `last_checkpoint.pt` plus `best_checkpoint.pt` may coexist."),
        code("training_state = run.train_or_resume()\ndisplay(pd.DataFrame([training_state]))\n"),
        markdown("## I — Training curves\n\nRender three separate matplotlib figures without seaborn, subplots, or manually assigned colors."),
        code("figures = run.plot_curves()\nprint(f\"[PASS] Curve figures available: {len(figures)}\")\n"),
        markdown("## J — Frozen best-checkpoint evaluation\n\nReload the validation-selected best checkpoint, report validation metrics, then access test exactly once."),
        code("final_summary = run.evaluate_best_checkpoint()\ndisplay(pd.DataFrame([{'split': 'validation', **final_summary['validation_metrics']}, {'split': 'test', **final_summary['test_metrics']}]))\n"),
        markdown("## K — Measured experiment evidence\n\nWrite COMPLETE JSON/Markdown evidence only after training and final evaluation succeeded."),
        code("evidence = run.write_run_evidence()\nprint(f\"[PASS] Evidence status: {evidence['status']}\")\n"),
        markdown("## Final notebook summary\n\nThis screenshot-friendly cell is intentionally last. Save the executed notebook to preserve visible evidence."),
        code("final_evidence = run.display_final_summary()\ndisplay(pd.DataFrame([{'backbone': final_evidence['backbone'], 'status': final_evidence['status'], 'best_epoch': final_evidence['best_epoch'], 'validation_mae_norm': final_evidence['validation_metrics']['normalized_mae'], 'test_mae_norm': final_evidence['test_metrics']['normalized_mae'], 'checkpoint_sha256': final_evidence['best_checkpoint_sha256']}]))\n"),
    ]


def main() -> None:
    write(
        "00_environment_check.ipynb",
        [
            markdown(
                "# MEPI v1 — environment check\n\n"
                "**Purpose:** fail-closed readiness check before any training notebook.  \n"
                "**Inputs:** project source, frozen protocol, Python/PyTorch/CUDA environment, project tests.  \n"
                "**Outputs:** visible PASS/FAIL output and `reports/notebook_workflow_test_state.json`.  \n"
                f"**Frozen protocol:** {PROTOCOL}.  \n**Trains a model:** no.  \n**Checkpoints/results:** no model checkpoint; test state is saved under `reports/`."
            ),
            config_cell(),
            markdown("## Load readiness checks\n\nImport the shared checker from the repository source."),
            setup_cell("from src.mepi_v1.notebook_workflow import run_environment_checks"),
            markdown("## Run all assertions and tests\n\nEvery assertion is printed. A failed test or missing CUDA raises an informative exception and stops Run All."),
            code("environment_summary = run_environment_checks(PROJECT_ROOT, require_cuda=True, run_tests=True)\n"),
            markdown("## Final readiness summary\n\nDisplay the final state only when protocol assertions, project tests, and CUDA have passed."),
            code(
                "import pandas as pd\n"
                "display(pd.DataFrame([{'environment_ready': environment_summary['environment_ready'], 'cuda_ready': environment_summary['cuda_ready'], 'tests': environment_summary['tests']['status'], 'protocol_assertions': environment_summary['protocol_assertions']}]))\n"
                'print("ENVIRONMENT CHECK: PASS")\n'
            ),
        ],
    )
    write(
        "01_magnet_dataset_audit.ipynb",
        [
            markdown(
                "# MEPI v1 — MagNet dataset audit\n\n"
                "**Purpose:** audit the real MagNet HDF5 and recreate/verify deterministic 80/10/10 manifests.  \n"
                "**Inputs:** `configs/magnet.yaml` and the local HDF5.  \n"
                "**Outputs:** audit JSON/Markdown, three split manifests, dataset and split fingerprints.  \n"
                f"**Frozen protocol:** {PROTOCOL}.  \n**Trains a model:** no.  \n**Checkpoints/results:** no checkpoints; outputs are under `reports/`, `data/splits/mepi_v1/`, and manuscript-support artifacts."
            ),
            config_cell(),
            markdown("## Load audit workflow\n\nImport the shared dataset audit and fingerprint implementation."),
            setup_cell("from src.mepi_v1.notebook_workflow import run_dataset_audit_notebook"),
            markdown("## Audit all stages\n\nThe cell prints HDF5 shapes, counts, numeric ranges, material distribution, duplicate checks, split integrity, and fingerprints."),
            code("audit_result = run_dataset_audit_notebook(PROJECT_ROOT)\n"),
            markdown("## Final audit summary\n\nShow an archive-friendly summary. A non-PASS audit cannot reach this cell."),
            code(
                "import pandas as pd\n"
                "display(pd.DataFrame([{'status': audit_result['status'], 'total_samples': audit_result['total_samples'], 'dataset_fingerprint': audit_result['dataset_sha256'], 'split_fingerprint': audit_result['split_fingerprint']}]))\n"
                'print("MAGNET DATASET AUDIT: PASS")\n'
            ),
        ],
    )
    mapping = [
        ("10_pretrain_TCN.ipynb", "TCN"), ("11_pretrain_LSTM.ipynb", "LSTM"),
        ("12_pretrain_BiLSTM.ipynb", "BiLSTM"), ("13_pretrain_LSTM_Attention.ipynb", "LSTM-Attention"),
        ("14_pretrain_GRU.ipynb", "GRU"), ("15_pretrain_BiGRU.ipynb", "BiGRU"),
        ("16_pretrain_RWKV.ipynb", "RWKV"), ("17_pretrain_xLSTM.ipynb", "xLSTM"),
    ]
    for filename, backbone in mapping:
        write(filename, training_notebook(backbone))
    write(
        "20_compare_pretraining.ipynb",
        [
            markdown(
                "# MEPI v1 — compare completed pretraining runs\n\n"
                "**Purpose:** read measured COMPLETE evidence and select strictly by minimum validation MAE_norm.  \n"
                "**Inputs:** eight `experiments/pretrain_v1/<backbone>/run_evidence.json` files.  \n"
                "**Outputs:** comparison CSV/Markdown and the measured selection JSON.  \n"
                f"**Frozen protocol:** {PROTOCOL}.  \n**Trains a model:** no.  \n**Checkpoints/results:** reads checkpoints only by recorded hash; writes reports and `artifacts/manuscript/selected_pretraining_backbone.json`."
            ),
            config_cell(),
            markdown("## Load read-only comparison workflow\n\nThe comparison reads evidence only and never imports or invokes a training loop."),
            setup_cell("from src.mepi_v1.notebook_workflow import compare_completed_pretraining"),
            markdown("## Compare measured COMPLETE runs\n\nPending backbones remain visibly pending; completed rows sort by validation MAE_norm ascending."),
            code("comparison_rows = compare_completed_pretraining(PROJECT_ROOT)\n"),
            markdown("## Final comparison table\n\nDisplay validation and test metrics separately. The winner is never hard-coded."),
            code(
                "import pandas as pd\n"
                "comparison = pd.DataFrame(comparison_rows)\n"
                "display(comparison)\n"
                "completed = comparison[comparison['status'] == 'COMPLETE'] if not comparison.empty else comparison\n"
                "if completed.empty:\n"
                "    print('COMPARISON STATUS: PENDING — no measured COMPLETE runs')\n"
                "else:\n"
                "    print(f\"VALIDATION-SELECTED BACKBONE: {completed.iloc[0]['backbone']}\")\n"
                "print('Comparison artifacts written under reports/; selection JSON is measured-only.')\n"
            ),
        ],
    )
    print(f"Generated 11 notebooks in {NOTEBOOKS}")


if __name__ == "__main__":
    main()
