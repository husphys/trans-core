# MEPI / Transformer Core Study

This repository is the public reproducibility package for the final MEPI v1.5 transformer-core study. It contains the frozen scientific protocols, executable preprocessing and modeling code, deterministic datasets and splits, authoritative notebooks, selected checkpoints, and persisted evidence used for manuscript tables and figures.

Earlier protocol versions v1.0–v1.4 are retained under `docs/` to preserve change provenance. **MEPI-FROZEN-PROTOCOL v1.5 is the current authoritative downstream experiment contract.**

## Scientific pipeline

```text
Instrument acquisitions / MagNet source
                │
                ▼
QC, B(t) reconstruction, group-wise splitting, train-only normalization
                │
                ▼
MagNet representation pretraining → xLSTM depth study → depth-8 checkpoint
                │                                      │
                └──────────────── transfer ─────────────┘
                                                       ▼
         9 tabular inputs + B(t)_1024 → multimodal MEPI xLSTM
                                                       │
                           ┌───────────────────────────┼──────────────────┐
                           ▼                           ▼                  ▼
                    efficiency (%)                 P_loss             LSP proxy
                                                       │
                     validation-only loss-weight selection
                                                       │
                     frozen final test, exactly once
                                                       │
                offline measured-domain frequency screening
```

The nine downstream tabular inputs are:

1. `frequency_hz`
2. `vin_rms_v`
3. `phase_shift_deg`
4. `temperature_ambient_c`
5. `B_peak_t`
6. `B_rms`
7. `B_thd_percent`
8. `dBdt_max`
9. `form_factor`

The waveform input is the reconstructed and frozen `B(t)_1024` sequence. The three outputs are efficiency, power loss (`P_loss`), and LSP.

**LSP is a physics-guided relative lifetime/thermal-stress proxy. It is not directly measured remaining useful life or service lifetime.**

## Frozen v1.5 model

- Backbone: xLSTM, depth 8, latent width 256
- Transfer: strict loading of `waveform_encoder` and `backbone`
- Final configuration: `l3_1_l4_0p05`
- Loss weights: `lambda1=1.0`, `lambda2=1.0`, `lambda3=1.0`, `lambda4=0.05`
- Selection: minimum validation-only `VAL_TASK_SCORE = validation_L_electrical + validation_L_LSP`
- Final-test evaluations: exactly 1

```text
final checkpoint SHA256 = 0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33
depth-8 representation checkpoint SHA256 = fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c
final-model manifest SHA256 = 607bad9d532777f11a6659459cfa59e6a78e054bbd6ed95abe0e4f3dfffe67c1
```

Both checkpoints are stored through Git LFS. Run `git lfs pull` after cloning.

## Dataset and isolation policy

The primary MEPI design contains 900 measured rows in 90 nominal operating-condition groups. The frozen QC-valid dataset has 862 rows:

| Split | Groups | Rows | Use |
|---|---:|---:|---|
| Train | 72 | 687 | Optimization and train-only statistics |
| Validation | 9 | 85 | Stopping and model/loss-weight selection |
| Test | 9 | 90 | Exactly-once final evaluation only |

All repetitions and both physical core types for an operating-condition group remain together. The test split was not used for training, early stopping, architecture selection, or loss-weight selection. See [DATA.md](DATA.md) for included derived data, portable demo waveforms, raw-data exclusions, layouts, sizes, and hashes.

## Final persisted evidence

The exactly-once final-test ledger is under `reports/MEPI_V1_5_FINAL_TEST_RESULTS.*`; it records `TEST_EVALUATION_COUNT = 1`.

| Target | MAE | RMSE | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| efficiency | 1.1456656562 | 1.4833748448 | 0.9817636937 | — |
| P_loss | 0.0136925422 | 0.0164594169 | 0.9706166407 | — |
| LSP_raw | 0.0279506859 | 0.0363657696 | 0.9162509237 | 4.7880477905 |

The manuscript evidence aggregation is available in `reports/MEPI_V1_5_MANUSCRIPT_EVIDENCE_SUMMARY.md` and `.json`.

## Frequency-screening scope

The persisted demonstration uses 150 measured demo rows, retains 146 usable rows after excluding four existing `HARD_FAIL` rows, and summarizes 15 predefined frequencies. It is an **offline proof-of-concept inside the measured operating domain**, not an independently validated universal optimal-frequency recommendation. No optimum is declared.

## Repository structure

| Path | Purpose |
|---|---|
| `docs/` | Frozen protocols v1.0–v1.5 and final selection documents |
| `configs/` | Pretraining, depth-study, fine-tuning, sensitivity, and final frozen configs |
| `src/` | Dataset, QC, preprocessing, model, training, evaluation, screening, and hashing code |
| `notebooks/` | Authoritative pretraining/depth and v1.5 workflow notebooks |
| `tests/` | Scientific-contract, integrity, test-isolation, and reproducibility tests |
| `data/` | Frozen derived MEPI data, deterministic splits, and portable demo waveforms |
| `artifacts/` | Train-only normalization/physics and manuscript metadata artifacts |
| `experiments/` | Compact run metadata, final manifest, and selected LFS checkpoints |
| `reports/` | Final scientific evidence and provenance reports |
| `archive_manifest/` | Hash inventory for external raw acquisition files |

## Installation

Conda is recommended:

```bash
conda env create -f environment.yml
conda activate transformer-repro
git lfs pull
python -m pytest -q
```

Alternatively, install the concise pinned Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Python 3.11, PyTorch 2.5.1 with CUDA 12.1, NumPy 2.1.3, pandas 2.2.3, SciPy 1.14.1, and scikit-learn 1.5.2 are recorded in the provided environment specification. CUDA is required for the original GPU experiment runtime; artifact inspection, hash verification, and many integrity tests are CPU-safe.

## Reproducibility workflow

First verify the frozen data and checkpoints:

```bash
(cd data/MEPI/v1_1 && sha256sum -c checksums_v4.sha256)
(cd data/MEPI/v1_2 && sha256sum -c checksums_v1_2.sha256)
(cd data/MEPI/v1_3 && sha256sum -c checksums_v1_3.sha256)
sha256sum experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt
sha256sum experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt
```

Authoritative notebook order:

1. `00_environment_check.ipynb`
2. `01_magnet_dataset_audit.ipynb`
3. `10_pretrain_TCN.ipynb` through `17_pretrain_xLSTM.ipynb`
4. `20_compare_pretraining.ipynb`
5. `21_xlstm_depth_experiment.ipynb`
6. `22_xlstm_transfer_audit.ipynb`
7. `30_finetune_dataset_audit.ipynb`
8. `32_xlstm_v1_5_finetune.ipynb`
9. `33_xlstm_v1_5_loss_weight_sensitivity.ipynb`
10. `34_xlstm_v1_5_final_training_and_test.ipynb`
11. `35_xlstm_v1_5_frequency_screening.ipynb`
12. `36_xlstm_v1_5_tables_and_figures.ipynb`

Notebooks 34 and 35 contain protected scientific execution stages. The published outputs and reports are the authoritative completed evidence; do not rerun them merely to regenerate display output. Notebook 36 is aggregation-only and reconstructs tables from persisted artifacts without model inference or test-data access.

## Citation and license

Citation metadata are provided in [CITATION.cff](CITATION.cff). Publication DOI and journal metadata are intentionally omitted until established. The code and repository-authored documentation are released under the [MIT License](LICENSE); third-party datasets remain subject to their original terms.
