# MEPI v1.5 post-hoc and demonstration application audit

## Outcome

Status: **IMPLEMENTED AND HEADLESS-VALIDATED**

The frozen selected checkpoint, protocol, final configuration, source model configuration, manifest, exactly-once final-test ledger, and demo manifest retain their pre-work SHA256 values. No training, fine-tuning, model selection, hyperparameter change, checkpoint change, split change, or manuscript edit occurred.

## Post-hoc final-test metrics

| Target | MAE | RMSE | R² | MAPE | Err95 |
|---|---:|---:|---:|---:|---:|
| Efficiency | 1.1456658998 pp | 1.4833747585 pp | 0.9817636969 | — | 4.4384757072% |
| Core loss | 0.0136925429 W | 0.0164594178 W | 0.9706166387 | — | 25.2739479617% |
| LSP | 0.0279506840 | 0.0363657660 | 0.9162509357 | 4.7880475755% | 13.0185276990% |

Err95 uses `epsilon = 1e-12` from the repository evaluation helper `src/evaluation/metrics.py`. It is not interpreted as a confidence interval. Predicted LSP sigma min/median/mean/max is `0.0252114789 / 0.0331328339 / 0.0351182685 / 0.0550108950`; no calibration claim is made.

Post-hoc MAE/RMSE/R² values agree with the existing frozen final-test report within `2.44e-7` or better. Notebook 37 executed from top to bottom with execution counts `[1, 2]` and no training/optimizer call.

## Demo provenance and live requirements

- `data/MEPI/demo_manifest_v2.csv`: 150 application-only 3.9-V rows, 75 FE and 75 Commercial, 15 measured frequency setpoints from 1.0 to 4.5 kHz, five repeats per core/frequency before QC exclusions.
- Existing screening output: 146 usable rows; four pre-existing hard failures excluded.
- Demo rows have `dataset_mode=demo`, no split column, and zero sample-ID overlap with the 862 model-development/final-test rows.
- Two real synchronized Keysight Scope #2 records were copied into the replay fixture with their original SHA256 and sample provenance.
- Live hardware: Raspberry Pi, direct Ethernet/LAN to a Keysight scope on TCP 5025; CH1 primary Vin and CH2 secondary Vout. Ambient temperature is required manual input. No Keithley, second scope, or core-temperature sensor is required for this prototype.

## Validation

- New application/post-hoc tests: **11 passed**.
- Headless real-record replay: **PASS**, exact 1024-point path, strict checkpoint load, finite predictions, domain warning emitted.
- Notebook 37: **PASS**, executed end to end on CUDA.
- Protected-artifact hash audit: **PASS (9/9 unchanged)**.
- Full local workspace suite with deterministic cuBLAS setting: **176 passed, 4 failed**. Clean `origin/main` publication tree with the intended overlay: **76 passed**. The four failures are pre-existing/stale assertions outside this task: one expects an older Notebook 32 kernelspec; two require literal Notebook 35 status strings although it prints the same values dynamically; one expects `RUN_GRID=False` in Notebook 33 although the repository currently contains `RUN_GRID=True`. These existing scientific notebooks were not changed merely to satisfy stale tests.

## Files created

- `apps/__init__.py`
- `apps/mepi_monitor/` application package, tests, and two replay captures
- `src/mepi_v1/posthoc_analysis_v1_5.py`
- `notebooks/37_mepi_v1_5_posthoc_test_analysis.ipynb`
- `reports/MEPI_V1_5_POSTHOC_TEST_PREDICTIONS.csv`
- `reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json`
- `reports/MEPI_V1_5_POSTHOC_TEST_METRICS.md`
- `reports/MEPI_V1_5_PREDICTION_DOMAIN.json`
- `reports/figures/MEPI_final_test_measured_vs_predicted.png`
- `reports/figures/MEPI_final_test_measured_vs_predicted.pdf`
- `reports/figures/MEPI_final_test_residual_diagnostics.png`
- `docs/MEPI_DEMO_APPLICATION.md`
- `requirements-demo.txt`
- this audit report

Files modified from published `origin/main`: **README.md only** for navigation and conservative LSP wording. Because the source workspace had no `.git` directory, publication was prepared in a clean clone of `origin/main`; only the explicit artifact allowlist and README edit were staged. The nine protected artifacts were independently compared with both the local hash manifest and remote baseline.

## Remaining limitations

- PySide6 is not installed in either checked local Python, so a GUI screenshot and target-Pi GUI launch were not possible. The PySide6 UI is implemented, compiles statically, and is documented; GUI-independent inference/replay passed.
- Physical Keysight LAN connectivity was not exercised because no hardware I/O was performed during validation.
- Two-channel voltage acquisition cannot directly measure input current, input power, efficiency, or transformer loss; those remain MEPI predictions.
- Domain status is a warning based on frozen training-feature coverage, not proof of validity.
- Custom profiles are explicitly marked `UNSEEN CORE — EXPLORATORY PREDICTION`.
- Frequency screening remains descriptive; no experimentally validated optimum is claimed.
