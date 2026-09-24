# MEPI-FROZEN-PROTOCOL v1.1 migration report

Status date: 2026-09-22 UTC  
Change classification: **CONSISTENCY FIX**  
Training or physical-instrument execution: **NOT PERFORMED**

## A. Protocol migration

- Active specification: `MEPI-FROZEN-PROTOCOL v1.1.md`.
- Authoritative copies: root and `docs/MEPI-FROZEN-PROTOCOL v1.1.md`.
- Both copies verified byte-identical with SHA-256
  `a515052cd2f8cf2970b731cfb48dee055c316a5d4acc0f0ca90313436c344c0f`.
- v1.0 remains only as superseded protocol/evidence; active guards reject it.

## B. Files changed

- `src/mepi_v1/acquisition.py`: added unit-safe shunt/current, synchronized Scope phase,
  input-power, protocol-version, and fail-fast validation functions.
- `src/mepi_v1/constants.py`: centralized v1.1 version/hash/paths, shunt constants, new raw
  provenance fields, and expanded leakage denylist.
- `src/mepi_v1/downstream_audit.py`: bound the dataset builder to explicit v1.1 Scope #1 and
  Scope #2 records, derived quantities, raw hashes, and legacy rejection.
- `src/mepi_v1/waveform.py`: changed the B(t) source contract from secondary voltage/Ns to
  Scope #2 CH1 primary voltage/Np.
- `src/mepi_v1/finetune.py`, `configs/finetune_v1.yaml`: changed required turns metadata to
  `primary_turns`; kept unresolved scientific values unset.
- `src/mepi_v1/notebook_workflow.py`: verifies both authoritative v1.1 copies against the
  expected hash before workflow execution.
- `src/data/audit.py`: explicitly labeled the old audit as legacy/pre-v1.1.
- `scripts/generate_mepi_notebooks.py`, `scripts/generate_xlstm_transition_notebooks.py`:
  updated generated notebook protocol identity, paths, and SHA-256.
- `notebooks/00_environment_check.ipynb`, `notebooks/01_magnet_dataset_audit.ipynb`,
  `notebooks/10_pretrain_TCN.ipynb`, `notebooks/11_pretrain_LSTM.ipynb`,
  `notebooks/12_pretrain_BiLSTM.ipynb`, `notebooks/13_pretrain_LSTM_Attention.ipynb`,
  `notebooks/14_pretrain_GRU.ipynb`, `notebooks/15_pretrain_BiGRU.ipynb`,
  `notebooks/16_pretrain_RWKV.ipynb`, `notebooks/17_pretrain_xLSTM.ipynb`,
  `notebooks/20_compare_pretraining.ipynb`, and `notebooks/21_xlstm_depth_experiment.ipynb`:
  updated active source cells to v1.1. Existing v1.0 outputs were retained unchanged and
  explicitly marked historical rather than relabeled.
- `tests/test_protocol_v1_1_acquisition.py`: added protocol hash, shunt, current, phase
  separation, power, feature/leakage, legacy rejection, and end-to-end preprocessing tests.
- `tests/test_mepi_v1_contract.py`, `tests/test_downstream_preparation_contract.py`,
  `tests/test_notebook_recovery.py`: updated primary-turn and v1.1 fixtures.
- `README.md`, `BLOCKERS.md`, `reports/implementation_status.md`, and
  `reports/protocol_compatibility_audit.md`: updated active documentation while retaining
  clearly labeled historical findings.

## C. Acquisition implementation

- Keithley 2000 COM3 -> AC `vshunt_rms_v` across both series shunts.
- Scope #1 CH2 -> voltage before the shunt pair.
- Scope #1 CH1 -> voltage after the pair / transformer primary terminal.
- Scope #1 MATH -> CH2 - CH1; this is the current-phase reference.
- Scope #2 CH1 -> primary Vin and the raw voltage waveform used for B(t).
- Scope #2 CH2 -> Vout, synchronous with Scope #2 CH1.
- The implementation contains no instrument I/O; tests use synthetic scope records only.

## D. Derived quantities

- `iin_rms_a = vshunt_rms_v / 1.5152`.
- `input_vi_phase_deg = phase(Scope1 CH1) - phase(Scope1 CH2 - CH1)`.
- `input_power_w = scope1_vin_rms_v * iin_rms_a * cos(input_vi_phase_deg)`.
- `phase_shift_deg = phase(Scope2 CH1 Vin) - phase(Scope2 CH2 Vout)`.
- `B(t) = integral(Scope2 CH1 primary voltage dt) / (Np * Ae)`, followed by the existing
  offset/drift correction, complete-cycle extraction, and 1024-point resampling.

## E. Frozen ML contract

- Fine-tuning tabular input remains exactly nine ordered features.
- Waveform input remains `B(t)_1024`.
- Targets remain efficiency, `P_loss`, and LSP.
- Operating-condition-group split remains 80/10/10.
- Currents, shunt voltage, power/current phase, power quantities, targets, and core
  temperature remain excluded from predictive features.
- MagNet inputs, candidate backbones, transfer logic, PIRL/MTPH, and model architecture
  were not redesigned.

## F. Validation

- Unit tests: **65 passed** with bytecode and pytest cache disabled.
- Protocol guard: **PASS** for both v1.1 copies and the expected SHA-256.
- Notebook static validation: **27 valid JSON notebooks**, **180/180 code cells parsed**.
- Ruff on the new acquisition code, migrated downstream audit, v1.1 test, and updated xLSTM
  generator: **PASS**.
- Fine-tuning readiness guard: **WAITING_FOR_REAL_FINETUNE_DATA**, as expected; no fake
  values were inserted.
- Existing CUDA smoke script: **NOT RUN TO COMPLETION** because CUDA was unavailable in
  this runner. It stopped before loading/training a model and issued no instrument command.

## G. Remaining legacy references

- `MEPI-FROZEN-PROTOCOL v1.0.md`: retained superseded source evidence.
- `experiments/**` v1.0 protocol strings: retained measured historical run evidence.
- Saved outputs/metadata in eight pretraining notebooks and the xLSTM depth notebook:
  retained historical execution evidence; active source cells are v1.1.
- `reports/protocol_compatibility_audit.md`, cleanup reports, and
  `environment/trans-core-system.txt`: retained dated v1.0 audit/environment evidence.
- The v1.1 protocol's own “supersedes v1.0” text and the negative legacy-rejection test are
  intentional documentation/test references.
- `0.2508` remains only in a negative test proving that the old shunt value is rejected.
- Other exact `0.25` occurrences are an unrelated benchmark polling interval and a metric
  fixture; neither is a shunt resistance.
- Old 12/22-feature and `held-out` strings remain only in superseded protocol/audit or
  correction context, not in the active predictive schema or split implementation.

## H. Blockers

- No code-migration blocker remains.
- Real-data end-to-end acquisition validation remains blocked because the v1.1 dataset path,
  `Np`, `Ae`, winding resistances, and Arrhenius constants are not present/configured.
- CUDA-only smoke validation remains environment-blocked in this runner; unit/static
  protocol validation passed independently.
