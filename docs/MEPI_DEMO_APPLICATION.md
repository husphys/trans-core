# MEPI v1.5 demonstration application

## Scope and scientific status

`apps/mepi_monitor` is an inference-only research prototype for the frozen MEPI v1.5 selected model. It does not train, fine-tune, select, refit, calibrate, or write checkpoints. The selected checkpoint is:

```text
experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt
SHA256 = 0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33
architecture = xLSTM, depth 8, latent width 256
lambda1 = 1.0
lambda2 = 1.0
lambda3 = 1.0
lambda4 = 0.05
```

The GUI separates values into **MEASURED**, **PHYSICALLY DERIVED**, and **MEPI PREDICTED** panels. LSP is a dimensionless Arrhenius-informed relative thermal-stress proxy. It is not lifetime, RUL, time-to-failure, or experimentally measured degradation. Predicted LSP sigma is an uncalibrated model output and is not a confidence interval.

## Architecture

```text
Keysight LAN or real replay capture
  -> synchronized CH1 Vin + CH2 Vout
  -> frozen cycle/phase/Faraday preprocessing in src/mepi_v1
  -> B(t), exactly 1024 points, and nine frozen features
  -> train-fitted scalers + strict frozen checkpoint
  -> predicted efficiency, core loss, LSP, LSP sigma
  -> warning-only feature-level domain guard
  -> PySide6 research-instrument UI
```

Scientific functions are reused rather than duplicated:

- `src/mepi_v1/waveform.py`: offset/drift correction, trapezoidal integration, complete-cycle extraction, 1024-point resampling;
- `src/mepi_v1/build_finetune_dataset.py`: synchronized cycle pairing, Vin/Vout phase, RMS, and magnetic feature definitions;
- `src/mepi_v1/finetune_v1_5.py` and the final-model manifest: architecture and strict loading;
- `src/mepi_v1/finetune_v1_4.py`: selected-checkpoint waveform normalization and Steinmetz latent reference.

Steinmetz-informed latent learning and the Arrhenius-based LSP target remain conceptually and programmatically separate.

## Hardware and Raspberry Pi requirements

- Validated deployment target: Raspberry Pi 3B, 64-bit Raspberry Pi OS, `aarch64`, Python 3.13.5.
- Keysight EDUX1052A Telnet/SCPI endpoint on TCP port 5024.
- Scope CH1: transformer primary voltage.
- Scope CH2: transformer secondary voltage.
- CH1 and CH2 must be captured in one synchronized acquisition record.
- Ethernet between the Pi and scope; configure both endpoints on the same subnet.
- No Keithley meter, second oscilloscope, or physical temperature sensor is required for this prototype.
- Ambient temperature is mandatory manual input and is visibly labeled `Manual input`; the application starts this field blank and never substitutes 25 °C or another default. Replay fills the value from recorded evidence, while live inference rejects a missing value.

The deployment uses a minimal Telnet-aware socket transport rather than PyVISA. It refuses Telnet options, consumes only whole banner/prompt records, preserves length-framed waveform payload bytes, and retains the established ASCII waveform commands. Live mode measures frequency using repeated Keysight CH1 `:MEASure:FREQuency?` queries and a median/stability check. The GUI defaults to `192.168.2.149:5024`, while both fields remain user-editable.

PySide6 was retained as requested. It is not installed in the current `trans-core` environment, and Raspberry Pi wheel availability depends on the exact 64-bit OS/Python combination. Install it only in the demo environment. If the PyPI wheel is unavailable, use the distribution's Qt-for-Python package; the GUI-independent engine and `--headless-smoke` do not import PySide6. No silent framework switch was made.

## Transformer profiles

The two investigated profiles share the frozen prototype geometry and electrical setup:

| Profile | OD | ID | Height | Np | Ns | Working Ae | Load |
|---|---:|---:|---:|---:|---:|---:|---:|
| FE | 34.58 mm | 20.81 mm | 17.68 mm | 10 | 10 | 1.217268e-4 m² | 49.6025 ohm |
| Commercial | 34.58 mm | 20.81 mm | 17.68 mm | 10 | 10 | 1.217268e-4 m² | 49.6025 ohm |

The values are corroborated by the retained acquisition source and the frozen B-reprocessing evidence. The geometric option uses the same rectangular toroid cross-section that reproduces the documented working area:

```text
Ae = height * (OD - ID) / 2
```

Lengths are converted from millimetres to metres squared. Manual Ae entry is also supported. Every custom profile is visibly marked `UNSEEN CORE — EXPLORATORY PREDICTION`, even if its dimensions resemble a built-in profile.

## Frozen input contract

The application assembles exactly `B(t)_1024` plus, in order:

1. `frequency_hz`
2. `vin_rms_v`
3. `phase_shift_deg`
4. `temperature_ambient_c`
5. `B_peak_t`
6. `B_rms`
7. `B_thd_percent`
8. `dBdt_max`
9. `form_factor`

Core identity is metadata, not a predictive input. Inputs are never clamped.

From CH1/CH2 alone, the prototype legitimately labels frequency, Vin RMS, and Vout RMS as measured. Vin-to-Vout phase and magnetic features are physically derived. Efficiency, core loss, LSP, and LSP sigma are model predictions. Directly measured input current, input power, efficiency, and loss require additional instruments and are intentionally not shown as measured in this two-channel prototype.

## Domain guard

`reports/MEPI_V1_5_PREDICTION_DOMAIN.json` is derived only from the 687 frozen training rows and all nine predictive columns. Each variable records observed training minimum/maximum, units, source column, and the empirical 5th–95th percentile central range.

- Green: every feature is within its training 5th–95th percentile central range.
- Yellow: every feature is within observed training min–max, but at least one lies outside the central range.
- Red: at least one feature lies outside observed training min–max.

This is an auditable interpretation warning, not a validity proof. `UNSEEN CORE` is distinct from a known FE/Commercial core operated outside the investigated feature ranges. The guard never clamps, substitutes, or changes input values.

Regenerate the artifact deterministically:

```bash
/home/diy-hus/miniconda3/envs/trans-core/bin/python -c "from apps.mepi_monitor.domain_guard.guard import build_prediction_domain; build_prediction_domain('.', 'reports/MEPI_V1_5_PREDICTION_DOMAIN.json')"
```

## Live and replay modes

Live mode connects directly to the Keysight by IP, requests a synchronized two-channel capture, runs the frozen pipeline, and refreshes at a user-selected interval. Physical Telnet identity/frequency/VRMS are verified, but physical CH1/CH2 waveform acquisition and live MEPI inference remain pending. Hard real-time behavior is not claimed.

Simulation / Replay uses two real 3.9-V demonstration captures under `apps/mepi_monitor/assets/replay`. Their SHA256 values and source sample IDs are recorded in `assets/replay/manifest.json`. They remain application-demonstration evidence and are not assigned a train/validation/test split. Replay and hardware captures call the same `run_capture` path.

## Offline screening

The Repository Demonstration view reads the existing 146 usable FE/Commercial predictions from the independent 150-row 3.9-V demo manifest (four hard failures remain excluded by the pre-existing screening contract). It groups measured repeats transparently and shows sample count, mean, and sample standard deviation for each measured candidate frequency from 1.0 to 4.5 kHz.

The interface uses “screened candidate” terminology. It never calls a minimum or maximum an optimal frequency. The visible claim boundary is:

> Frequency screening is descriptive and restricted to the investigated measurement domain. It does not constitute experimentally validated frequency optimization.

Compatible CSV input is supported by `offline.load_compatible_csv`. Each row must contain the nine frozen features plus `b_waveform_file`, a relative path to a 1024-value `.npy` file or CSV whose final column is B. This adapter validates but does not silently repair inputs.

## Model Performance panel

The read-only panel appears only after Notebook 37 reproducibly generates `reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json`. It shows final-test N and MAE/RMSE/R²/Err95 for each target, plus LSP MAPE. Err95 is explicitly a dataset-level error percentile, not a confidence interval and not a live per-sample measure.

The repository evaluation helper defines the requested denominator epsilon as `1e-12` in `src/evaluation/metrics.py`. Notebook 37 records that exact provenance rather than introducing a new value.

## Install and run

Reuse the verified scientific interpreter. Install only the missing demo packages in a separate/controlled environment if needed:

```bash
/home/diy-hus/miniconda3/envs/trans-core/bin/python -m pip install -r requirements-demo.txt
/home/diy-hus/miniconda3/envs/trans-core/bin/python -m apps.mepi_monitor.main
```

On Raspberry Pi, install a platform-supported CPU PyTorch build first if the generic requirement cannot resolve it. Do not replace the frozen model or scalers.

Hardware-free validation:

```bash
/home/diy-hus/miniconda3/envs/trans-core/bin/python -m apps.mepi_monitor.main --headless-smoke
```

Post-hoc notebook execution from the repository root:

```bash
/home/diy-hus/miniconda3/envs/trans-core/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/37_mepi_v1_5_posthoc_test_analysis.ipynb
```

Optional screenshot, after PySide6 is installed:

```bash
QT_QPA_PLATFORM=offscreen /home/diy-hus/miniconda3/envs/trans-core/bin/python \
  -m apps.mepi_monitor.main --screenshot reports/figures/MEPI_monitor_live.png
```

## Expected files

- Post-hoc predictions, JSON/Markdown metrics, and measured-vs-predicted PNG/PDF.
- Residual diagnostic PNG (not automatically placed in the manuscript).
- Training-derived prediction-domain JSON.
- Replay manifest and two real raw Scope #2 captures.
- No modified checkpoint, protocol, final configuration, split, scaler, final-test ledger, manuscript, or model-development dataset.

## Limitations

- Two voltage channels cannot directly measure input current, input power, measured efficiency, or measured transformer loss.
- Ambient temperature is manual unless a future source implements the same explicit interface; core temperature is not used for live inference.
- Domain membership does not establish generalization or experimental validation.
- Custom-core predictions are exploratory.
- Demo repeat variability is descriptive measurement/prediction variability, not uncertainty calibration.
- PySide6 GUI rendering and physical Keysight connectivity must be verified on the target Raspberry Pi; the current environment supports only GUI-independent/headless validation until PySide6 is installed.
- The application does not change the manuscript automatically.
