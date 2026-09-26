# MEPI Tkinter Raspberry Pi deployment audit

## Outcome

Status: **PASS — SELF-CONTAINED DEPLOYMENT VERIFIED**

Source baseline: `e7bc42b2c261e86cd4df0eaba220d3992bd4f4ac` (`main`). The existing PySide6 frontend remains present and its scientific backend was not modified. Tkinter is a view/control layer over the existing acquisition, replay, preprocessing, inference, profile, domain-guard, and offline-screening modules.

No training, fine-tuning, model selection, checkpoint selection, target change, scaler change, split change, preprocessing change, feature change, model architecture change, domain-data change, or silent input clamping/defaulting occurred.

## Protected scientific evidence

The nine protected hashes remained unchanged, including:

- selected checkpoint: `0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33`
- depth-8 transfer checkpoint: `fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c`
- final manifest: `607bad9d532777f11a6659459cfa59e6a78e054bbd6ed95abe0e4f3dfffe67c1`
- train-only normalization: `ea9858df0b5367a8cf8c7a73cf36073a6e255564d35b290e61fabf3d56be6ab2`

The deployment manifest checks all runtime-critical checkpoint/config/scaler/domain/report/demo/replay artifacts before model startup and fails with `CHECKPOINT INTEGRITY ERROR` on mismatch.

## Validation

- Tkinter replay rendering on the available display: **PASS**.
- Background worker lifecycle, queue delivery, duplicate-worker prevention, and clean stop: **PASS**.
- New and existing complete repository suite: **84 passed**.
- External standalone `test_pi.sh`: **12/12 PASS**.
- Standalone location: a clean `/tmp/MEPI_PI_DEPLOY_TEST.*` directory containing only `MEPI_PI_DEPLOY`; repository root absent from `PYTHONPATH`; working directory outside the repository.
- Headless smoke: strict checkpoint load, frozen train scaler, domain JSON, real replay, 1024-point B waveform, finite predictions, `training_performed=false`: **PASS**.
- Replay smoke: **PASS**.
- Physical Keysight I/O: **NOT PERFORMED**; connectivity-only `*IDN?` support is implemented.

## Scientific equivalence

Repository backend and standalone deployment were compared on the same FE real-record replay using `rtol=1e-7`, `atol=1e-9` for frequency, Vin RMS, phase, all nine predictive features, every B(t) point, efficiency, core loss, LSP, and LSP sigma; domain status was compared exactly.

Result: **PASS**. Maximum B(t) absolute difference and every scalar absolute difference were `0.0`; domain status matched (`YELLOW — NEAR DOMAIN BOUNDARY`).

## Package

- Directory: `deploy/MEPI_PI_DEPLOY/`
- ZIP: `deploy/MEPI_PI_DEPLOY.zip`
- ZIP SHA256: `5574c91b4be15a81dcea43b4e75cac51ef18696c226c0ac03dc32ea1b31b07dc`
- ZIP layout: 58 files, CRC check passed, and every entry is beneath top-level `MEPI_PI_DEPLOY/`.
- Uncompressed payload: 95,400,644 bytes; ZIP size: 87,272,019 bytes.

The deployment contains no repository-location dependency in executable code, shell scripts, or runtime configuration. Two absolute development paths retained inside the byte-frozen Steinmetz JSON and raw/demo provenance columns are descriptive provenance strings only; they are never resolved or accessed at runtime and cannot be removed without violating protected scientific hashes.

## Scientific interpretation boundaries

Efficiency and core loss are MEPI predictions, not direct two-channel power measurements. LSP is a dimensionless Arrhenius-informed relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or measured degradation. Predicted LSP sigma is not calibrated uncertainty. Domain membership does not prove generalization. Custom-core predictions are exploratory. Frequency screening is descriptive and is not experimentally validated optimization. Hard real-time performance is not claimed.
