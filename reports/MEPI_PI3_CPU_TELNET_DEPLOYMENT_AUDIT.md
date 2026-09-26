# MEPI Pi 3B CPU/Telnet deployment audit

Date: 2026-09-26
Source baseline: `6e15e566961eaeecac0d9f4b19491041a6a51c8a` (`main`)
Outcome: **BUILD VALIDATION PASS; UPDATED PHYSICAL BUILD TEST PENDING**

## Scope

This was a deployment/runtime compatibility change only. No training, fine-tuning, model selection, checkpoint change, quantization, architecture change, target/scaler change, preprocessing or B(t) change, domain-guard change, dataset/split change, post-hoc result change, or manuscript change was performed.

The target is a Raspberry Pi 3B running 64-bit Raspberry Pi OS (`aarch64`) and CPython 3.13.5. The standalone runtime is pinned to `torch==2.14.0+cpu` from `https://download.pytorch.org/whl/cpu`, wheel `torch-2.14.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl`, SHA-256 `092d5c12938850dfbd90a654b3c8dac34c33e300f88eb19ee6f4ef93992c6347`. Resolver-plan and installed-package guards reject `nvidia-*`, `cuda-*`, and `triton`.

## Compatibility fixes

- The installer uses a disk-backed run directory under `$HOME/.cache/mepi-install-tmp` instead of the approximately 453 MB `/tmp` tmpfs, validates root/TMPDIR capacity, distinguishes failure classes, and preserves failure logs.
- PEP 668 override is limited to the exact PyTorch user-site dry-run/install operations; no virtual environment is created.
- `run_mepi.py`, `install_pi.sh`, and `test_pi.sh` dynamically locate the wheel-bundled `torch/lib/libopenblas.so.0` on `aarch64` and preload it before importing torch. No username or home path is hard-coded.
- Keysight transport now uses Telnet-aware SCPI on TCP 5024: option refusal, banner/prompt handling, CRLF command termination, exact textual response cleanup, repeated queries, explicit timeouts, close/reconnect, and length-driven IEEE waveform-block handling.
- `KeysightLanScope`, `ScopeCapture`, CH1-primary/CH2-secondary acquisition commands, synchronized time-base check, and all downstream scientific interfaces remain unchanged.
- Tkinter defaults are user-editable `192.168.2.149:5024`; controller and CLI defaults are 5024.

## Files changed

Source/runtime:

- `apps/mepi_monitor/acquisition/keysight_lan.py`
- `apps/mepi_monitor/pi_launcher.py`
- `apps/mepi_monitor/pi_runtime.py`
- `apps/mepi_monitor/tk_main.py`
- `apps/mepi_monitor/ui_tk/controller.py`
- `apps/mepi_monitor/ui_tk/main_window.py`
- `scripts/build_mepi_pi_deploy.py`
- `scripts/pi_deploy_templates/README_PI.md`
- `scripts/pi_deploy_templates/install_pi.sh`
- `scripts/pi_deploy_templates/requirements-pi.txt`
- `scripts/pi_deploy_templates/test_pi.sh`
- `docs/MEPI_DEMO_APPLICATION.md`

Tests and audit:

- `apps/mepi_monitor/tests/test_keysight_telnet.py`
- `apps/mepi_monitor/tests/test_pi_launcher.py`
- `apps/mepi_monitor/tests/test_tk_deployment.py`
- `reports/MEPI_PI3_CPU_TELNET_DEPLOYMENT_AUDIT.md`

Generated deliverables:

- changed runtime/template mirrors under `deploy/MEPI_PI_DEPLOY/`
- `deploy/MEPI_PI_DEPLOY/deploy_manifest.json`
- `deploy/MEPI_PI_DEPLOY.zip`

## Validation

- Repository suite: **100 passed**.
- Telnet transport mock: **4 passed**, covering split Telnet negotiation, welcome/banner/prompt, command echo, identity, numeric responses, repeated queries, non-query prompts, timeout, close/reconnect, IEEE length framing, CH1/CH2 waveform flow, and unchanged `ScopeCapture` output.
- Launcher tests: **4 passed**, covering dynamic user-site discovery, existing preload preservation, no re-exec loop, non-target no-op, and clear missing-library failure.
- Deployment/launcher/Telnet regression group: **24 passed**.
- Standalone clean-copy `test_pi.sh`: **PASS** on the development host, including manifest, strict model load, 1024-point path, replay, required operations, and benchmark.
- Exact target-runtime operations under isolated `torch 2.14.0+cpu`: **PASS**; `torch.version.cuda=None`, CUDA unavailable, Conv1d, RFFT, adaptive pooling, LayerNorm, GELU, softplus, attention, eight LSTM blocks, checkpoint strict load, and CPU inference passed.
- Two real replay records, repository PyTorch 2.5.1 versus target CPU PyTorch 2.14.0+cpu: **PASS** at `rtol=1e-7`, `atol=1e-9`; 2,092 numeric values compared; maximum absolute difference `1.3997410519550613e-07`; maximum relative difference `8.315874451550452e-08`; domain status matched exactly.
- Protected scientific artifacts: **14/14 unchanged**, including both v1.5 protocol copies, final/fine-tune configs, train scaler, split manifest, both checkpoints, final manifest/results/screening, demo manifest, model definitions, and preprocessing pipeline.
- ZIP integrity: **PASS**, 60 files, CRC clean, one `MEPI_PI_DEPLOY/` top-level, zero cache artifacts, and zero developer paths in executable/runtime configuration.
- ZIP SHA-256: `55fd6c80b24d123e452b1fd39703752a7c79b68db20529b7497b326224fb86af`.

## User-supplied physical evidence

- Raspberry Pi 3B, 64-bit Raspberry Pi OS, `aarch64`, Python 3.13.5.
- With the bundled OpenBLAS manually preloaded, the previous standalone package passed Tkinter, CPU torch, checkpoint/scaler/domain checks, headless smoke, replay smoke, and GUI launch/render on the physical Pi.
- Measurement network: Pi `eth0=192.168.2.129/24`; Keysight EDUX1052A `192.168.2.149/24`; Internet/VNC remains on separate `wlan0=192.168.1.x`.
- Physical Telnet port 5024 returned identity `KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625`, CH1 frequency approximately 2129 Hz, and CH1 VRMS approximately 3.86 V.

## Remaining physical validation

The rebuilt package still requires a physical-Pi rerun to confirm automatic (non-manual) OpenBLAS preload and the new Python Telnet client. Physical CH1/CH2 waveform acquisition and physical live MEPI inference are **not yet verified** and are not claimed. The next bounded test is identity via `python3 run_mepi.py --test-scope 192.168.2.149`, followed by one observed live two-channel capture with no scientific changes.
