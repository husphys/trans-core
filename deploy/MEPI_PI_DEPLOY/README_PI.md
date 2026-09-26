# MEPI Monitor — Raspberry Pi deployment

This directory is self-contained. It runs the frozen MEPI v1.5 depth-8 xLSTM on CPU and does not require the research repository.

## Required architecture

This package requires a **64-bit operating system** reporting `aarch64` or `arm64` from `uname -m`. It is not a validated ARM32 package. On `armv7l`, `install_pi.sh` and `test_pi.sh` fail closed before inference or package installation.

The frozen model requires PyTorch Conv1d, real FFT, adaptive pooling, multi-head attention, LayerNorm, eight LSTM blocks, and strict checkpoint loading. Official PyTorch and Debian packages do not provide a suitable `armhf/armv7l` runtime, and no scientifically equivalent alternative runtime has been validated on ARM32. Do not install an unofficial wheel or substitute an approximate model.

If `uname -m` reports `armv7l` but the board is a Raspberry Pi 3, 4, 5, 400, or Zero 2, back up required files and use Raspberry Pi Imager to install **Raspberry Pi OS (64-bit)**. Recheck that `uname -m` reports `aarch64` before continuing. Original Raspberry Pi, Raspberry Pi 2 with its original 32-bit CPU, and original Pi Zero hardware require a newer 64-bit-capable board for this application. See `docs/MEPI_ARMV7_DEPLOYMENT_AUDIT.md` in the research repository for the compatibility evidence.

## Install and run

1. Verify `uname -m` reports `aarch64` or `arm64`, then copy the complete `MEPI_PI_DEPLOY` directory to the Raspberry Pi.
2. Open a terminal and enter the directory: `cd MEPI_PI_DEPLOY`
3. Enable the scripts: `chmod +x install_pi.sh test_pi.sh`
4. Install runtime dependencies: `./install_pi.sh`
5. Validate the package: `./test_pi.sh`
6. Launch the GUI: `python3 run_mepi.py`

No virtual environment is used. `install_pi.sh` prefers Raspberry Pi OS packages, installs Tkinter through `python3-tk`, and uses a user-site, PEP 668-aware CPU PyTorch installation only when torch is absent. It does not install PySide6, PyVISA, CUDA, Jupyter, or training tools.

## Keysight LAN setup

- Connect transformer primary voltage to Keysight CH1.
- Connect transformer secondary voltage to Keysight CH2.
- Connect the Keysight and Raspberry Pi by Ethernet/LAN.
- The default raw SCPI TCP port is `5025`.
- Determine the scope IP from its LAN/I/O menu, configure the Pi Ethernet interface in the same subnet, and verify `ping SCOPE_IP`.
- Connectivity-only CLI check: `python3 run_mepi.py --test-scope 192.168.1.149`
- In the GUI, enter the actual IP/port, click **Test Scope Connection**, enter ambient temperature manually, then click **Start Monitoring**.

The example IP is configurable and is not required. The connectivity test sends only `*IDN?`, displays the identity, and closes the socket; it does not run inference.

## Replay before hardware

Choose **Simulation / Replay** and click **Start Monitoring**. Replay uses real synchronized scope records and the same preprocessing, 1024-point B(t), model, and domain-guard path as LAN acquisition. CLI checks are `python3 run_mepi.py --headless-smoke` and `python3 run_mepi.py --replay-smoke`.

## Troubleshooting

- No network/ping: check cable, link LEDs, IP addresses, netmask, and that Pi/scope are in the same subnet.
- Port 5025 closed: enable raw socket/SCPI LAN service in the scope I/O settings and verify firewall/routing.
- `*IDN?` timeout: confirm the entered IP and port; power-cycle only after saving any scope work.
- Tk display error: run from the Pi desktop session with `DISPLAY` set; SSH requires X forwarding or a local display.
- Torch import error: use 64-bit Raspberry Pi OS and install the CPU wheel/package for the exact Python version; rerun `./test_pi.sh`.
- Missing/checkpoint integrity error: recopy the entire directory without altering files. The application fails closed on hash mismatch.
- Out-of-domain warning: do not clamp or change inputs to suppress it. Treat results as outside/near the investigated domain.

## Scientific limitations

Efficiency and core loss are predictions, not direct two-channel power measurements. LSP is a dimensionless Arrhenius-informed relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or measured degradation. Predicted LSP sigma is not calibrated uncertainty. Domain membership does not prove generalization; custom-core predictions are exploratory; frequency screening is descriptive and is not experimentally validated optimization. Hard real-time performance is not claimed.
