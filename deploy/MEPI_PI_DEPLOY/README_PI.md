# MEPI Monitor — Raspberry Pi 3 Model B deployment

This self-contained directory runs the frozen MEPI v1.5 depth-8 xLSTM on CPU. The research repository and a Python virtual environment are not required.

## Validated target contract

- Raspberry Pi 3 Model B
- Raspberry Pi OS 64-bit
- `uname -m`: `aarch64`
- CPython 3.13.5
- CPU inference only; CUDA is neither required nor permitted

The pinned runtime is `torch==2.14.0+cpu` from the official PyTorch CPU index, `https://download.pytorch.org/whl/cpu`. The installer accepts only the CPython 3.13/aarch64 CPU wheel named `torch-2.14.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl` with SHA-256 `092d5c12938850dfbd90a654b3c8dac34c33e300f88eb19ee6f4ef93992c6347`. It rejects any resolver plan or installed package matching `nvidia-*`, `cuda-*`, or `triton`.

## Clean installation

```bash
cd MEPI_PI_DEPLOY
chmod +x install_pi.sh test_pi.sh
./install_pi.sh
./test_pi.sh
python3 run_mepi.py
```

No virtual environment is created. Raspberry Pi OS packages provide Tkinter, NumPy, Matplotlib, PyYAML, and OpenBLAS. Because Raspberry Pi OS marks system Python as externally managed under PEP 668, the installer uses `--user --break-system-packages` narrowly and only for the pinned PyTorch CPU-wheel plan/install. It does not replace unrelated apt-managed packages. On `aarch64`, `run_mepi.py` locates the wheel-bundled `torch/lib/libopenblas.so.0` dynamically and re-executes Python with that library preloaded before importing torch; no home directory is hard-coded and no manual `LD_PRELOAD` command is needed.

## Temporary and disk storage

The Pi's `/tmp` may be a roughly 453 MB tmpfs. The installer never uses it for the large wheel operation. It creates a run-specific, disk-backed directory below `$HOME/.cache/mepi-install-tmp`, prints the backing filesystem and free capacity, requires at least 2 GiB there and 4 GiB on the root filesystem, and exports that path as `TMPDIR`. It removes the run directory after success and preserves logs there after failure.

The official aarch64 CPU wheel is approximately 159 MB. This differs from the approximately 454 MB PyPI wheel that previously resolved NVIDIA/CUDA/Triton dependencies and is intentionally not used.

## What the checks cover

`test_pi.sh` reports architecture, OS, Python, disk/TMPDIR, Tkinter, NumPy, Matplotlib, exact CPU PyTorch version, forbidden-package guard, CPU tensor operations, FFT/RFFT, attention, eight LSTM blocks, CUDA absence, checkpoint/hash, scalers, domain data, 1024-point preprocessing, strict model load, headless/replay smoke, and a bounded CPU benchmark.

Expected target lines include:

```text
INFERENCE DEVICE: CPU
CUDA AVAILABLE: FALSE (EXPECTED)
```

## Keysight physical status

The verified measurement topology is Pi `eth0=192.168.2.129/24` directly connected to a Keysight EDUX1052A at `192.168.2.149/24`; the Pi's `wlan0` remains on the separate `192.168.1.x` network. The physical instrument is serial `CN63260332`, firmware `02.12.2021071625`. An actual Telnet session on TCP 5024 verified `*IDN?`, CH1 frequency (approximately 2129 Hz), and CH1 VRMS (approximately 3.86 V). Raw Python sockets to 5024/5025 accepted connections but timed out because they did not implement the Telnet session.

The application uses a Telnet-aware SCPI transport on port 5024. Based on the physical packet trace, it sends an initial CRLF wakeup, allows a bounded 30-second initialization window, accepts only the observed server WILL options 1 and 3, rejects unsupported options, and then restores the ordinary 6-second SCPI query timeout. Banner/prompt handling, repeated queries, reconnect, and length-framed waveform responses remain protected. Test identity only with `python3 run_mepi.py --test-scope 192.168.2.149`; the port defaults to 5024 and remains configurable with `--port`. Physical CH1/CH2 waveform acquisition and physical live MEPI inference are **not yet verified** and must not be claimed until the next scope test.

## Replay and GUI

Run `python3 run_mepi.py --headless-smoke` and `python3 run_mepi.py --replay-smoke` before launching the Tkinter GUI. In the GUI select **Simulation / Replay** first. Live Monitor, Offline Screening, Domain Guard, transformer profiles, and the background worker remain unchanged.

## Failure classes

The installer distinguishes unsupported architecture/Python, insufficient temporary/root storage, missing compatible CPU runtime, network/download errors, dependency resolution errors, forbidden CUDA/NVIDIA dependencies, PyTorch import/functionality errors, and model-load errors. Do not bypass a fail-closed error or install an arbitrary wheel.

## Scientific limitations

Efficiency and core loss are predictions, not direct two-channel power measurements. LSP is a dimensionless Arrhenius-informed relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or measured degradation. Predicted LSP sigma is not calibrated uncertainty. Domain membership does not prove generalization; custom-core predictions are exploratory; frequency screening is descriptive and is not experimentally validated optimization. Hard real-time performance is not claimed.
