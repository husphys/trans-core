"""Build the self-contained CPU-only MEPI Raspberry Pi deployment package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEPLOYMENT_VERSION = "1.0.1"

RUNTIME_PATHS = (
    "apps/__init__.py", "apps/mepi_monitor/__init__.py", "apps/mepi_monitor/acquisition",
    "apps/mepi_monitor/preprocessing", "apps/mepi_monitor/inference", "apps/mepi_monitor/domain_guard",
    "apps/mepi_monitor/profiles", "apps/mepi_monitor/offline", "apps/mepi_monitor/live",
    "apps/mepi_monitor/assets/replay", "apps/mepi_monitor/ui_tk",
    "apps/mepi_monitor/deployment.py", "apps/mepi_monitor/tk_main.py",
    "src/__init__.py", "src/mepi_v1/__init__.py", "src/mepi_v1/backbones.py",
    "src/mepi_v1/build_finetune_dataset.py", "src/mepi_v1/checkpointing.py", "src/mepi_v1/config.py",
    "src/mepi_v1/constants.py", "src/mepi_v1/final_model_manifest_v1_5.py",
    "src/mepi_v1/finetune_notebook.py", "src/mepi_v1/finetune_v1_4.py",
    "src/mepi_v1/finetune_v1_5.py", "src/mepi_v1/models.py", "src/mepi_v1/waveform.py",
    "configs/final_model_v1_5.yaml", "configs/finetune_v1_5.yaml",
    "MEPI-FROZEN-PROTOCOL v1.5.md", "MEPI-V1.5-LOSS-WEIGHT-SELECTION-RULE.md",
    "artifacts/finetune_v1_4/steinmetz_prior_train_v1_4.json",
    "data/MEPI/v1_2/train_normalization_v1_2.json", "data/MEPI/demo_manifest_v2.csv",
    "experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt",
    "experiments/finetune_v1_5_xlstm_depth8/final_model_manifest.json",
    "experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt",
    "reports/MEPI_V1_5_FINAL_TEST_RESULTS.json", "reports/MEPI_V1_5_FREQUENCY_SCREENING_PREDICTIONS.csv",
    "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json", "reports/MEPI_V1_5_PREDICTION_DOMAIN.json",
)

INTEGRITY_PATHS = (
    "experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt",
    "experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt",
    "experiments/finetune_v1_5_xlstm_depth8/final_model_manifest.json",
    "configs/final_model_v1_5.yaml", "configs/finetune_v1_5.yaml",
    "data/MEPI/v1_2/train_normalization_v1_2.json",
    "artifacts/finetune_v1_4/steinmetz_prior_train_v1_4.json",
    "reports/MEPI_V1_5_PREDICTION_DOMAIN.json", "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json",
    "data/MEPI/demo_manifest_v2.csv", "apps/mepi_monitor/assets/replay/manifest.json",
)

RUNNER = '''#!/usr/bin/env python3
"""Standalone MEPI Monitor entry point."""
from apps.mepi_monitor.tk_main import main
if __name__ == "__main__":
    main()
'''

REQUIREMENTS = '''# Minimal runtime only; Tkinter is installed with apt as python3-tk.
numpy>=1.26,<3
torch>=2.5,<3
PyYAML>=6,<7
matplotlib>=3.8,<4
'''

INSTALL = r'''#!/usr/bin/env bash
set -u
echo "MEPI Pi installer: CPU inference only; no PySide6, PyVISA, CUDA, or training packages."
arch="$(uname -m)"
python_version="$(python3 --version 2>&1 || true)"
if [ -r /etc/os-release ]; then
  os_name="$(. /etc/os-release; printf '%s' "${PRETTY_NAME:-unknown}")"
else
  os_name="unknown"
fi
echo "Detected architecture: $arch"
echo "Detected OS: $os_name"
echo "Detected Python: ${python_version:-unavailable}"
case "$arch" in
  aarch64|arm64) ;;
  armv7l)
    echo "ERROR: ARMV7L DEPLOYMENT NOT SUPPORTED"
    echo "The frozen MEPI v1.5 runtime requires PyTorch operations for which no validated armv7l runtime is packaged."
    echo "No official PyTorch or ONNX Runtime Python wheel is available for this 32-bit target."
    echo "If the Raspberry Pi CPU supports 64-bit operation, install Raspberry Pi OS (64-bit), verify 'uname -m' reports aarch64, and use this package unchanged."
    exit 2
    ;;
  *)
    echo "ERROR: unsupported deployment architecture: $arch (expected aarch64/arm64)."
    exit 2
    ;;
esac
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-tk python3-numpy python3-matplotlib python3-yaml libopenblas-dev
python3 - <<'PY' >/dev/null 2>&1 && have_torch=1 || have_torch=0
import torch
PY
if [ "$have_torch" -eq 0 ]; then
  arch="$(uname -m)"
  case "$arch" in aarch64|arm64) ;; *) echo "ERROR: automatic torch install is supported only on 64-bit ARM (got $arch)."; exit 1;; esac
  echo "PyTorch is missing; installing a CPU wheel into the user site (PEP 668-aware)."
  if ! python3 -m pip install --user --break-system-packages 'torch>=2.5,<3'; then
    echo "ERROR: no compatible aarch64 PyTorch wheel was found for $(python3 --version)."
    echo "Install the Raspberry Pi OS/PyTorch CPU package recommended for your exact OS and Python, then rerun ./test_pi.sh."
    exit 1
  fi
fi
echo "Installation complete. Run ./test_pi.sh before launching the GUI."
'''

TEST = r'''#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")"
arch="$(uname -m)"
python_version="$(python3 --version 2>&1 || true)"
if [ -r /etc/os-release ]; then
  os_name="$(. /etc/os-release; printf '%s' "${PRETTY_NAME:-unknown}")"
else
  os_name="unknown"
fi
echo "Detected architecture: $arch"
echo "Detected OS: $os_name"
echo "Detected Python: ${python_version:-unavailable}"
if [ "$arch" = "armv7l" ]; then
  echo "FAIL  64-bit architecture"
  echo "ARMV7L DEPLOYMENT NOT SUPPORTED: migrate supported hardware to Raspberry Pi OS (64-bit)."
  exit 2
fi
fail=0
check() { label="$1"; shift; if "$@" >/dev/null 2>&1; then echo "PASS  $label"; else echo "FAIL  $label"; fail=1; fi; }
check "Python version" python3 -c 'import sys; assert sys.version_info >= (3,9)'
check "64-bit architecture" python3 -c 'import platform; assert platform.machine() in {"aarch64","arm64","x86_64","AMD64"}'
check "tkinter import" python3 -c 'import tkinter'
check "torch import / CPU" python3 -c 'import torch; torch.zeros(1, device="cpu")'
check "numpy import" python3 -c 'import numpy'
check "matplotlib Tk backend import" python3 -c 'from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg'
check "selected checkpoint exists" test -f experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt
check "scaler and config exist" test -f data/MEPI/v1_2/train_normalization_v1_2.json
check "prediction domain exists" test -f reports/MEPI_V1_5_PREDICTION_DOMAIN.json
check "replay fixture exists" test -f apps/mepi_monitor/assets/replay/FE_F1000_V3p9_R01_scope2.csv
check "headless smoke" python3 run_mepi.py --headless-smoke
check "replay smoke" python3 run_mepi.py --replay-smoke
if [ "$fail" -ne 0 ]; then echo "MEPI_PI_DEPLOY validation: FAIL"; exit 1; fi
echo "MEPI_PI_DEPLOY validation: PASS"
'''

README = '''# MEPI Monitor — Raspberry Pi deployment

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
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def copy_runtime(root: Path, target: Path) -> None:
    for relative in RUNTIME_PATHS:
        source, destination = root / relative, target / relative
        if not source.exists(): raise FileNotFoundError(f"Required runtime artifact is missing: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir(): shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        else: shutil.copy2(source, destination)


def build(root: Path, output_parent: Path) -> tuple[Path, Path, str]:
    root, output_parent = root.resolve(), output_parent.resolve()
    target, archive = output_parent / "MEPI_PI_DEPLOY", output_parent / "MEPI_PI_DEPLOY.zip"
    if target.exists(): shutil.rmtree(target)
    if archive.exists(): archive.unlink()
    target.mkdir(parents=True); copy_runtime(root, target)
    (target / "run_mepi.py").write_text(RUNNER, encoding="utf-8")
    (target / "requirements-pi.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (target / "install_pi.sh").write_text(INSTALL, encoding="utf-8")
    (target / "test_pi.sh").write_text(TEST, encoding="utf-8")
    (target / "README_PI.md").write_text(README, encoding="utf-8")
    for name in ("run_mepi.py", "install_pi.sh", "test_pi.sh"):
        (target / name).chmod((target / name).stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    try: source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError): source_commit = "UNKNOWN"
    artifacts = {relative: sha256(target / relative) for relative in INTEGRITY_PATHS}
    selected = "experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt"
    payload = {
        "deployment_version": DEPLOYMENT_VERSION, "source_git_commit": source_commit,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(), "scientific_model_version": "MEPI v1.5",
        "architecture": "8-layer xLSTM", "loss_weights": {"lambda1":1.0,"lambda2":1.0,"lambda3":1.0,"lambda4":0.05},
        "checkpoint_filename": selected, "checkpoint_sha256": artifacts[selected],
        "normalization_sha256": artifacts["data/MEPI/v1_2/train_normalization_v1_2.json"],
        "source_config_sha256": artifacts["configs/finetune_v1_5.yaml"],
        "prediction_domain_sha256": artifacts["reports/MEPI_V1_5_PREDICTION_DOMAIN.json"],
        "training_performed": False, "model_selection_performed": False, "runtime_artifacts": artifacts,
    }
    (target / "deploy_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    forbidden = (str(root), "/home/diy-hus/transfomfer")
    dependency_suffixes = {".py", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
    for path in target.rglob("*"):
        if path.is_file() and path.suffix.lower() in dependency_suffixes:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(value in text for value in forbidden): raise RuntimeError(f"Development path leaked into {path}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(target.rglob("*")):
            if path.is_file(): handle.write(path, Path(target.name) / path.relative_to(target))
    return target, archive, sha256(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-parent", type=Path, default=Path("deploy"))
    args = parser.parse_args(); target, archive, digest = build(args.root, args.output_parent)
    print(json.dumps({"deploy_directory":str(target),"zip":str(archive),"zip_sha256":digest}, indent=2))


if __name__ == "__main__": main()
