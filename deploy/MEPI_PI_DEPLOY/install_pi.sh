#!/usr/bin/env bash
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
