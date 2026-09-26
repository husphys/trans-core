#!/usr/bin/env bash
set -u
echo "MEPI Pi installer: CPU inference only; no PySide6, PyVISA, CUDA, or training packages."
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
