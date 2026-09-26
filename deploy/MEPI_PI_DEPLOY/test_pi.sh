#!/usr/bin/env bash
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
