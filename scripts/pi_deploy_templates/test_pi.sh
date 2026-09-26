#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")"

fail=0
pass() { echo "PASS  $1"; }
fail_check() { echo "FAIL  $1"; fail=1; }
check() { label="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$label"; else fail_check "$label"; fi; }

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

target_mode=0
case "$arch" in
  aarch64|arm64) target_mode=1; pass "ARCHITECTURE (aarch64 target)" ;;
  x86_64|AMD64) pass "ARCHITECTURE (development-host mode; physical Pi still pending)" ;;
  *) fail_check "ARCHITECTURE (expected aarch64 target)" ;;
esac
check "OS" test -r /etc/os-release
if [ "$target_mode" -eq 1 ]; then
  check "PYTHON VERSION (3.13.x target)" python3 -c 'import sys; assert sys.version_info[:2] == (3, 13)'
else
  check "PYTHON VERSION (development host)" python3 -c 'import sys; assert sys.version_info >= (3, 9)'
fi

root_free_kib="$(df -Pk / 2>/dev/null | awk 'NR == 2 {print $4}')"
if [ "${root_free_kib:-0}" -ge 1048576 ]; then pass "DISK SPACE"; else fail_check "DISK SPACE"; fi
test_tmp_parent="${HOME:?HOME is required}/.cache/mepi-test-tmp"
test_tmp="$test_tmp_parent/run-$$"
mkdir -p -m 700 "$test_tmp"
test_tmp_fstype="$(findmnt -no FSTYPE -T "$test_tmp" 2>/dev/null || echo unknown)"
case "$test_tmp_fstype" in tmpfs|ramfs|unknown) fail_check "TMPDIR (disk-backed)" ;; *) pass "TMPDIR (disk-backed: $test_tmp_fstype)" ;; esac

if [ "$target_mode" -eq 1 ]; then
  openblas_path="$(python3 -c 'from apps.mepi_monitor.pi_launcher import find_torch_openblas; path=find_torch_openblas(); print(path or "")')"
  if [ -z "$openblas_path" ] || [ ! -f "$openblas_path" ]; then
    fail_check "PYTORCH BUNDLED OPENBLAS PRELOAD"
    echo "Required torch/lib/libopenblas.so.0 is missing; rerun ./install_pi.sh."
    exit 1
  fi
  case " ${LD_PRELOAD:-} " in
    *" $openblas_path "*) ;;
    *) export LD_PRELOAD="$openblas_path ${LD_PRELOAD:-}" ;;
  esac
  pass "PYTORCH BUNDLED OPENBLAS PRELOAD ($openblas_path)"
else
  pass "PYTORCH BUNDLED OPENBLAS PRELOAD (aarch64 target only)"
fi

check "TKINTER" python3 -c 'import tkinter'
check "NUMPY" python3 -c 'import numpy'
check "MATPLOTLIB" python3 -c 'from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg'
check "PYTORCH" python3 -c 'import torch'
if [ "$target_mode" -eq 1 ]; then
  check "PYTORCH VERSION (2.14.0+cpu)" python3 -c 'import torch; assert torch.__version__ == "2.14.0+cpu"'
  check "FORBIDDEN CUDA/NVIDIA/TRITON PACKAGE GUARD" python3 -m apps.mepi_monitor.pi_runtime installed-guard
  check "CUDA AVAILABILITY (FALSE EXPECTED)" python3 -c 'import torch; assert torch.version.cuda is None and not torch.cuda.is_available()'
else
  pass "PYTORCH VERSION (development-host runtime recorded below)"
  pass "FORBIDDEN CUDA/NVIDIA/TRITON PACKAGE GUARD (target-only enforcement; development host may contain CUDA tooling)"
  pass "CUDA AVAILABILITY (target expectation deferred to physical Pi)"
fi
check "PYTORCH CPU TENSOR OPERATION" python3 -c 'import torch; x=torch.ones(4,device="cpu"); assert x.sum().item()==4 and x.device.type=="cpu"'
check "PYTORCH FFT/RFFT" python3 -c 'import torch; y=torch.fft.rfft(torch.arange(32,dtype=torch.float32)); assert y.shape==(17,) and torch.isfinite(y).all()'
if [ "$target_mode" -eq 1 ]; then
  check "PYTORCH REQUIRED MODEL OPERATIONS" python3 -m apps.mepi_monitor.pi_runtime runtime --require-target
else
  check "PYTORCH REQUIRED MODEL OPERATIONS" python3 -m apps.mepi_monitor.pi_runtime runtime
fi
echo "INFERENCE DEVICE: CPU"
if [ "$target_mode" -eq 1 ]; then echo "CUDA AVAILABLE: FALSE (EXPECTED)"; fi

checkpoint="experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt"
check "CHECKPOINT" test -f "$checkpoint"
check "CHECKPOINT HASH" python3 -c 'import hashlib,pathlib; p=pathlib.Path("experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/best_checkpoint.pt"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33"'
check "SCALERS" test -f data/MEPI/v1_2/train_normalization_v1_2.json
check "DOMAIN DATA" test -f reports/MEPI_V1_5_PREDICTION_DOMAIN.json
check "DEPLOYMENT MANIFEST INTEGRITY" python3 -c 'from pathlib import Path; from apps.mepi_monitor.deployment import verify_deployment; assert verify_deployment(Path.cwd())["status"] == "PASS"'

test_log="$test_tmp/headless.json"
if python3 run_mepi.py --headless-smoke >"$test_log" 2>&1; then
  pass "MODEL LOAD"
  if python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); assert p["waveform_length"]==1024; assert p["strict_load"] and p["scaler_loaded"] and not p["training_performed"]; assert all(math.isfinite(float(p[k])) for k in ("predicted_efficiency_percent","predicted_core_loss_w","predicted_lsp","predicted_lsp_sigma"))' "$test_log"; then pass "1024-POINT PIPELINE"; else fail_check "1024-POINT PIPELINE"; fi
  pass "HEADLESS SMOKE"
else
  cat "$test_log"
  fail_check "MODEL LOAD"
  fail_check "1024-POINT PIPELINE"
  fail_check "HEADLESS SMOKE"
fi
check "REPLAY SMOKE" python3 run_mepi.py --replay-smoke

benchmark_log="$test_tmp/benchmark.json"
if python3 -m apps.mepi_monitor.pi_runtime benchmark --root . >"$benchmark_log" 2>&1; then
  pass "CPU BENCHMARK"
  cat "$benchmark_log"
else
  cat "$benchmark_log"
  fail_check "CPU BENCHMARK"
fi

if [ "$fail" -ne 0 ]; then
  echo "MEPI_PI_DEPLOY validation: FAIL"
  echo "Diagnostics retained at: $test_tmp"
  exit 1
fi
case "$test_tmp" in "$test_tmp_parent"/run-*) rm -rf -- "$test_tmp" ;; esac
echo "MEPI_PI_DEPLOY build/development validation: PASS"
if [ "$target_mode" -eq 1 ]; then
  echo "Physical Raspberry Pi runtime checks: PASS on this host invocation"
else
  echo "PHYSICAL PI TEST: PENDING"
fi
