#!/usr/bin/env bash
set -u

TORCH_VERSION="2.14.0+cpu"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
MIN_ROOT_KIB=4194304
MIN_TMP_KIB=2097152
GUARD="apps/mepi_monitor/pi_runtime.py"

fail_with_code() {
  exit_code="$1"
  shift
  failure_class="$1"
  shift
  echo "ERROR [$failure_class] $*"
  if [ -n "${install_tmp:-}" ]; then
    echo "Diagnostics preserved at: $install_tmp"
  fi
  exit "$exit_code"
}

fail() {
  fail_with_code 1 "$@"
}

available_kib() {
  df -Pk "$1" 2>/dev/null | awk 'NR == 2 {print $4}'
}

classify_pip_failure() {
  log_path="$1"
  if grep -Eqi 'No space left on device' "$log_path"; then
    selected_now="$(available_kib "$install_tmp")"
    root_now="$(available_kib /)"
    if [ "${selected_now:-0}" -lt 262144 ]; then
      fail "INSUFFICIENT TEMPORARY STORAGE" "pip exhausted $install_tmp"
    fi
    fail "INSUFFICIENT ROOT STORAGE" "pip exhausted the root/user-site filesystem (root free KiB: ${root_now:-unknown})"
  fi
  if grep -Eqi 'No matching distribution|Could not find a version that satisfies' "$log_path"; then
    fail "NO COMPATIBLE CPU PYTORCH RUNTIME" "No official $TORCH_VERSION wheel matched this architecture/Python/ABI"
  fi
  if grep -Eqi 'Temporary failure in name resolution|Name or service not known|Connection (timed out|error)|ReadTimeout|SSLError|Network is unreachable|Max retries exceeded' "$log_path"; then
    fail "NETWORK/DOWNLOAD FAILURE" "See $log_path"
  fi
  fail "DEPENDENCY RESOLUTION FAILURE" "See $log_path"
}

configure_torch_openblas_preload() {
  case "$arch" in
    aarch64|arm64) ;;
    *) return 0 ;;
  esac
  openblas_path="$(python3 -c 'from apps.mepi_monitor.pi_launcher import find_torch_openblas; path=find_torch_openblas(); print(path or "")')"
  if [ -z "$openblas_path" ] || [ ! -f "$openblas_path" ]; then
    fail "PYTORCH IMPORT FAILURE" "required bundled torch/lib/libopenblas.so.0 is missing after installation"
  fi
  case " ${LD_PRELOAD:-} " in
    *" $openblas_path "*) ;;
    *) export LD_PRELOAD="$openblas_path ${LD_PRELOAD:-}" ;;
  esac
  echo "PyTorch bundled OpenBLAS preload: $openblas_path"
}

cd -- "$(dirname -- "$0")"
echo "MEPI Pi installer: frozen CPU inference only; no CUDA, NVIDIA, Triton, PySide6, PyVISA, or training packages."

arch="$(uname -m)"
python_version="$(python3 --version 2>&1 || true)"
if [ -r /etc/os-release ]; then
  os_name="$(. /etc/os-release; printf '%s' "${PRETTY_NAME:-unknown}")"
else
  os_name="unknown"
fi
ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
root_free_kib="$(available_kib /)"
tmp_free_kib="$(available_kib /tmp)"
tmp_mount="$(findmnt -no SOURCE,FSTYPE -T /tmp 2>/dev/null || echo 'unknown unknown')"

echo "Detected OS: $os_name"
echo "Detected architecture: $arch"
echo "Detected Python: ${python_version:-unavailable}"
echo "Detected RAM: ${ram_kib:-unknown} KiB"
echo "Root free space: ${root_free_kib:-unknown} KiB"
echo "System /tmp: $tmp_mount; free ${tmp_free_kib:-unknown} KiB"

case "$arch" in
  aarch64|arm64) ;;
  armv7l) fail_with_code 2 "UNSUPPORTED ARCHITECTURE" "ARMV7L DEPLOYMENT NOT SUPPORTED; install Raspberry Pi OS 64-bit and verify aarch64" ;;
  *) fail_with_code 2 "UNSUPPORTED ARCHITECTURE" "Expected aarch64/arm64, got $arch" ;;
esac

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)'; then
  fail "UNSUPPORTED PYTHON VERSION" "The pinned cp313 runtime requires Python 3.13.x; got ${python_version:-unavailable}"
fi
if [ ! -f "$GUARD" ]; then
  fail "DEPLOYMENT INTEGRITY FAILURE" "Missing $GUARD"
fi
if [ "${root_free_kib:-0}" -lt "$MIN_ROOT_KIB" ]; then
  fail "INSUFFICIENT ROOT STORAGE" "Need at least $MIN_ROOT_KIB KiB free; found ${root_free_kib:-unknown} KiB"
fi

install_tmp_parent="${HOME:?HOME is required}/.cache/mepi-install-tmp"
install_tmp="$install_tmp_parent/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p -m 700 "$install_tmp"
export TMPDIR="$install_tmp"
selected_mount="$(findmnt -no SOURCE,FSTYPE -T "$TMPDIR" 2>/dev/null || echo 'unknown unknown')"
selected_fstype="${selected_mount##* }"
selected_free_kib="$(available_kib "$TMPDIR")"
echo "Selected TMPDIR: $TMPDIR"
echo "Selected TMPDIR filesystem: $selected_mount; free ${selected_free_kib:-unknown} KiB"
case "$selected_fstype" in
  tmpfs|ramfs) fail "INSUFFICIENT TEMPORARY STORAGE" "Selected TMPDIR must be disk-backed, got $selected_fstype" ;;
esac
if [ "${selected_free_kib:-0}" -lt "$MIN_TMP_KIB" ]; then
  fail "INSUFFICIENT TEMPORARY STORAGE" "Need at least $MIN_TMP_KIB KiB at TMPDIR; found ${selected_free_kib:-unknown} KiB"
fi

echo "Selected PyTorch source: $TORCH_INDEX"
echo "Selected PyTorch version: $TORCH_VERSION"
echo "Selected runtime policy: official CPU-only wheel; exact cp313/aarch64 filename and SHA256 enforced"

if [ "${MEPI_INSTALL_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "MEPI installer preflight: PASS"
  rmdir "$install_tmp" 2>/dev/null || true
  exit 0
fi

if ! sudo apt-get update; then
  fail "OS PACKAGE INSTALLATION FAILURE" "apt-get update failed"
fi
if ! sudo apt-get install -y python3 python3-pip python3-tk python3-numpy python3-matplotlib python3-yaml libopenblas-dev; then
  fail "OS PACKAGE INSTALLATION FAILURE" "Required Raspberry Pi OS packages failed to install"
fi

installed_guard_log="$install_tmp/installed-guard.log"
if ! python3 "$GUARD" installed-guard >"$installed_guard_log" 2>&1; then
  cat "$installed_guard_log"
  fail "FORBIDDEN CUDA/NVIDIA DEPENDENCY" "Remove all nvidia-*, cuda-*, and triton packages before continuing"
fi

torch_present="$(python3 -c 'import importlib.util; print(1 if importlib.util.find_spec("torch") else 0)')"
if [ "$torch_present" = "1" ]; then
  echo "A PyTorch installation is present; validating it after OpenBLAS preload."
else
  plan_json="$install_tmp/pip-plan.json"
  plan_log="$install_tmp/pip-plan.log"
  echo "Resolving dependency plan without installing packages..."
  if ! python3 -m pip install --user --break-system-packages --dry-run --ignore-installed --only-binary=:all: --no-cache-dir --disable-pip-version-check --index-url "$TORCH_INDEX" --report "$plan_json" "torch==$TORCH_VERSION" >"$plan_log" 2>&1; then
    cat "$plan_log"
    classify_pip_failure "$plan_log"
  fi
  cat "$plan_log"
  plan_guard_log="$install_tmp/plan-guard.log"
  if ! python3 "$GUARD" plan "$plan_json" >"$plan_guard_log" 2>&1; then
    cat "$plan_guard_log"
    if grep -q 'FORBIDDEN CUDA/NVIDIA DEPENDENCY' "$plan_guard_log"; then
      fail "FORBIDDEN CUDA/NVIDIA DEPENDENCY" "The resolver plan was rejected before installation"
    fi
    fail "NO COMPATIBLE CPU PYTORCH RUNTIME" "The resolver plan did not match the pinned official CPU wheel"
  fi
  cat "$plan_guard_log"
  install_log="$install_tmp/pip-install.log"
  echo "Installing the validated CPU-only wheel into the user site."
  echo "PEP 668 note: --break-system-packages is used only for this user-site pip operation because no virtual environment is requested."
  if ! python3 -m pip install --user --break-system-packages --only-binary=:all: --no-cache-dir --disable-pip-version-check --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION" >"$install_log" 2>&1; then
    cat "$install_log"
    classify_pip_failure "$install_log"
  fi
  cat "$install_log"
fi

configure_torch_openblas_preload
if ! python3 -c 'import torch; print(torch.__version__)' >"$install_tmp/final-import.log" 2>&1; then
  cat "$install_tmp/final-import.log"
  fail "PYTORCH IMPORT FAILURE" "The installed CPU runtime cannot be imported"
fi
if ! python3 "$GUARD" installed-guard >"$install_tmp/final-package-guard.log" 2>&1; then
  cat "$install_tmp/final-package-guard.log"
  fail "FORBIDDEN CUDA/NVIDIA DEPENDENCY" "Forbidden package detected after installation"
fi
if ! python3 "$GUARD" runtime --require-target >"$install_tmp/final-runtime.log" 2>&1; then
  cat "$install_tmp/final-runtime.log"
  fail "PYTORCH FUNCTIONALITY FAILURE" "Required CPU operations failed"
fi
cat "$install_tmp/final-runtime.log"
if ! python3 run_mepi.py --headless-smoke >"$install_tmp/model-smoke.log" 2>&1; then
  cat "$install_tmp/model-smoke.log"
  fail "MODEL LOAD FAILURE" "Frozen checkpoint/model smoke failed"
fi
cat "$install_tmp/model-smoke.log"

case "$install_tmp" in
  "$install_tmp_parent"/run-*) rm -rf -- "$install_tmp" ;;
  *) fail "INTERNAL SAFETY FAILURE" "Refusing to clean unexpected TMPDIR path" ;;
esac
echo "Installation complete: exact CPU-only PyTorch and frozen MEPI model smoke PASS."
echo "Run ./test_pi.sh before launching python3 run_mepi.py."
