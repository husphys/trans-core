"""Raspberry Pi CPU-runtime guards, signatures, and bounded benchmarks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

CPU_TORCH_VERSION = "2.14.0+cpu"
CPU_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
CPU_TORCH_WHEEL = "torch-2.14.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl"
CPU_TORCH_WHEEL_SHA256 = "092d5c12938850dfbd90a654b3c8dac34c33e300f88eb19ee6f4ef93992c6347"
TARGET_PYTHON = (3, 13)
FORBIDDEN_PACKAGE_PREFIXES = ("nvidia-", "cuda-")
FORBIDDEN_PACKAGE_NAMES = frozenset({"triton"})


class PiRuntimeError(RuntimeError):
    """A fail-closed Raspberry Pi runtime validation error."""


def canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def forbidden_packages(names: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    normalized = sorted({canonical_package_name(name) for name in names})
    return [
        name
        for name in normalized
        if name in FORBIDDEN_PACKAGE_NAMES
        or any(name.startswith(prefix) for prefix in FORBIDDEN_PACKAGE_PREFIXES)
    ]


def installed_package_names() -> list[str]:
    names: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            names.append(name)
    return sorted(set(names))


def validate_installed_package_guard() -> dict[str, object]:
    blocked = forbidden_packages(installed_package_names())
    if blocked:
        raise PiRuntimeError(
            "FORBIDDEN CUDA/NVIDIA DEPENDENCY: " + ", ".join(blocked)
        )
    return {"status": "PASS", "forbidden_packages": []}


def _report_hash(download_info: dict[str, object]) -> str:
    archive = download_info.get("archive_info", {})
    if not isinstance(archive, dict):
        return ""
    hashes = archive.get("hashes", {})
    if isinstance(hashes, dict) and hashes.get("sha256"):
        return str(hashes["sha256"])
    value = str(archive.get("hash", ""))
    return value.removeprefix("sha256=")


def validate_pip_report(report_path: str | Path) -> dict[str, object]:
    """Reject non-CPU or forbidden dependency plans before installation."""

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    planned = payload.get("install")
    if not isinstance(planned, list) or not planned:
        raise PiRuntimeError("DEPENDENCY RESOLUTION FAILURE: empty pip plan")
    names: list[str] = []
    torch_items: list[dict[str, object]] = []
    for item in planned:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise PiRuntimeError("DEPENDENCY RESOLUTION FAILURE: malformed pip plan")
        metadata = item["metadata"]
        name = str(metadata.get("name", ""))
        names.append(name)
        if canonical_package_name(name) == "torch":
            torch_items.append(item)
    blocked = forbidden_packages(names)
    if blocked:
        raise PiRuntimeError(
            "FORBIDDEN CUDA/NVIDIA DEPENDENCY: " + ", ".join(blocked)
        )
    if len(torch_items) != 1:
        raise PiRuntimeError("NO COMPATIBLE CPU PYTORCH RUNTIME: plan must contain one torch wheel")
    torch_item = torch_items[0]
    metadata = torch_item["metadata"]
    if str(metadata.get("version")) != CPU_TORCH_VERSION:
        raise PiRuntimeError(
            f"NO COMPATIBLE CPU PYTORCH RUNTIME: expected {CPU_TORCH_VERSION}, "
            f"planned {metadata.get('version')}"
        )
    download_info = torch_item.get("download_info", {})
    if not isinstance(download_info, dict):
        raise PiRuntimeError("NO COMPATIBLE CPU PYTORCH RUNTIME: missing wheel provenance")
    url = str(download_info.get("url", ""))
    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    allowed_hosts = {"download.pytorch.org", "download-r2.pytorch.org"}
    if parsed.hostname not in allowed_hosts or "/whl/cpu/" not in parsed.path:
        raise PiRuntimeError(f"NO COMPATIBLE CPU PYTORCH RUNTIME: non-CPU source {url}")
    if filename != CPU_TORCH_WHEEL:
        raise PiRuntimeError(
            f"NO COMPATIBLE CPU PYTORCH RUNTIME: expected {CPU_TORCH_WHEEL}, got {filename}"
        )
    digest = _report_hash(download_info)
    if digest != CPU_TORCH_WHEEL_SHA256:
        raise PiRuntimeError(
            "NO COMPATIBLE CPU PYTORCH RUNTIME: official wheel SHA256 mismatch"
        )
    requirements = metadata.get("requires_dist", [])
    dependency_names = []
    if isinstance(requirements, list):
        dependency_names = [re.split(r"[ (;]", str(value), maxsplit=1)[0] for value in requirements]
    blocked_metadata = forbidden_packages(dependency_names)
    if blocked_metadata:
        raise PiRuntimeError(
            "FORBIDDEN CUDA/NVIDIA DEPENDENCY in torch metadata: "
            + ", ".join(blocked_metadata)
        )
    return {
        "status": "PASS",
        "torch_version": CPU_TORCH_VERSION,
        "wheel": CPU_TORCH_WHEEL,
        "wheel_sha256": digest,
        "source": CPU_TORCH_INDEX_URL,
        "planned_packages": sorted(canonical_package_name(name) for name in names),
        "forbidden_packages": [],
    }


def required_torch_operations(*, require_target_runtime: bool) -> dict[str, object]:
    """Execute every PyTorch operation family required by frozen MEPI inference."""

    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except Exception as exc:  # pragma: no cover - exercised by shell deployment checks
        raise PiRuntimeError(f"PYTORCH IMPORT FAILURE: {exc}") from exc
    if require_target_runtime and torch.__version__ != CPU_TORCH_VERSION:
        raise PiRuntimeError(
            f"PYTORCH FUNCTIONALITY FAILURE: expected {CPU_TORCH_VERSION}, got {torch.__version__}"
        )
    if require_target_runtime and torch.version.cuda is not None:
        raise PiRuntimeError(
            f"FORBIDDEN CUDA/NVIDIA DEPENDENCY: torch.version.cuda={torch.version.cuda}"
        )
    if require_target_runtime and torch.cuda.is_available():
        raise PiRuntimeError("FORBIDDEN CUDA/NVIDIA DEPENDENCY: CUDA unexpectedly available")
    try:
        tensor = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
        results: dict[str, object] = {
            "conv1d": list(nn.Conv1d(1, 2, 3)(tensor).shape),
            "rfft": list(torch.fft.rfft(tensor, dim=-1).shape),
            "adaptive_avg_pool1d": list(functional.adaptive_avg_pool1d(tensor, 8).shape),
            "layer_norm": list(nn.LayerNorm(32)(tensor).shape),
            "gelu_finite": bool(torch.isfinite(functional.gelu(tensor)).all()),
            "softplus_finite": bool(torch.isfinite(functional.softplus(tensor)).all()),
            "pow_finite": bool(torch.isfinite(torch.tensor([2.0]).pow(1.5)).all()),
        }
        attention = nn.MultiheadAttention(8, 2, batch_first=True)
        zeros = torch.zeros(1, 4, 8)
        results["attention"] = list(
            attention(zeros, zeros, zeros, need_weights=False)[0].shape
        )
        sequence = zeros
        for _ in range(8):
            sequence = nn.LSTM(8, 8, batch_first=True)(sequence)[0]
        results["eight_lstm_blocks"] = list(sequence.shape)
        if not all(value for key, value in results.items() if key.endswith("_finite")):
            raise ArithmeticError("non-finite required-operation result")
    except Exception as exc:
        raise PiRuntimeError(f"PYTORCH FUNCTIONALITY FAILURE: {exc}") from exc
    return {
        "status": "PASS",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "inference_device": "CPU",
        "operations": results,
    }


def replay_signatures(root: str | Path) -> dict[str, object]:
    """Return full scientific signatures for every packaged real replay record."""

    from apps.mepi_monitor.domain_guard.guard import PredictionDomain
    from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
    from apps.mepi_monitor.live.pipeline import run_capture
    from apps.mepi_monitor.live.replay import ReplaySource
    from apps.mepi_monitor.profiles.models import builtin_profiles

    root_path = Path(root).resolve()
    engine = FrozenMEPIEngine(root_path, device="cpu")
    domain = PredictionDomain(root_path / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json")
    replay = ReplaySource(root_path / "apps/mepi_monitor/assets/replay/manifest.json")
    profiles = builtin_profiles()
    records: list[dict[str, object]] = []
    for index, entry in enumerate(replay.entries()):
        capture, evidence = replay.load(index)
        result = run_capture(
            capture,
            ambient_temperature_c=float(evidence["ambient_temperature_c"]),
            profile=profiles[str(evidence["core_id"])],
            engine=engine,
            domain=domain,
        )
        records.append(
            {
                "sample_id": evidence["sample_id"],
                "measured": result.processed.measured,
                "derived": result.processed.derived,
                "features": result.processed.features,
                "b_waveform_t": result.processed.b_waveform_t.tolist(),
                "prediction": {
                    "efficiency_percent": result.prediction.efficiency_percent,
                    "core_loss_w": result.prediction.core_loss_w,
                    "lsp": result.prediction.lsp,
                    "lsp_sigma": result.prediction.lsp_sigma,
                },
                "domain_status": result.domain.status,
            }
        )
    return {
        "status": "PASS",
        "torch_version": __import__("torch").__version__,
        "records": records,
    }


def benchmark_runtime(root: str | Path) -> dict[str, object]:
    """Run a bounded CPU benchmark without changing scientific computation."""

    import torch

    from apps.mepi_monitor.domain_guard.guard import PredictionDomain
    from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
    from apps.mepi_monitor.live.pipeline import run_capture
    from apps.mepi_monitor.live.replay import ReplaySource
    from apps.mepi_monitor.preprocessing.pipeline import process_scope_capture
    from apps.mepi_monitor.profiles.models import builtin_profiles

    def process_seconds(code: str) -> float:
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return time.perf_counter() - started

    root_path = Path(root).resolve()
    python_startup = process_seconds("pass")
    torch_import = process_seconds("import torch")
    started = time.perf_counter()
    engine = FrozenMEPIEngine(root_path, device="cpu")
    model_load = time.perf_counter() - started
    domain = PredictionDomain(root_path / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json")
    replay = ReplaySource(root_path / "apps/mepi_monitor/assets/replay/manifest.json")
    capture, evidence = replay.load(0)
    profile = builtin_profiles()[str(evidence["core_id"])]
    started = time.perf_counter()
    processed = process_scope_capture(
        capture.time_s,
        capture.vin_v,
        capture.vout_v,
        frequency_hz=capture.frequency_hz,
        ambient_temperature_c=float(evidence["ambient_temperature_c"]),
        profile=profile,
    )
    preprocessing = time.perf_counter() - started
    started = time.perf_counter()
    prediction = engine.predict(processed.b_waveform_t, processed.features)
    single_inference = time.perf_counter() - started
    started = time.perf_counter()
    complete = run_capture(
        capture,
        ambient_temperature_c=float(evidence["ambient_temperature_c"]),
        profile=profile,
        engine=engine,
        domain=domain,
    )
    complete_replay = time.perf_counter() - started
    if not all(
        value == value
        for value in (
            prediction.efficiency_percent,
            prediction.core_loss_w,
            prediction.lsp,
            prediction.lsp_sigma,
            complete.prediction.efficiency_percent,
        )
    ):
        raise PiRuntimeError("PYTORCH FUNCTIONALITY FAILURE: non-finite benchmark prediction")
    return {
        "status": "PASS",
        "platform": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "inference_device": "CPU",
        "cuda_available": torch.cuda.is_available(),
        "python_startup_s": python_startup,
        "torch_import_s": torch_import,
        "model_load_s": model_load,
        "preprocessing_s": preprocessing,
        "single_inference_s": single_inference,
        "complete_replay_s": complete_replay,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "two_second_interval_reasonable_observation": complete_replay <= 2.0,
        "hard_realtime_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("report", type=Path)
    runtime = subparsers.add_parser("runtime")
    runtime.add_argument("--require-target", action="store_true")
    signatures = subparsers.add_parser("signatures")
    signatures.add_argument("--root", type=Path, required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--root", type=Path, required=True)
    subparsers.add_parser("installed-guard")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = validate_pip_report(args.report)
        elif args.command == "runtime":
            result = required_torch_operations(require_target_runtime=args.require_target)
        elif args.command == "signatures":
            result = replay_signatures(args.root)
        elif args.command == "benchmark":
            result = benchmark_runtime(args.root)
        else:
            result = validate_installed_package_guard()
    except PiRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
