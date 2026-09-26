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

DEPLOYMENT_VERSION = "1.2.1"

RUNTIME_PATHS = (
    "apps/__init__.py", "apps/mepi_monitor/__init__.py", "apps/mepi_monitor/acquisition",
    "apps/mepi_monitor/preprocessing", "apps/mepi_monitor/inference", "apps/mepi_monitor/domain_guard",
    "apps/mepi_monitor/profiles", "apps/mepi_monitor/offline", "apps/mepi_monitor/live",
    "apps/mepi_monitor/assets/replay", "apps/mepi_monitor/ui_tk",
    "apps/mepi_monitor/deployment.py", "apps/mepi_monitor/pi_launcher.py",
    "apps/mepi_monitor/pi_runtime.py", "apps/mepi_monitor/tk_main.py",
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
    "apps/mepi_monitor/pi_launcher.py", "apps/mepi_monitor/pi_runtime.py",
    "run_mepi.py", "requirements-pi.txt", "install_pi.sh", "test_pi.sh",
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
from apps.mepi_monitor.pi_launcher import ensure_pi_openblas_preload

try:
    ensure_pi_openblas_preload()
except RuntimeError as error:
    raise SystemExit(f"MEPI RUNTIME ERROR: {error}") from error

from apps.mepi_monitor.tk_main import main
if __name__ == "__main__":
    main()
'''

TEMPLATE_FILES = {
    "scripts/pi_deploy_templates/requirements-pi.txt": "requirements-pi.txt",
    "scripts/pi_deploy_templates/install_pi.sh": "install_pi.sh",
    "scripts/pi_deploy_templates/test_pi.sh": "test_pi.sh",
    "scripts/pi_deploy_templates/README_PI.md": "README_PI.md",
}

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
    for source_relative, target_name in TEMPLATE_FILES.items():
        shutil.copy2(root / source_relative, target / target_name)
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
        "pytorch_runtime": {
            "version": "2.14.0+cpu", "index_url": "https://download.pytorch.org/whl/cpu",
            "wheel_filename": "torch-2.14.0+cpu-cp313-cp313-manylinux_2_28_aarch64.whl",
            "wheel_sha256": "092d5c12938850dfbd90a654b3c8dac34c33e300f88eb19ee6f4ef93992c6347",
            "python": "CPython 3.13.x", "platform": "manylinux_2_28_aarch64",
            "cpu_only": True, "forbidden_packages": ["nvidia-*", "cuda-*", "triton"],
        },
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
