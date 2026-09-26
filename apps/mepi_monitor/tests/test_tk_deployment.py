from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from apps.mepi_monitor.deployment import sha256_file, verify_deployment
from apps.mepi_monitor.domain_guard.guard import PredictionDomain
from apps.mepi_monitor.pi_runtime import (
    CPU_TORCH_VERSION,
    CPU_TORCH_WHEEL,
    CPU_TORCH_WHEEL_SHA256,
    PiRuntimeError,
    replay_signatures,
    validate_pip_report,
)
from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
from apps.mepi_monitor.live.replay import ReplaySource
from apps.mepi_monitor.profiles.models import builtin_profiles
from apps.mepi_monitor.ui_tk.controller import (
    MonitorWorker, aggregate_screening_rows, domain_display, load_model_information,
    make_custom_profile, repository_demo, scientific_signature,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def runtime():
    return (
        FrozenMEPIEngine(ROOT, device="cpu"),
        PredictionDomain(ROOT / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json"),
        ReplaySource(ROOT / "apps/mepi_monitor/assets/replay/manifest.json"),
    )


def test_domain_display_mapping() -> None:
    assert domain_display("GREEN — IN DOMAIN")[0] == "IN DOMAIN"
    assert domain_display("YELLOW — NEAR DOMAIN BOUNDARY")[0] == "NEAR DOMAIN BOUNDARY"
    assert domain_display("RED — OUT OF INVESTIGATED DOMAIN")[0] == "OUT OF INVESTIGATED DOMAIN"


def test_profile_selection_and_custom_warning() -> None:
    assert set(builtin_profiles()) == {"FE", "COMMERCIAL"}
    custom = make_custom_profile({"name":"Lab custom", "od_mm":34.58, "id_mm":20.81, "height_mm":17.68,
        "primary_turns":10, "secondary_turns":10, "ae_mode":"geometric", "effective_area_m2":1,
        "material_label":"unknown", "load_resistance_ohm":49.6025})
    assert custom.known_core is False
    assert custom.prediction_label == "UNSEEN CORE — EXPLORATORY PREDICTION"


def test_model_information_is_loaded_not_hardcoded() -> None:
    payload = load_model_information(ROOT)
    assert payload["test_rows"] == 90
    assert payload["metrics"]["efficiency"]["err95_percent"] == pytest.approx(4.438475707225697)


def test_offline_demo_and_transparent_aggregation(runtime) -> None:
    _engine, domain, _replay = runtime
    rows = repository_demo(ROOT, "FE")
    assert len(rows) > 0
    summary = aggregate_screening_rows(rows, domain, known_core=True)
    assert len(summary) == 15
    assert all(item["n"] >= 1 for item in summary)


def test_worker_lifecycle_queue_and_replay(runtime) -> None:
    engine, domain, replay = runtime
    worker = MonitorWorker(engine, domain, replay)
    worker.start(source="Simulation / Replay", profile=builtin_profiles()["FE"], ambient_temperature_c=None, once=True)
    deadline = time.monotonic() + 15
    messages = []
    while time.monotonic() < deadline:
        worker.drain(messages.append)
        if any(message.kind == "stopped" for message in messages): break
        time.sleep(.02)
    worker.stop()
    assert not worker.running
    assert [message.kind for message in messages].count("result") == 1
    assert not [message for message in messages if message.kind == "error"]
    result = next(message.payload for message in messages if message.kind == "result")
    assert scientific_signature(result)["training_performed"] is False
    assert len(result.processed.b_waveform_t) == 1024


def test_worker_rejects_accumulation(runtime) -> None:
    engine, domain, replay = runtime
    worker = MonitorWorker(engine, domain, replay)
    worker.start(source="Simulation / Replay", profile=builtin_profiles()["FE"], ambient_temperature_c=None, interval_s=2)
    with pytest.raises(RuntimeError, match="already running"):
        worker.start(source="Simulation / Replay", profile=builtin_profiles()["FE"], ambient_temperature_c=None)
    worker.stop()


def test_deployment_integrity_and_relative_paths_if_built() -> None:
    deployed = ROOT / "deploy/MEPI_PI_DEPLOY"
    if not deployed.is_dir(): pytest.skip("deployment not built yet")
    payload = verify_deployment(deployed)
    assert payload["status"] == "PASS"
    assert payload["source_git_commit"]
    for forbidden in (str(ROOT), "/home/diy-hus/transfomfer"):
        for path in deployed.rglob("*.py"):
            assert forbidden not in path.read_text(encoding="utf-8")


def test_checkpoint_hash_verification_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"correct")
    manifest = {"scientific_model_version":"MEPI v1.5", "training_performed":False,
                "runtime_artifacts":{"artifact.bin":sha256_file(tmp_path / "artifact.bin")}}
    (tmp_path / "deploy_manifest.json").write_text(json.dumps(manifest))
    assert verify_deployment(tmp_path)["status"] == "PASS"
    (tmp_path / "artifact.bin").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="CHECKPOINT INTEGRITY ERROR"):
        verify_deployment(tmp_path)


def test_pi_installer_rejects_armv7l_before_install(tmp_path: Path) -> None:
    installer = ROOT / "deploy/MEPI_PI_DEPLOY/install_pi.sh"
    if not installer.is_file():
        pytest.skip("deployment not built yet")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uname").write_text("#!/bin/sh\necho armv7l\n", encoding="utf-8")
    (fake_bin / "python3").write_text("#!/bin/sh\necho Python 3.11.2\n", encoding="utf-8")
    for path in fake_bin.iterdir():
        path.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=installer.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Detected architecture: armv7l" in result.stdout
    assert "Detected Python: Python 3.11.2" in result.stdout
    assert "ARMV7L DEPLOYMENT NOT SUPPORTED" in result.stdout
    assert "apt-get" not in result.stdout
    assert "pip install" not in result.stdout
    assert "sudo" not in result.stdout
    assert "aarch64" in result.stdout
    assert result.stderr == ""


def _cpu_pip_plan(*extra_packages: str) -> dict[str, object]:
    torch_item = {
        "download_info": {
            "url": (
                "https://download-r2.pytorch.org/whl/cpu/"
                "torch-2.14.0%2Bcpu-cp313-cp313-manylinux_2_28_aarch64.whl"
            ),
            "archive_info": {"hashes": {"sha256": CPU_TORCH_WHEEL_SHA256}},
        },
        "metadata": {
            "name": "torch",
            "version": CPU_TORCH_VERSION,
            "requires_dist": ["filelock", "typing-extensions>=4.10.0", "sympy>=1.13.3"],
        },
    }
    dependencies = [
        {"download_info": {}, "metadata": {"name": name, "version": "1.0"}}
        for name in extra_packages
    ]
    return {"install": [torch_item, *dependencies]}


def test_cpu_pip_plan_accepts_only_exact_official_wheel(tmp_path: Path) -> None:
    report = tmp_path / "pip-plan.json"
    report.write_text(json.dumps(_cpu_pip_plan("filelock", "sympy")), encoding="utf-8")
    result = validate_pip_report(report)
    assert result["status"] == "PASS"
    assert result["torch_version"] == CPU_TORCH_VERSION
    assert result["wheel"] == CPU_TORCH_WHEEL
    assert result["wheel_sha256"] == CPU_TORCH_WHEEL_SHA256
    assert result["forbidden_packages"] == []


@pytest.mark.parametrize("forbidden", ["nvidia-cublas", "cuda-toolkit", "triton"])
def test_cpu_pip_plan_rejects_gpu_dependencies_before_install(
    tmp_path: Path, forbidden: str
) -> None:
    report = tmp_path / "pip-plan.json"
    report.write_text(json.dumps(_cpu_pip_plan(forbidden)), encoding="utf-8")
    with pytest.raises(PiRuntimeError, match="FORBIDDEN CUDA/NVIDIA DEPENDENCY"):
        validate_pip_report(report)


def test_pi_installer_uses_disk_backed_tmp_when_system_tmp_is_small(
    tmp_path: Path,
) -> None:
    installer = ROOT / "deploy/MEPI_PI_DEPLOY/install_pi.sh"
    if not installer.is_file():
        pytest.skip("deployment not built yet")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "sudo-was-called"
    scripts = {
        "uname": "#!/bin/sh\necho aarch64\n",
        "python3": (
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = \"--version\" ]; then echo 'Python 3.13.5'; fi\n"
            "exit 0\n"
        ),
        "df": (
            "#!/bin/sh\n"
            "for last do :; done\n"
            "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
            "if [ \"$last\" = \"/tmp\" ]; then\n"
            "  echo 'tmpfs 463872 13872 450000 3% /tmp'\n"
            "else\n"
            "  echo '/dev/mmcblk0p2 30000000 7000000 23000000 24% /'\n"
            "fi\n"
        ),
        "findmnt": (
            "#!/bin/sh\n"
            "for last do :; done\n"
            "if [ \"$last\" = \"/tmp\" ]; then echo 'tmpfs tmpfs';\n"
            "else echo '/dev/mmcblk0p2 ext4'; fi\n"
        ),
        "sudo": f"#!/bin/sh\ntouch '{marker}'\nexit 99\n",
    }
    for name, content in scripts.items():
        path = fake_bin / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(fake_home),
            "MEPI_INSTALL_PREFLIGHT_ONLY": "1",
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=installer.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "System /tmp: tmpfs tmpfs; free 450000 KiB" in result.stdout
    assert f"Selected TMPDIR: {fake_home}/.cache/mepi-install-tmp/run-" in result.stdout
    assert "Selected TMPDIR filesystem: /dev/mmcblk0p2 ext4" in result.stdout
    assert "MEPI installer preflight: PASS" in result.stdout
    assert not marker.exists()


def test_pi_requirements_are_exact_cpu_only() -> None:
    requirements = (ROOT / "scripts/pi_deploy_templates/requirements-pi.txt").read_text(
        encoding="utf-8"
    )
    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert f"torch=={CPU_TORCH_VERSION}" in requirements
    assert "torch>=" not in requirements


def test_all_real_replay_records_have_complete_cpu_signatures() -> None:
    payload = replay_signatures(ROOT)
    assert payload["status"] == "PASS"
    assert len(payload["records"]) >= 2
    for record in payload["records"]:
        assert len(record["b_waveform_t"]) == 1024
        assert set(record["prediction"]) == {
            "efficiency_percent", "core_loss_w", "lsp", "lsp_sigma"
        }
        assert record["domain_status"]
