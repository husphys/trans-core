from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from apps.mepi_monitor.deployment import sha256_file, verify_deployment
from apps.mepi_monitor.domain_guard.guard import PredictionDomain
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
