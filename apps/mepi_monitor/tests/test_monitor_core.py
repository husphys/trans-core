from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apps.mepi_monitor.domain_guard.guard import PredictionDomain, build_prediction_domain
from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
from apps.mepi_monitor.live.pipeline import run_capture
from apps.mepi_monitor.live.replay import ReplaySource
from apps.mepi_monitor.offline.loader import load_repository_screening
from apps.mepi_monitor.preprocessing.pipeline import process_scope_capture
from apps.mepi_monitor.profiles.models import builtin_profiles, geometric_effective_area_m2
from src.evaluation.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "apps/mepi_monitor/assets/replay/manifest.json"
DOMAIN = ROOT / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json"


def test_err95_uses_repository_epsilon_and_formula() -> None:
    target = np.asarray([1.0, 2.0, 4.0])
    prediction = np.asarray([1.1, 1.8, 4.8])
    expected = 100.0 * np.quantile(np.abs(target - prediction) / (np.abs(target) + 1e-12), 0.95)
    assert regression_metrics(target, prediction)["err95"] == pytest.approx(expected)


def test_profile_calculation_reproduces_frozen_working_area() -> None:
    assert geometric_effective_area_m2(34.58, 20.81, 17.68) == pytest.approx(1.217268e-4)
    assert set(builtin_profiles()) == {"FE", "COMMERCIAL"}
    assert all(profile.known_core for profile in builtin_profiles().values())


def test_replay_uses_exact_1024_point_feature_pipeline() -> None:
    capture, evidence = ReplaySource(REPLAY).load(0)
    result = process_scope_capture(capture.time_s, capture.vin_v, capture.vout_v, frequency_hz=capture.frequency_hz, ambient_temperature_c=float(evidence["ambient_temperature_c"]), profile=builtin_profiles()["FE"])
    assert result.b_waveform_t.shape == (1024,)
    for name in ("vin_rms_v", "phase_shift_deg", "B_peak_t", "B_rms", "B_thd_percent", "dBdt_max", "form_factor"):
        actual = result.features[name]
        assert actual == pytest.approx(float(evidence[name]), rel=2e-5, abs=2e-6)


def test_domain_artifact_is_train_only_and_warns_without_clamping(tmp_path: Path) -> None:
    payload = build_prediction_domain(ROOT, tmp_path / "domain.json")
    assert payload["training_rows"] == 687 and payload["source_split"] == "train"
    guard = PredictionDomain(tmp_path / "domain.json")
    features = {name: (rule["training_min"] + rule["training_max"]) / 2 for name, rule in payload["variables"].items()}
    features["frequency_hz"] = payload["variables"]["frequency_hz"]["training_max"] + 1.0
    assessment = guard.assess(features, known_core=True)
    assert assessment.status.startswith("RED")
    assert features["frequency_hz"] > payload["variables"]["frequency_hz"]["training_max"]


def test_offline_demo_is_independent_and_complete() -> None:
    rows = load_repository_screening(ROOT)
    assert len(rows) == 146
    manifest = list(__import__("csv").DictReader((ROOT / "data/MEPI/demo_manifest_v2.csv").open(encoding="utf-8-sig")))
    assert {row["dataset_mode"] for row in manifest} == {"demo"}
    assert "split" not in manifest[0]


@pytest.fixture(scope="module")
def engine() -> FrozenMEPIEngine:
    return FrozenMEPIEngine(ROOT, device="cpu")


def test_frozen_model_strict_load_and_gui_independent_inference(engine: FrozenMEPIEngine) -> None:
    assert engine.status()["strict_load"] is True
    assert engine.status()["training_performed"] is False
    capture, evidence = ReplaySource(REPLAY).load(0)
    result = run_capture(capture, ambient_temperature_c=float(evidence["ambient_temperature_c"]), profile=builtin_profiles()["FE"], engine=engine, domain=PredictionDomain(DOMAIN))
    assert np.isfinite([result.prediction.efficiency_percent, result.prediction.core_loss_w, result.prediction.lsp, result.prediction.lsp_sigma]).all()
    assert result.prediction.lsp_sigma > 0.0


def test_custom_core_is_visibly_exploratory() -> None:
    source = builtin_profiles()["FE"]
    custom = source.__class__("Custom", source.od_mm, source.id_mm, source.height_mm, source.primary_turns, source.secondary_turns, source.effective_area_m2, None, source.load_resistance_ohm, False)
    assert custom.prediction_label == "UNSEEN CORE — EXPLORATORY PREDICTION"
