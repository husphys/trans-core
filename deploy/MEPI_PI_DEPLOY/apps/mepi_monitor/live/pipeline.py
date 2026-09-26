"""One shared live/replay processing and inference path."""

from __future__ import annotations

from dataclasses import dataclass

from apps.mepi_monitor.acquisition.keysight_lan import ScopeCapture
from apps.mepi_monitor.domain_guard.guard import DomainAssessment, PredictionDomain
from apps.mepi_monitor.inference.engine import FrozenMEPIEngine, Prediction
from apps.mepi_monitor.preprocessing.pipeline import ProcessedCapture, process_scope_capture
from apps.mepi_monitor.profiles.models import TransformerProfile


@dataclass(frozen=True)
class LiveResult:
    processed: ProcessedCapture
    prediction: Prediction
    domain: DomainAssessment
    ambient_source: str = "Manual input"


def run_capture(
    capture: ScopeCapture,
    *,
    ambient_temperature_c: float | None,
    profile: TransformerProfile,
    engine: FrozenMEPIEngine,
    domain: PredictionDomain,
) -> LiveResult:
    processed = process_scope_capture(
        capture.time_s,
        capture.vin_v,
        capture.vout_v,
        frequency_hz=capture.frequency_hz,
        ambient_temperature_c=ambient_temperature_c,
        profile=profile,
    )
    prediction = engine.predict(processed.b_waveform_t, processed.features)
    assessment = domain.assess(processed.features, known_core=profile.known_core)
    return LiveResult(processed, prediction, assessment)
