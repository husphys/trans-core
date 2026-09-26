"""Display-independent controllers for the Tkinter MEPI frontend."""

from __future__ import annotations

import json
import queue
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from apps.mepi_monitor.acquisition.keysight_lan import SCPI_PORT, KeysightLanScope
from apps.mepi_monitor.domain_guard.guard import PredictionDomain
from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
from apps.mepi_monitor.live.pipeline import LiveResult, run_capture
from apps.mepi_monitor.live.replay import ReplaySource
from apps.mepi_monitor.offline.loader import load_compatible_csv, load_repository_screening
from apps.mepi_monitor.profiles.models import (
    TransformerProfile,
    geometric_effective_area_m2,
)


@dataclass(frozen=True)
class WorkerMessage:
    kind: str
    payload: object


def domain_display(status: str) -> tuple[str, str]:
    """Return conservative display text and color for a backend status."""

    key = status.split()[0]
    mapping = {
        "GREEN": ("IN DOMAIN", "#176b43"),
        "YELLOW": ("NEAR DOMAIN BOUNDARY", "#8a6619"),
        "RED": ("OUT OF INVESTIGATED DOMAIN", "#862f39"),
    }
    if key not in mapping:
        raise ValueError(f"Unknown domain status: {status}")
    return mapping[key]


def make_custom_profile(values: dict[str, object]) -> TransformerProfile:
    mode = str(values["ae_mode"]).lower()
    od_mm, id_mm, height_mm = (float(values[key]) for key in ("od_mm", "id_mm", "height_mm"))
    area = (
        geometric_effective_area_m2(od_mm, id_mm, height_mm)
        if mode == "geometric"
        else float(values["effective_area_m2"])
    )
    profile = TransformerProfile(
        name=str(values["name"]), od_mm=od_mm, id_mm=id_mm, height_mm=height_mm,
        primary_turns=int(values["primary_turns"]), secondary_turns=int(values["secondary_turns"]),
        effective_area_m2=area, material_label=str(values.get("material_label") or "") or None,
        load_resistance_ohm=float(values["load_resistance_ohm"]), known_core=False, ae_mode=mode,
    )
    profile.validate()
    return profile


def load_model_information(root: str | Path) -> dict[str, object]:
    payload = json.loads((Path(root) / "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json").read_text())
    if payload.get("TRAINING_PERFORMED") is not False or payload.get("test_rows") != 90:
        raise AssertionError("Frozen post-hoc model information changed")
    return payload


def aggregate_screening_rows(
    rows: list[dict[str, Any]], domain: PredictionDomain, *, known_core: bool
) -> list[dict[str, object]]:
    """Aggregate repeats exactly as the existing desktop view: mean and sample SD."""

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row.get("candidate_frequency_hz", row["frequency_hz"]))].append(row)
    result: list[dict[str, object]] = []
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    for frequency in sorted(grouped):
        values = grouped[frequency]

        def stats(key: str) -> tuple[float, float]:
            series = [float(item[key]) for item in values]
            return statistics.mean(series), statistics.stdev(series) if len(series) > 1 else 0.0

        statuses = [
            domain.assess(
                {name: float(item[name]) for name in domain.payload["feature_order"]},
                known_core=known_core,
            ).status
            for item in values
        ]
        result.append({
            "frequency_hz": frequency,
            "n": len(values),
            "efficiency": stats("predicted_efficiency_percent"),
            "core_loss": stats("predicted_P_loss"),
            "lsp": stats("predicted_LSP_raw"),
            "lsp_sigma": statistics.mean(float(item["predicted_LSP_std_raw"]) for item in values),
            "domain_status": max(statuses, key=lambda value: rank[value.split()[0]]),
        })
    return result


def screen_compatible_csv(
    path: str | Path, engine: FrozenMEPIEngine, domain: PredictionDomain
) -> list[dict[str, Any]]:
    rows, waveforms = load_compatible_csv(path)
    screened = []
    for row, waveform in zip(rows, waveforms):
        features = row["features"]
        prediction = engine.predict(waveform, features)
        known = str(row.get("core_id", "")).upper() in {"FE", "COMMERCIAL"}
        assessment = domain.assess(features, known_core=known)
        screened.append({
            **row, **features,
            "predicted_efficiency_percent": prediction.efficiency_percent,
            "predicted_P_loss": prediction.core_loss_w,
            "predicted_LSP_raw": prediction.lsp,
            "predicted_LSP_std_raw": prediction.lsp_sigma,
            "domain_status": assessment.status,
        })
    return screened


def test_scope_connection(ip: str, port: int = SCPI_PORT, timeout_s: float = 6.0) -> str:
    """Issue only ``*IDN?`` and close; no acquisition or inference occurs."""

    if not ip.strip():
        raise ValueError("Scope IP is required")
    scope = KeysightLanScope(ip.strip(), port=int(port), timeout_s=timeout_s)
    try:
        return scope.connect()
    finally:
        scope.disconnect()


class MonitorWorker:
    """Single background acquisition/inference worker with a thread-safe result queue."""

    def __init__(self, engine: FrozenMEPIEngine, domain: PredictionDomain, replay: ReplaySource) -> None:
        self.engine, self.domain, self.replay = engine, domain, replay
        self.messages: queue.Queue[WorkerMessage] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self, *, source: str, profile: TransformerProfile, ambient_temperature_c: float | None,
        interval_s: float = 2.0, ip: str = "", port: int = SCPI_PORT, replay_index: int = 0,
        once: bool = False,
    ) -> None:
        if self.running:
            raise RuntimeError("A monitoring worker is already running")
        if interval_s < 0.25:
            raise ValueError("Update interval must be at least 0.25 seconds")
        self._stop.clear()
        kwargs = dict(source=source, profile=profile, ambient_temperature_c=ambient_temperature_c,
                      interval_s=float(interval_s), ip=ip, port=int(port), replay_index=int(replay_index), once=once)
        self._thread = threading.Thread(target=self._run, kwargs=kwargs, name="mepi-monitor-worker", daemon=True)
        self._thread.start()

    def _run(self, **request: object) -> None:
        scope: KeysightLanScope | None = None
        try:
            source = str(request["source"])
            if source == "Keysight LAN":
                scope = KeysightLanScope(str(request["ip"]), port=int(request["port"]))
                identity = scope.connect()
                self.messages.put(WorkerMessage("connection", {"status": "CONNECTED", "identity": identity}))
            elif source != "Simulation / Replay":
                raise ValueError(f"Unsupported source: {source}")
            while not self._stop.is_set():
                ambient = request["ambient_temperature_c"]
                if scope is None:
                    capture, evidence = self.replay.load(int(request["replay_index"]))
                    if ambient is None:
                        ambient = float(evidence["ambient_temperature_c"])
                else:
                    capture = scope.acquire()
                result = run_capture(
                    capture, ambient_temperature_c=None if ambient is None else float(ambient),
                    profile=request["profile"], engine=self.engine, domain=self.domain,
                )
                self.messages.put(WorkerMessage("result", result))
                if bool(request["once"]):
                    break
                if self._stop.wait(float(request["interval_s"])):
                    break
        except Exception as error:
            self.messages.put(WorkerMessage("error", error))
        finally:
            if scope is not None:
                scope.disconnect()
            self.messages.put(WorkerMessage("stopped", None))

    def stop(self, timeout_s: float = 8.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout_s)
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None

    def drain(self, callback: Callable[[WorkerMessage], None]) -> int:
        count = 0
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                return count
            callback(message)
            count += 1


def repository_demo(root: str | Path, core: str) -> list[dict[str, Any]]:
    return [row for row in load_repository_screening(root) if row["core_id"] == core]


def scientific_signature(result: LiveResult) -> dict[str, object]:
    """Serializable values used by standalone equivalence tests."""

    processed, prediction = result.processed, result.prediction
    return {
        "frequency_hz": processed.measured["frequency_hz"],
        "vin_rms_v": processed.measured["vin_rms_v"],
        "phase_shift_deg": processed.derived["phase_shift_deg"],
        "features": processed.features,
        "b_waveform_t": np.asarray(processed.b_waveform_t).tolist(),
        "predicted_efficiency_percent": prediction.efficiency_percent,
        "predicted_core_loss_w": prediction.core_loss_w,
        "predicted_lsp": prediction.lsp,
        "predicted_lsp_sigma": prediction.lsp_sigma,
        "domain_status": result.domain.status,
        "training_performed": False,
    }
