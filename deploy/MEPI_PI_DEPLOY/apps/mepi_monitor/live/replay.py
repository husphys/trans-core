"""Replay real synchronized scope records through the live path."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from apps.mepi_monitor.acquisition.keysight_lan import ScopeCapture


class ReplaySource:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def entries(self) -> tuple[dict[str, object], ...]:
        return tuple(self.payload["captures"])

    def load(self, index: int = 0) -> tuple[ScopeCapture, dict[str, object]]:
        entry = self.payload["captures"][index]
        path = self.manifest_path.parent / str(entry["file"])
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != entry["sha256"]:
            raise AssertionError("Replay capture SHA256 mismatch")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        expected = {"time_vin_s", "vin_primary_v", "time_vout_s", "vout_v"}
        if set(rows[0]) != expected:
            raise ValueError("Replay scope schema changed")
        t1 = np.asarray([float(row["time_vin_s"]) for row in rows])
        t2 = np.asarray([float(row["time_vout_s"]) for row in rows])
        if not np.allclose(t1, t2, rtol=0.0, atol=1e-12):
            raise ValueError("Replay CH1/CH2 are not synchronized")
        capture = ScopeCapture(
            time_s=t1,
            vin_v=np.asarray([float(row["vin_primary_v"]) for row in rows]),
            vout_v=np.asarray([float(row["vout_v"]) for row in rows]),
            frequency_hz=float(entry["frequency_hz"]),
            frequency_source="Recorded Keysight CH1 measurement",
        )
        return capture, entry
