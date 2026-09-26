"""Launch the PySide6 MEPI monitor or run its GUI-independent smoke test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def headless_smoke(root: Path) -> dict[str, object]:
    from apps.mepi_monitor.domain_guard.guard import PredictionDomain
    from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
    from apps.mepi_monitor.live.pipeline import run_capture
    from apps.mepi_monitor.live.replay import ReplaySource
    from apps.mepi_monitor.profiles.models import builtin_profiles

    engine = FrozenMEPIEngine(root, device="cpu")
    domain = PredictionDomain(root / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json")
    replay = ReplaySource(root / "apps/mepi_monitor/assets/replay/manifest.json")
    capture, entry = replay.load(0)
    result = run_capture(capture, ambient_temperature_c=float(entry["ambient_temperature_c"]), profile=builtin_profiles()["FE"], engine=engine, domain=domain)
    return {
        "status": "PASS", "source": "Simulation / Replay", "sample_id": entry["sample_id"],
        "waveform_length": len(result.processed.b_waveform_t), "ambient_source": result.ambient_source,
        "checkpoint_sha256": engine.checkpoint_sha256, "domain_status": result.domain.status,
        "training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(); root = project_root()
    if args.headless_smoke:
        print(json.dumps(headless_smoke(root), indent=2, sort_keys=True)); return
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        raise SystemExit("PySide6 is required for the desktop GUI. Install requirements-demo.txt or use --headless-smoke.") from error
    from apps.mepi_monitor.domain_guard.guard import PredictionDomain
    from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
    from apps.mepi_monitor.ui.main_window import MainWindow
    if args.screenshot: os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv); window = MainWindow(root, FrozenMEPIEngine(root), PredictionDomain(root / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json")); window.show()
    if args.screenshot:
        from PySide6.QtCore import QTimer
        def save() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True); window.grab().save(str(args.screenshot)); app.quit()
        QTimer.singleShot(1200, save)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
