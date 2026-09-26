"""Launch the Tkinter MEPI monitor or run standalone-safe smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runtime(root: Path):
    from apps.mepi_monitor.deployment import verify_deployment
    from apps.mepi_monitor.domain_guard.guard import PredictionDomain
    from apps.mepi_monitor.inference.engine import FrozenMEPIEngine

    integrity = verify_deployment(root)
    engine = FrozenMEPIEngine(root, device="cpu")
    domain = PredictionDomain(root / "reports/MEPI_V1_5_PREDICTION_DOMAIN.json")
    return engine, domain, integrity


def replay_smoke(root: Path) -> dict[str, object]:
    from apps.mepi_monitor.live.pipeline import run_capture
    from apps.mepi_monitor.live.replay import ReplaySource
    from apps.mepi_monitor.profiles.models import builtin_profiles
    from apps.mepi_monitor.ui_tk.controller import scientific_signature

    engine, domain, integrity = build_runtime(root)
    replay = ReplaySource(root / "apps/mepi_monitor/assets/replay/manifest.json")
    capture, evidence = replay.load(0)
    result = run_capture(
        capture, ambient_temperature_c=float(evidence["ambient_temperature_c"]),
        profile=builtin_profiles()["FE"], engine=engine, domain=domain,
    )
    signature = scientific_signature(result)
    signature.update({
        "status": "PASS", "source": "Simulation / Replay", "sample_id": evidence["sample_id"],
        "waveform_length": len(result.processed.b_waveform_t),
        "checkpoint_sha256": engine.checkpoint_sha256,
        "deployment_integrity": integrity["status"],
        "strict_load": engine.status()["strict_load"],
        "scaler_loaded": engine.normalization.get("fitted_split") == "train",
    })
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--replay-smoke", action="store_true")
    parser.add_argument("--test-scope", metavar="IP")
    parser.add_argument("--port", type=int, default=5025)
    args = parser.parse_args()
    root = project_root()
    if args.test_scope:
        from apps.mepi_monitor.ui_tk.controller import test_scope_connection
        print(json.dumps({"status": "PASS", "identity": test_scope_connection(args.test_scope, args.port)}, indent=2))
        return
    if args.headless_smoke or args.replay_smoke:
        print(json.dumps(replay_smoke(root), indent=2, sort_keys=True))
        return
    engine, domain, _integrity = build_runtime(root)
    from apps.mepi_monitor.ui_tk.main_window import launch
    launch(root, engine, domain)


if __name__ == "__main__":
    main()
