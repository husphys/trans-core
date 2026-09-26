"""Fail-closed integrity checks for the standalone Raspberry Pi package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_deployment(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    path = root / "deploy_manifest.json"
    if not path.is_file():
        return {"status": "REPOSITORY_MODE", "deployment_manifest": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scientific_model_version") != "MEPI v1.5" or payload.get("training_performed") is not False:
        raise RuntimeError("CHECKPOINT INTEGRITY ERROR: invalid deployment scientific identity")
    artifacts = payload.get("runtime_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("CHECKPOINT INTEGRITY ERROR: deployment artifact manifest is missing")
    for relative, expected in artifacts.items():
        candidate = root / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise RuntimeError(f"CHECKPOINT INTEGRITY ERROR: {relative}")
    return {"status": "PASS", "deployment_manifest": True, **payload}
