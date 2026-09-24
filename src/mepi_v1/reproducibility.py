"""Runtime and artifact provenance helpers."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _command(arguments: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            arguments, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(project_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _command(["git", "rev-parse", "HEAD"], project_root),
    }
    try:
        import torch

        metadata.update(
            {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        metadata.update(
            {
                "pytorch_version": None,
                "cuda_version": None,
                "cuda_available": False,
                "gpu_model": None,
            }
        )
    return metadata
