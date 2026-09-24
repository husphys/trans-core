from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_record(project_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _command(["git", "rev-parse", "HEAD"], project_root),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["torch_cuda_build"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        if record["cuda_available"]:
            record["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        record["torch"] = None
        record["cuda_available"] = False
    return record


def write_runtime_record(path: Path, project_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_record(project_root), indent=2) + "\n", encoding="utf-8")

