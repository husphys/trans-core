"""Raspberry Pi launcher guard for the bundled PyTorch OpenBLAS runtime."""

from __future__ import annotations

import os
import platform
import site
import sys
from pathlib import Path
from typing import Iterable, Mapping

TARGET_ARCHITECTURES = {"aarch64", "arm64"}
OPENBLAS_RELATIVE_PATH = Path("torch/lib/libopenblas.so.0")


def find_torch_openblas(
    *, home: Path | None = None, search_paths: Iterable[str] | None = None
) -> Path | None:
    """Locate the wheel-bundled library without importing torch."""

    candidates: set[Path] = set()
    paths = list(sys.path if search_paths is None else search_paths)
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        paths.append(user_site)
    else:
        paths.extend(user_site)
    for value in paths:
        if value:
            candidates.add(Path(value).expanduser() / OPENBLAS_RELATIVE_PATH)
    home_path = Path.home() if home is None else Path(home)
    candidates.update(
        home_path.glob(".local/lib/python*/site-packages/torch/lib/libopenblas.so.0")
    )
    for candidate in sorted(candidates, key=str):
        if candidate.is_file():
            return candidate.resolve()
    return None


def prepare_pi_openblas_environment(
    *,
    machine: str | None = None,
    home: Path | None = None,
    search_paths: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return a re-exec environment on Pi, or ``None`` when already ready/not needed."""

    architecture = platform.machine() if machine is None else machine
    if architecture not in TARGET_ARCHITECTURES:
        return None
    library = find_torch_openblas(home=home, search_paths=search_paths)
    if library is None:
        raise RuntimeError(
            "required bundled torch/lib/libopenblas.so.0 is missing; "
            "rerun ./install_pi.sh before launching MEPI"
        )
    current = dict(os.environ if environ is None else environ)
    existing = current.get("LD_PRELOAD", "")
    tokens = existing.replace(":", " ").split()
    if str(library) in tokens:
        return None
    current["LD_PRELOAD"] = f"{library} {existing}".strip()
    current["MEPI_OPENBLAS_PRELOAD"] = str(library)
    return current


def ensure_pi_openblas_preload() -> None:
    """Re-exec Python once so OpenBLAS is loaded before any torch import."""

    environment = prepare_pi_openblas_environment()
    if environment is not None:
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
