from __future__ import annotations

from pathlib import Path

import pytest

from apps.mepi_monitor.pi_launcher import (
    find_torch_openblas,
    prepare_pi_openblas_environment,
)


def _bundled_library(home: Path) -> Path:
    library = home / ".local/lib/python3.13/site-packages/torch/lib/libopenblas.so.0"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"test-library")
    return library.resolve()


def test_pi_launcher_finds_user_site_openblas_without_hardcoded_home(tmp_path: Path) -> None:
    library = _bundled_library(tmp_path)
    assert find_torch_openblas(home=tmp_path, search_paths=[]) == library
    environment = prepare_pi_openblas_environment(
        machine="aarch64", home=tmp_path, search_paths=[], environ={"PATH": "/usr/bin"}
    )
    assert environment is not None
    assert environment["LD_PRELOAD"] == str(library)
    assert environment["MEPI_OPENBLAS_PRELOAD"] == str(library)


def test_pi_launcher_preserves_existing_preloads_and_does_not_loop(tmp_path: Path) -> None:
    library = _bundled_library(tmp_path)
    environment = prepare_pi_openblas_environment(
        machine="aarch64",
        home=tmp_path,
        search_paths=[],
        environ={"LD_PRELOAD": "/opt/other.so"},
    )
    assert environment is not None
    assert environment["LD_PRELOAD"] == f"{library} /opt/other.so"
    assert prepare_pi_openblas_environment(
        machine="aarch64",
        home=tmp_path,
        search_paths=[],
        environ={"LD_PRELOAD": environment["LD_PRELOAD"]},
    ) is None


def test_pi_launcher_fails_clearly_when_bundled_library_is_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="required bundled torch/lib/libopenblas.so.0 is missing"):
        prepare_pi_openblas_environment(
            machine="aarch64", home=tmp_path, search_paths=[], environ={}
        )


def test_pi_launcher_is_noop_on_development_host(tmp_path: Path) -> None:
    assert prepare_pi_openblas_environment(
        machine="x86_64", home=tmp_path, search_paths=[], environ={}
    ) is None
