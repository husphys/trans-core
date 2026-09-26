#!/usr/bin/env python3
"""Standalone MEPI Monitor entry point."""
from apps.mepi_monitor.pi_launcher import ensure_pi_openblas_preload

try:
    ensure_pi_openblas_preload()
except RuntimeError as error:
    raise SystemExit(f"MEPI RUNTIME ERROR: {error}") from error

from apps.mepi_monitor.tk_main import main
if __name__ == "__main__":
    main()
