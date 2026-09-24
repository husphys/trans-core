from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.config import load_yaml, resolve_path


def download_sources(config_path: str, source_names: list[str] | None = None) -> None:
    config = load_yaml(config_path)
    sources = config["sources"]
    selected = source_names or list(sources)
    for name in selected:
        spec = sources[name]
        destination = resolve_path(config, spec["path"])
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            spec["url"],
            "--output",
            f"{destination}/",
            "--remaining-ok",
            "--continue",
        ]
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--source", action="append", choices=["datasets", "legacy_artifacts", "new_measurements"])
    args = parser.parse_args()
    download_sources(args.config, args.source)


if __name__ == "__main__":
    main()

