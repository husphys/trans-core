from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import load_yaml, resolve_path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(config_path: str) -> Path:
    config = load_yaml(config_path)
    roots = [resolve_path(config, spec["path"]) for spec in config["sources"].values()]
    output = resolve_path(config, config["data"]["manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            stat = path.stat()
            rows.append(
                {
                    "source_root": str(root),
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(path),
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["source_root", "relative_path", "size_bytes", "sha256", "recorded_utc"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    print(build_manifest(args.config))


if __name__ == "__main__":
    main()

