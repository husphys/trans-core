"""Audit MagNet and create the clean deterministic 80/10/10 sample manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, project_root, resolve_path
from .constants import FINETUNE_TABULAR_FEATURES, PRETRAIN_TABULAR_FEATURES, WAVEFORM_LENGTH
from .reproducibility import sha256_file
from .splitting import deterministic_stratified_sample_split

REQUIRED_KEYS = ("B", "f", "T", "P", "material")


def _decode(value: Any) -> str:
    return value.decode("utf-8", errors="strict") if isinstance(value, bytes) else str(value)


def _numeric_dataset_summary(dataset: Any, *, batch_size: int = 4096) -> dict[str, Any]:
    minimum = float("inf")
    maximum = float("-inf")
    missing = nonfinite = 0
    for start in range(0, len(dataset), batch_size):
        values = np.asarray(dataset[start : start + batch_size])
        missing += int(np.isnan(values).sum())
        nonfinite += int((~np.isfinite(values)).sum())
        finite = values[np.isfinite(values)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
    return {
        "min": None if minimum == float("inf") else minimum,
        "max": None if maximum == float("-inf") else maximum,
        "missing_values": missing,
        "nonfinite_values": nonfinite,
    }


def audit_magnet(config_path: str | Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required for the MagNet audit") from error

    config = load_config(config_path)
    root = project_root(config)
    dataset_path = resolve_path(config, config["dataset"]["path"])
    split_dir = resolve_path(config, config["split"]["output_dir"])
    report_json = resolve_path(config, config["outputs"]["audit_json"])
    report_md = resolve_path(config, config["outputs"]["audit_markdown"])
    seed = int(config["split"]["seed"])

    split_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset_path, "r") as handle:
        missing_keys = sorted(set(REQUIRED_KEYS) - set(handle.keys()))
        if missing_keys:
            raise ValueError(f"MagNet HDF5 is missing required datasets: {missing_keys}")
        counts = {key: len(handle[key]) for key in REQUIRED_KEYS}
        if len(set(counts.values())) != 1:
            raise ValueError(f"MagNet datasets have inconsistent sample counts: {counts}")
        total = counts["B"]
        if handle["B"].ndim != 2:
            raise ValueError(f"B must be rank 2, got shape {handle['B'].shape}")

        materials = np.asarray([_decode(value) for value in handle["material"][:]])
        splits = deterministic_stratified_sample_split(materials, seed=seed)
        ids: list[str] = []
        waveform_hashes: list[str] = []
        invalid_waveform_length = 0
        waveform_nonfinite = 0
        batch_size = int(config["audit"].get("batch_size", 1024))
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            waveforms = np.asarray(handle["B"][start:stop], dtype=np.float32)
            frequencies = np.asarray(handle["f"][start:stop], dtype=np.float64).reshape(-1)
            temperatures = np.asarray(handle["T"][start:stop], dtype=np.float64).reshape(-1)
            invalid_waveform_length += int(
                waveforms.ndim != 2 or waveforms.shape[1] != WAVEFORM_LENGTH
            ) * len(waveforms)
            waveform_nonfinite += int((~np.isfinite(waveforms)).sum())
            for offset, waveform in enumerate(waveforms):
                index = start + offset
                waveform_digest = hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest()
                waveform_hashes.append(waveform_digest)
                sample_digest = hashlib.sha256()
                sample_digest.update(materials[index].encode("utf-8"))
                sample_digest.update(
                    np.asarray(
                        [frequencies[offset], temperatures[offset]], dtype="<f8"
                    ).tobytes()
                )
                sample_digest.update(waveform.astype("<f4").tobytes())
                ids.append("mag_" + sample_digest.hexdigest()[:24])

        duplicate_sample_ids = len(ids) - len(set(ids))
        waveform_hash_counts = Counter(waveform_hashes)
        exact_duplicate_waveforms = sum(value - 1 for value in waveform_hash_counts.values() if value > 1)
        split_counts = {name: int(np.sum(splits == name)) for name in ("train", "validation", "test")}
        material_counts = dict(sorted(Counter(materials).items()))

        for split_name in ("train", "validation", "test"):
            path = split_dir / f"magnet_{split_name}.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    ["sample_id", "sample_index", "split", "material", "waveform_sha256"]
                )
                for index in np.flatnonzero(splits == split_name):
                    writer.writerow(
                        [ids[index], int(index), split_name, materials[index], waveform_hashes[index]]
                    )

        report: dict[str, Any] = {
            "status": "PASS"
            if not duplicate_sample_ids and not waveform_nonfinite and not invalid_waveform_length
            else "FAIL",
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "total_samples": total,
            "samples_per_material": material_counts,
            "B_waveform_shape": list(handle["B"].shape),
            "expected_waveform_length": WAVEFORM_LENGTH,
            "invalid_waveform_length_samples": invalid_waveform_length,
            "ranges": {
                "B_waveform": _numeric_dataset_summary(handle["B"]),
                "frequency_hz": _numeric_dataset_summary(handle["f"]),
                "temperature_c": _numeric_dataset_summary(handle["T"]),
                "core_loss": _numeric_dataset_summary(handle["P"]),
            },
            "missing_values": {
                key: _numeric_dataset_summary(handle[key])["missing_values"]
                for key in ("B", "f", "T", "P")
            }
            | {"material": int(sum(not value.strip() for value in materials))},
            "nonfinite_values": {
                "B": waveform_nonfinite,
                **{
                    key: _numeric_dataset_summary(handle[key])["nonfinite_values"]
                    for key in ("f", "T", "P")
                },
            },
            "duplicate_sample_ids": duplicate_sample_ids,
            "exact_duplicate_waveforms": int(exact_duplicate_waveforms),
            "split": {
                "method": "seeded material-stratified sample split",
                "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "seed": seed,
                "counts": split_counts,
                "manifest_directory": str(split_dir),
            },
            "feature_contract": {
                "waveform": "B_waveform[1024]",
                "tabular": list(PRETRAIN_TABULAR_FEATURES),
                "material_usage": "metadata and temporary-head routing only",
                "target": "core_loss",
            },
        }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MagNet dataset audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Dataset SHA-256: `{report['dataset_sha256']}`",
        f"- Total samples: {report['total_samples']:,}",
        f"- B waveform shape: `{tuple(report['B_waveform_shape'])}`",
        f"- Duplicate sample IDs: {report['duplicate_sample_ids']}",
        f"- Exact duplicate waveforms: {report['exact_duplicate_waveforms']}",
        f"- Split seed: {seed}",
        "",
        "## Samples per material",
        "",
        "| Material | Samples |",
        "|---|---:|",
        *[f"| {name} | {count:,} |" for name, count in material_counts.items()],
        "",
        "## Exact split counts",
        "",
        "| Split | Samples |",
        "|---|---:|",
        *[f"| {name} | {count:,} |" for name, count in split_counts.items()],
        "",
        "Normalization is not fitted by this audit. The pretraining runner fits every scaler "
        "from the training manifest only and does not load the test manifest.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manuscript_dir = root / "artifacts" / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    (manuscript_dir / "magnet_sample_statistics.json").write_text(
        json.dumps(
            {
                "dataset_sha256": report["dataset_sha256"],
                "total_samples": total,
                "samples_per_material": material_counts,
                "ranges": report["ranges"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (manuscript_dir / "split_statistics.json").write_text(
        json.dumps(report["split"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (manuscript_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "pretraining": list(PRETRAIN_TABULAR_FEATURES),
                "fine_tuning": list(FINETUNE_TABULAR_FEATURES),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (manuscript_dir / "preprocessing_description.json").write_text(
        json.dumps(
            {
                "waveform": "validated finite B(t), deterministically resampled to 1024 when required",
                "inputs": "frequency_hz and temperature_c only",
                "target": "log1p(core_loss), then training-only population standardization",
                "inverse_target": "expm1(z * scale + mean)",
                "material": "metadata and temporary-head routing only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/magnet.yaml")
    arguments = parser.parse_args()
    report = audit_magnet(arguments.config)
    print(json.dumps({"status": report["status"], "total_samples": report["total_samples"]}, indent=2))


if __name__ == "__main__":
    main()
