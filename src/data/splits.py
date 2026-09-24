"""Legacy pre-protocol split generator retained as audit evidence.

The frozen-protocol implementation is :mod:`src.mepi_v1.splitting`; this module's
70/15/15 outputs must not be used for MEPI v1 experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import load_yaml, resolve_path


SEED = 42
FINE_TUNE_GROUP_COLUMNS = [
    "output_power_w",
    "phase_shift_deg",
    "dBdt_max",
    "B_thd_percent",
    "form_factor",
]


def _quantile_bin(series: pd.Series, bins: int = 4) -> pd.Series:
    ranked = series.rank(method="average")
    return pd.qcut(ranked, q=bins, labels=False, duplicates="drop").fillna(0).astype(int).astype(str)


def _assign_groups(
    frame: pd.DataFrame,
    group_column: str,
    material_column: str,
    frequency_column: str,
    temperature_column: str,
    loss_column: str,
    waveform_column: str | None,
    seed: int = SEED,
) -> pd.Series:
    aggregations = {
        material_column: "first",
        frequency_column: "median",
        temperature_column: "median",
        loss_column: "median",
    }
    if waveform_column:
        aggregations[waveform_column] = "first"
    grouped = frame.groupby(group_column, sort=True).agg(aggregations).reset_index()
    material = grouped[material_column].astype(str)
    waveform = grouped[waveform_column].astype(str) if waveform_column else pd.Series("MISSING", index=grouped.index)
    f_bin = _quantile_bin(grouped[frequency_column])
    t_bin = _quantile_bin(grouped[temperature_column])
    p_bin = _quantile_bin(grouped[loss_column])
    fallback = material + "|" + waveform + "|f" + f_bin
    # Twenty approximately balanced, stratified group folds map exactly to 14/3/3 folds,
    # i.e. the requested 70/15/15 ratio without ever breaking a group.
    group_strata = pd.Series(fallback.to_numpy(), index=grouped[group_column])
    sample_strata = frame[group_column].map(group_strata)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.full(len(frame), -1, dtype=np.int16)
    for fold_index, (_, held_indices) in enumerate(
        splitter.split(frame, sample_strata, groups=frame[group_column])
    ):
        fold[held_indices] = fold_index
    if np.any(fold < 0):
        raise RuntimeError("Not every sample was assigned to a split fold")
    sizes = np.bincount(fold, minlength=20)
    target_train = 0.70 * len(frame)
    target_validation = 0.15 * len(frame)
    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    all_folds = set(range(20))
    for train_folds in itertools.combinations(range(20), 14):
        train_error = abs(int(sizes[list(train_folds)].sum()) - target_train)
        remaining = sorted(all_folds - set(train_folds))
        for validation_folds in itertools.combinations(remaining, 3):
            validation_error = abs(int(sizes[list(validation_folds)].sum()) - target_validation)
            score = train_error + validation_error
            if best is None or score < best[0]:
                best = (score, train_folds, validation_folds)
    assert best is not None
    train_folds = set(best[1])
    validation_folds = set(best[2])
    names = np.asarray(
        ["train" if value in train_folds else "validation" if value in validation_folds else "test" for value in fold]
    )
    return pd.Series(names, index=frame.index, dtype="string")


def _assert_no_overlap(frame: pd.DataFrame) -> None:
    sets = {
        name: set(frame.loc[frame["split"] == name, "group_id"])
        for name in ("train", "validation", "test")
    }
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])


def create_finetune_splits(path: Path, output_dir: Path) -> dict:
    frame = pd.read_csv(path, usecols=lambda column: column != "B_waveform")
    missing = [column for column in FINE_TUNE_GROUP_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"Cannot reconstruct fine-tune operating-condition groups; missing {missing}")
    group_key = frame[FINE_TUNE_GROUP_COLUMNS].astype(str).agg("|".join, axis=1)
    frame["group_id"] = group_key.map(lambda value: "ft_" + hashlib.sha256(value.encode()).hexdigest()[:16])
    frame["sample_id"] = [f"ft_{index:05d}" for index in range(len(frame))]
    frame["split"] = _assign_groups(
        frame,
        "group_id",
        material_column="core_type",
        frequency_column="frequency_hz",
        temperature_column="temperature_core_c",
        loss_column="P_loss",
        waveform_column="waveform_type",
    )
    _assert_no_overlap(frame)
    columns = [
        "sample_id",
        "group_id",
        "split",
        "core_type",
        "frequency_hz",
        "temperature_core_c",
        "waveform_type",
    ]
    for split_name in ("train", "validation", "test"):
        frame.loc[frame["split"] == split_name, columns].to_csv(
            output_dir / f"finetune_{split_name}.csv", index=False
        )
    return {
        "source": str(path),
        "group_definition": FINE_TUNE_GROUP_COLUMNS,
        "samples": {name: int((frame["split"] == name).sum()) for name in ("train", "validation", "test")},
        "groups": {
            name: int(frame.loc[frame["split"] == name, "group_id"].nunique())
            for name in ("train", "validation", "test")
        },
        "group_overlap": 0,
    }


def _decode(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def create_magnet_splits(path: Path, output_dir: Path) -> dict:
    rows = []
    with h5py.File(path, "r") as handle:
        missing = {"f", "T", "material"} - set(handle.keys())
        if missing:
            raise RuntimeError(f"Cannot reconstruct MagNet groups; missing keys: {sorted(missing)}")
        materials = handle["material"][:]
        frequencies = np.asarray(handle["f"][:]).reshape(-1)
        temperatures = np.asarray(handle["T"][:]).reshape(-1)
        losses = np.asarray(handle["P"][:]).reshape(-1)
        for index, (raw_material, raw_frequency, raw_temperature, raw_loss) in enumerate(
            zip(materials, frequencies, temperatures, losses, strict=True)
        ):
            material = _decode(raw_material)
            frequency = float(raw_frequency)
            temperature = float(raw_temperature)
            # The HDF5 file has no source/curve IDs or waveform-class field. Grouping all
            # records with the exact same material, temperature, and measured frequency is
            # deliberately conservative: different Bmax points and even different waveform
            # classes cannot leak across splits. No arbitrary frequency rounding is used.
            key = f"{material}|{temperature:.12g}|{frequency:.12g}"
            rows.append(
                {
                    "sample_id": f"magnet_{index:06d}",
                    "sample_index": index,
                    "group_id": "mag_" + hashlib.sha256(key.encode()).hexdigest()[:16],
                    "material": material,
                    "frequency": frequency,
                    "temperature": temperature,
                    "core_loss": float(raw_loss),
                    "waveform_type": "MISSING",
                    "source_filename": path.name,
                }
            )
    frame = pd.DataFrame(rows)
    frame["split"] = _assign_groups(
        frame,
        "group_id",
        material_column="material",
        frequency_column="frequency",
        temperature_column="temperature",
        loss_column="core_loss",
        waveform_column=None,
    )
    _assert_no_overlap(frame)
    for split_name in ("train", "validation", "test"):
        frame.loc[frame["split"] == split_name].to_csv(output_dir / f"magnet_{split_name}.csv", index=False)
    return {
        "source": str(path),
        "group_definition": ["material", "temperature", "exact_measured_frequency"],
        "samples": {name: int((frame["split"] == name).sum()) for name in ("train", "validation", "test")},
        "groups": {
            name: int(frame.loc[frame["split"] == name, "group_id"].nunique())
            for name in ("train", "validation", "test")
        },
        "group_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = resolve_path(config, ".")
    output = resolve_path(config, config["data"]["splits_dir"])
    output.mkdir(parents=True, exist_ok=True)
    magnet = sorted(root.glob(config["data"]["magnet_glob"]))
    fine = sorted(root.glob(config["data"]["legacy_finetune_glob"]))
    if len(magnet) != 1 or len(fine) != 1:
        raise RuntimeError(f"Expected one MagNet HDF5 and one fine-tune CSV; found {len(magnet)} and {len(fine)}")
    summary = {
        "seed": SEED,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "magnet": create_magnet_splits(magnet[0], output),
        "finetune": create_finetune_splits(fine[0], output),
    }
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
