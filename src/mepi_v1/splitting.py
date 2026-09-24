"""Deterministic leakage-safe split utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .constants import GROUP_KEY_FIELDS

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def stable_group_id(record: Mapping[str, Any]) -> str:
    missing = [field for field in GROUP_KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(
            "Cannot construct frozen operating-condition group; missing " + ", ".join(missing)
        )
    key = "|".join(str(record[field]) for field in GROUP_KEY_FIELDS)
    return "op_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def deterministic_group_split(
    group_ids: Sequence[str], *, seed: int = 42
) -> np.ndarray:
    """Assign complete groups while approximating 80/10/10 sample counts."""

    groups = np.asarray(group_ids, dtype=str)
    unique, counts = np.unique(groups, return_counts=True)
    if unique.size < 3:
        raise ValueError("At least three operating-condition groups are required")
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(unique.size)
    order = np.lexsort((tie_breaker, -counts))
    targets = np.asarray(SPLIT_RATIOS) * len(groups)
    assigned_counts = np.zeros(3, dtype=np.int64)
    assignments: dict[str, str] = {}
    for group_index in order:
        group_size = counts[group_index]
        scores = []
        for split_index in range(3):
            proposed = assigned_counts.copy()
            proposed[split_index] += group_size
            scores.append(float(np.sum(((proposed - targets) / targets) ** 2)))
        selected = int(np.argmin(scores))
        assigned_counts[selected] += group_size
        assignments[unique[group_index]] = SPLIT_NAMES[selected]
    if any(name not in assignments.values() for name in SPLIT_NAMES):
        raise RuntimeError("Group assignment did not produce three non-empty subsets")
    return np.asarray([assignments[group] for group in groups])


def deterministic_exact_group_split(
    group_ids: Sequence[str],
    *,
    seed: int = 42,
    expected_group_count: int = 90,
    train_group_count: int = 72,
    validation_group_count: int = 9,
    test_group_count: int = 9,
) -> np.ndarray:
    """Assign exact seeded group counts while keeping every group indivisible."""

    groups = np.asarray(group_ids, dtype=str)
    unique = np.unique(groups)
    requested = train_group_count + validation_group_count + test_group_count
    if unique.size != expected_group_count or requested != expected_group_count:
        raise ValueError(
            f"Expected {expected_group_count} unique groups and matching requested counts; "
            f"found {unique.size} groups and {requested} requested assignments"
        )
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    assignments = {
        group: (
            "train"
            if index < train_group_count
            else "validation"
            if index < train_group_count + validation_group_count
            else "test"
        )
        for index, group in enumerate(shuffled)
    }
    return np.asarray([assignments[group] for group in groups])


def deterministic_stratified_sample_split(
    strata: Sequence[str], *, seed: int = 42
) -> np.ndarray:
    """Create a seeded 80/10/10 sample split stratified by material metadata."""

    labels = np.asarray(strata, dtype=str)
    result = np.empty(labels.shape[0], dtype="U10")
    for offset, label in enumerate(sorted(np.unique(labels))):
        indices = np.flatnonzero(labels == label)
        rng = np.random.default_rng(seed + offset)
        indices = indices.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_validation = int(round(0.1 * n))
        n_test = int(round(0.1 * n))
        result[indices[:n_validation]] = "validation"
        result[indices[n_validation : n_validation + n_test]] = "test"
        result[indices[n_validation + n_test :]] = "train"
    return result


def assert_disjoint_splits(
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    splits: Sequence[str],
    waveform_hashes: Sequence[str] | None = None,
) -> None:
    split_array = np.asarray(splits, dtype=str)
    if set(np.unique(split_array)) != set(SPLIT_NAMES):
        raise AssertionError(f"Expected exactly {SPLIT_NAMES}, got {sorted(set(split_array))}")

    def sets(values: Iterable[str]) -> dict[str, set[str]]:
        array = np.asarray(list(values), dtype=str)
        return {name: set(array[split_array == name]) for name in SPLIT_NAMES}

    for label, values in (("sample ID", sample_ids), ("group ID", group_ids)):
        members = sets(values)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = members[left] & members[right]
            if overlap:
                raise AssertionError(f"{label} overlap between {left} and {right}: {len(overlap)}")
    if waveform_hashes is not None:
        members = sets(waveform_hashes)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = members[left] & members[right]
            if overlap:
                raise AssertionError(
                    f"waveform hash overlap between {left} and {right}: {len(overlap)}"
                )
