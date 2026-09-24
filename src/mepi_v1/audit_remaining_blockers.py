"""Produce fail-closed MEPI v1.1 voltage and LSP blocker audits.

This command is intentionally read-only with respect to the reprocessed dataset.
It does not infer unresolved voltages, create splits, fit normalization, access a
test subset, or start training.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .build_finetune_dataset import EXPECTED_FREQUENCIES, EXPECTED_VOLTAGES, _write_csv

PROVENANCE_FIELDS = [
    "session_id",
    "core_id",
    "row_count",
    "old_vin_set_group_v",
    "min_vin_reference_v",
    "median_vin_reference_v",
    "max_vin_reference_v",
    "min_vin_rms_v",
    "median_vin_rms_v",
    "max_vin_rms_v",
    "frequency_count",
    "repeat_count",
    "current_canonical_voltage",
    "mapping_status",
]

MAPPING_FIELDS = [
    "session_id",
    "core_id",
    "original_group_v",
    "median_vin_reference_v",
    "median_vin_rms_v",
    "canonical_nominal_v",
    "mapping_basis",
    "confirmed_by_experimenter",
]

GROUP_FIELDS = [
    "condition_group_id",
    "FE_valid_count",
    "COMMERCIAL_valid_count",
    "total_valid_count",
    "hardfail_count",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: float) -> str:
    return f"{value:.15g}"


def session_voltage_provenance(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["dataset_mode"] != "finetune":
            raise AssertionError("The finetune master must not contain demo rows")
        by_session[row["session_id"]].append(row)
    if len(rows) != 900 or len(by_session) != 12:
        raise AssertionError("Expected exactly 900 rows in 12 finetune sessions")

    output: list[dict[str, Any]] = []
    for session_id, members in sorted(
        by_session.items(), key=lambda item: (item[1][0]["core_id"], item[0])
    ):
        def unique(field: str) -> str:
            values = {row[field] for row in members}
            if len(values) != 1:
                raise AssertionError(f"{session_id}: inconsistent {field}")
            return values.pop()

        references = [float(row["vin_reference_v"]) for row in members]
        measured = [float(row["vin_rms_v"]) for row in members]
        canonical = unique("canonical_vin_set_group_v")
        output.append(
            {
                "session_id": session_id,
                "core_id": unique("core_id"),
                "row_count": len(members),
                "old_vin_set_group_v": unique("original_vin_set_group_v"),
                "min_vin_reference_v": _number(min(references)),
                "median_vin_reference_v": _number(statistics.median(references)),
                "max_vin_reference_v": _number(max(references)),
                "min_vin_rms_v": _number(min(measured)),
                "median_vin_rms_v": _number(statistics.median(measured)),
                "max_vin_rms_v": _number(max(measured)),
                "frequency_count": len({float(row["frequency_set_hz"]) for row in members}),
                "repeat_count": len({int(row["repeat_id"]) for row in members}),
                "current_canonical_voltage": canonical,
                "mapping_status": unique("voltage_grouping_provenance"),
            }
        )
    return output


def manual_mapping_template(provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in provenance:
        canonical = row["current_canonical_voltage"]
        certain = canonical not in (None, "")
        output.append(
            {
                "session_id": row["session_id"],
                "core_id": row["core_id"],
                "original_group_v": row["old_vin_set_group_v"],
                "median_vin_reference_v": row["median_vin_reference_v"],
                "median_vin_rms_v": row["median_vin_rms_v"],
                "canonical_nominal_v": canonical if certain else "",
                "mapping_basis": (
                    "EXISTING_EVIDENCE_EXACT_FROZEN_LABEL"
                    if certain
                    else "AWAITING_EXPERIMENTER_CONFIRMATION"
                ),
                # This field records a human action, so it must never be inferred.
                "confirmed_by_experimenter": 0,
            }
        )
    return output


def _condition_group_id(frequency: int, voltage: float) -> str:
    voltage_text = str(voltage).replace(".", "p")
    return f"F{frequency}_V{voltage_text}_RL49p6025_SINE"


def group_completeness(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize all 90 intended groups using only current supported mappings."""

    output = []
    for voltage in EXPECTED_VOLTAGES:
        for frequency in EXPECTED_FREQUENCIES:
            members = [
                row
                for row in rows
                if row["canonical_vin_set_group_v"] not in (None, "")
                and float(row["canonical_vin_set_group_v"]) == voltage
                and int(round(float(row["frequency_set_hz"]))) == frequency
            ]
            valid = [row for row in members if row["qc_status"] != "HARD_FAIL"]
            fe_valid = sum(row["core_id"] == "FE" for row in valid)
            commercial_valid = sum(row["core_id"] == "COMMERCIAL" for row in valid)
            output.append(
                {
                    "condition_group_id": _condition_group_id(frequency, voltage),
                    "FE_valid_count": fe_valid,
                    "COMMERCIAL_valid_count": commercial_valid,
                    "total_valid_count": fe_valid + commercial_valid,
                    "hardfail_count": sum(row["qc_status"] == "HARD_FAIL" for row in members),
                }
            )
    if len(output) != 90:
        raise AssertionError("Expected exactly 90 intended operating groups")
    return output


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = PROVENANCE_FIELDS
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in headers) + " |")
    return "\n".join(lines)


def _terminal_table(rows: list[dict[str, Any]]) -> str:
    widths = {
        field: max(len(field), *(len(str(row[field])) for row in rows))
        for field in PROVENANCE_FIELDS
    }
    header = "  ".join(field.ljust(widths[field]) for field in PROVENANCE_FIELDS)
    divider = "  ".join("-" * widths[field] for field in PROVENANCE_FIELDS)
    body = [
        "  ".join(str(row[field]).ljust(widths[field]) for field in PROVENANCE_FIELDS)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_lsp_trace(path: Path) -> None:
    path.write_text(
        """# LSP source trace

## Search scope

Focused terms: `LSP`, `lifespan`, `life span`, `Arrhenius`, `activation energy`, `Ea`, `epsilon`, `Tref`, `T_ref`, `temperature acceleration`, `aging`, `ageing`, `lifetime`, and `thermal degradation`.

Searched manuscript DOCX text and Office Math, protocol documents, Markdown reports, YAML/JSON configuration, Python source, notebook source cells, the frozen reference checkout, archived acquisition code/configuration, and bibliography entries. No PDF, BibTeX, or TeX manuscript copy exists in the project or raw archive. DOCX locations are paragraph numbers because the file has no stable fixed-page representation.

## Relevant occurrences

| file | location/line/page | exact role | equation/definition found | constants found | citation/source | active or historical |
|---|---|---|---|---|---|---|
| `Manuscript_clean.docx` | paragraphs 43-49, equations 1-2 | Primary manuscript LSP target definition | `Drel = exp[-Ea/kB (1/Tc - 1/Tref)]`; `LSP = 1/(Drel + epsilon)`; `Tc` and `Tref` are Kelvin; higher LSP means lower relative thermal stress | Names `Ea`, `kB`, `Tref`, `epsilon`; no numeric `Ea`, no numeric `epsilon`, and no executable `Tref` estimator | Paragraph 15 points to reference [15]; paragraph 49 says training-set determination | active manuscript |
| `Manuscript_clean.docx` | paragraphs 99-110, equations 14-19 | PIRL Arrhenius residual definition | `Trise = Tcore - Tamb`; `TK = Tcore + 273.15`; `LSPAr = 1/(Drel + epsilon)`; residual against provisional LSP and latent projection | Still no numeric `Ea`, `epsilon`, or exact `Tref`; residual normalization is mentioned but not mathematically specified | Internal references to equations 1-2 | active manuscript |
| `Manuscript_clean.docx` | paragraph 15 and bibliography entry [15], paragraph 260 | General Arrhenius prior citation | No project-local reproducible LSP parameterization | No LSP constants | Bilyaz et al. (2024), Heliyon 10(6), e27783, DOI 10.1016/j.heliyon.2024.e27783; full text not present locally | active bibliography |
| `MEPI-FROZEN-PROTOCOL v1.1.md` | lines 350-396 | Frozen target/input contract | Delegates LSP construction to the manuscript | No constants or estimator | Frozen protocol | active authoritative protocol |
| `docs/MEPI-FROZEN-PROTOCOL v1.1.md` | lines 350-396 | Byte-identical documentation copy | Same delegation as root protocol | No constants or estimator | Frozen protocol copy | active authoritative protocol |
| `configs/finetune_v1.yaml` | lines 26-29 | Machine-readable scientific configuration | Required fields exist | `activation_energy_ev`, `reference_temperature_k`, and `epsilon` are all `null` | Project configuration | active, fail-closed |
| `src/mepi_v1/targets.py` | lines 11-83 | Executable Arrhenius/LSP implementation shape | Implements the manuscript equation and Celsius-to-Kelvin conversion | Numeric `kB = 8.617333262145e-5 eV/K`; requires caller-supplied positive `Ea`, `Tref`, `epsilon` | Project source | active, blocked without config |
| `src/mepi_v1/finetune.py` | lines 29-32 | Training readiness gate | Requires three Arrhenius configuration fields | Rejects missing values; supplies none | Project source | active, fail-closed |
| `src/mepi_v1/downstream_audit.py` | lines 355-375 | Dataset audit and target-construction gate | Constructs `ArrheniusConfig` only after metadata checks | Supplies none; reads config | Project source | active, fail-closed |
| `BLOCKERS.md` | lines 40-49 | Scientific blocker registry | Records manuscript under-specification | Missing numeric `Ea`, numeric `epsilon`, and exact train-only `Tref` estimator | Project evidence register | active |
| `/mnt/e/MEPI/mepi_data_acquisition.py` | lines 89, 2442-2443, 2548 | Archived acquisition behavior | Leaves LSP blank and pending | Explicitly records exact manuscript formulation as pending | Checksummed raw archive | historical evidence |
| `/mnt/e/MEPI/MEPI_DATA/{finetune,demo}/session_*/config.json` | line 29 in all 14 configs | Session-level provenance | `lsp_status` is pending exact manuscript formulation | None | Checksummed raw session configs | historical evidence |
| `references/transformer-core/pretrain_mepi_bigru.py` | lines 118-128 | Legacy learned residual feature, not the manuscript LSP target | Uses learned `A*exp(-Ea/T)` and stacks it with a learned Steinmetz term | Clamps learned `Ea` to 0.1-15 and `A` to 1e-3-50; these are model outputs, not scientific constants | Frozen legacy reference checkout | historical, non-authoritative |
| `references/transformer-core/lambda_optimize.py` | lines 131-142 | Legacy learned residual feature | Same learned `A*exp(-Ea/T)` residual; no `Drel`, `Tref`, epsilon, or LSP target construction | Same learned clamps only | Frozen legacy reference checkout | historical, non-authoritative |
| `references/transformer-core/pretrain_experiment_1.ipynb` | source cell 3, lines 9-19 | Notebook copy of legacy residual | Same learned Arrhenius feature | Same learned clamps only | Frozen legacy reference checkout | historical, non-authoritative |
| `tests/test_mepi_v1_targets.py` | lines 20-22 | Unit-test fixture | Exercises the executable function | `Ea=0.5`, `Tref=300`, `epsilon=1e-8` are synthetic test inputs, not provenance | Automated test | active test only |
| `tests/test_downstream_preparation_contract.py` | lines 137-144 | Fail-closed and synthetic-positive test fixtures | Confirms missing config blocks and illustrative config executes | Same illustrative values; not authoritative | Automated test | active test only |
| `reports/lsp_definition_audit.json`, `reports/MEPI_V1_1_FINAL_TRAIN_READINESS.md`, `results/reports/data_audit.md` | prior audit outputs | Derived summaries of the sources above | Repeat the symbolic definition/incompleteness finding | Supply no new constants | Project-generated reports | historical audit evidence |

## Reproducibility determination

The project defines the symbolic temperature-to-proxy relationship and an executable function shape, but it does not contain enough authoritative information to reproduce the LSP target. The numeric examples in tests and the learned/clamped quantities in the legacy reference model are not scientific provenance and must not be promoted into the frozen target definition.

Exact missing mathematical quantities/rules:

1. Numeric effective activation energy `Ea`, its units, material applicability, and provenance.
2. Numeric stability constant `epsilon` and its intended scale.
3. Exact train-only `Tref` estimator (for example, which statistic, population, grouping, and units); “determined from the training set” is not executable.
4. Exact LSP normalization/scaling procedure and any bounds, including which parameters are learned from train only and how they are applied to validation/test.
5. Exact residual-normalization rule if the manuscript PIRL residual is to be reproduced alongside the target.

LSP_DEFINITION_INCOMPLETE
""",
        encoding="utf-8",
    )


def run(project_root: Path) -> list[dict[str, Any]]:
    master_path = project_root / "data/MEPI/v1_1/samples_master_reprocessed_v2.csv"
    master_before = master_path.read_bytes()
    rows = _read_csv(master_path)
    provenance = session_voltage_provenance(rows)
    mapping = manual_mapping_template(provenance)
    groups = group_completeness(rows)

    reports = project_root / "reports"
    configs = project_root / "configs"
    _write_csv(reports / "finetune_session_voltage_provenance.csv", provenance, PROVENANCE_FIELDS)
    _write_csv(configs / "finetune_session_voltage_mapping.csv", mapping, MAPPING_FIELDS)
    _write_csv(reports / "finetune_group_completeness_after_mapping.csv", groups, GROUP_FIELDS)
    write_lsp_trace(reports / "LSP_SOURCE_TRACE.md")

    unresolved = [row for row in provenance if not row["current_canonical_voltage"]]
    known_rows = sum(int(row["row_count"]) for row in provenance) - sum(
        int(row["row_count"]) for row in unresolved
    )
    blockers = f"""# MEPI v1.1 Remaining Blockers

## Voltage provenance

The 12-session provenance table below is sorted by `core_id` and session timestamp. It contains 12 sessions, each with 75 rows, 15 frequencies and five repeat identifiers. Existing evidence supports canonical voltage for {len(provenance) - len(unresolved)} sessions ({known_rows}/900 rows); {len(unresolved)} sessions remain blank and require explicit experimenter confirmation. No measured `vin_rms_v` or reprocessed dataset row was modified.

{_markdown_table(provenance)}

The manual template is `configs/finetune_session_voltage_mapping.csv`. Evidence-supported exact-grid values are retained, but `confirmed_by_experimenter` remains `0` for every row because no human confirmation was fabricated. Unresolved canonical values remain empty.

Under currently available voltage evidence, the COMMERCIAL session `session_20260802_075027` has reference voltage 6.28 V and measured median 6.909851507406745 V. It cannot be automatically treated as the intended 5.6-V session. Current evidence result: `MISSING_NOMINAL_SESSION` for COMMERCIAL 5.6 V. Only a targeted 5.6-V remeasurement may be needed if the experimenter confirms that this 6.3-V session is genuinely distinct; no full-dataset repetition is requested.

## Electrical QC and group completeness

All 900 rows remain in the v2 master. The 38 deterministically confirmed `VOUT_THD > 3%` rows remain `HARD_FAIL` and are excluded only from valid/train-ready counts. They were not reprocessed or cleared again.

`reports/finetune_group_completeness_after_mapping.csv` contains all 90 intended condition groups and the requested valid/hard-fail counts. Because manual mapping is not yet authoritative, its counts use only the current evidence-supported canonical mappings; unresolved sessions contribute to no nominal group. No whole group was discarded because a repeat failed.

## Temperature

The verified physical mapping remains board field 1 = TempCore and field 2 = TempRoom, exposed as ambient = field 2 and core = field 1. Negative corrected rise values remain untouched. No independent reference-thermometer calibration was found or inferred.

## LSP

`reports/LSP_SOURCE_TRACE.md` records the focused source audit. The manuscript supplies the symbolic equations, but the project lacks authoritative numeric `Ea`, numeric `epsilon`, an executable train-only `Tref` rule, and exact normalization/scaling/bounds. Legacy learned parameters and unit-test fixtures were explicitly rejected as provenance.

## Split and execution boundary

No train/validation/test split was generated, normalization was not fitted, the test subset was not accessed, and no training was run.

FALSE explanations: absolute temperature calibration lacks independent reference evidence; voltage grouping lacks experimenter-confirmed provenance for six sessions and currently lacks a supported COMMERCIAL 5.6-V session; the LSP definition lacks the quantities/rules listed above; therefore LSP targets and training remain blocked.

B_READY = TRUE
TEMPERATURE_MAPPING_READY = TRUE
TEMPERATURE_CALIBRATION_READY = FALSE
VOLTAGE_GROUPING_READY = FALSE
ELECTRICAL_QC_READY = TRUE
LSP_DEFINITION_READY = FALSE
LSP_READY = FALSE
TRAIN_READY = FALSE
"""
    (reports / "MEPI_V1_1_REMAINING_BLOCKERS.md").write_text(blockers, encoding="utf-8")
    if master_path.read_bytes() != master_before:
        raise AssertionError("Audit unexpectedly modified the v2 master dataset")
    print(_terminal_table(provenance))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
