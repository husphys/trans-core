# LSP source trace

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
