# MEPI v1.5 loss-weight sensitivity readiness

## OBSERVED

- The frozen grid contains 12 configurations: `lambda3 = [0.1, 0.3, 0.5, 1.0]`, `lambda4 = [0.05, 0.10, 0.20]`, and `lambda1 = lambda2 = 1.0`.
- The byte-identical selection-rule artifact has SHA-256 `8c4eda9035e7e5e4f75a4b0ccde643c8946fd3b990575ec0323977cb8497a5be`.
- Primary cross-configuration metric: `VAL_TASK_SCORE = validation_L_electrical + validation_L_LSP`; direction: minimize.
- `weighted_validation_total_loss` is reported but prohibited for cross-configuration ranking.
- Configurations with any required numerical failure are excluded. Exact primary-score ties remain tied without an invented secondary rule.
- Every new run uses seed 42 and representation checkpoint SHA-256 `fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c`.
- The completed `(0.3, 0.20)` baseline passed the identical initialization/training-contract audit. Its validation-only `VAL_TASK_SCORE` is `0.25775656331987945`.
- The reusable baseline leaves 11 new configurations. `RUN_GRID = False`; none of those 11 runs was launched.
- Test remains inaccessible. Protocol v1.5 scientific training settings were not changed.

## INTERPRETATION

Notebook 33 is ready for deliberate execution of the 11 remaining configurations. Once all 12 eligible result rows exist, selection uses only the minimum comparable `VAL_TASK_SCORE`.

```text
LOSS_WEIGHT_SELECTION_RULE_READY = TRUE
PRIMARY_SELECTION_METRIC = VAL_TASK_SCORE
BASELINE_REUSE_VALID = TRUE
GRID_TOTAL = 12
NEW_RUNS_REQUIRED = 11
RUN_GRID = FALSE
TEST_ACCESSED = FALSE
```
