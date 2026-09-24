# LSP v1.2 Audit

## Frozen definition

`T_core_K = temperature_core_c + 273.15`; `log_LSP_raw = (125000.0 / 8.314462618) * (1/T_core_K - 1/298.15)`; `LSP_raw = exp(log_LSP_raw)`. Construction used float64 and no epsilon. LSP is a dimensionless relative thermal-stress proxy, not lifetime, RUL, time-to-failure, or service hours. `Ea_eff` is a literature-informed effective thermal-aging sensitivity parameter, not a dataset-identified or experimentally established material constant.

## Eligibility and temperature provenance

LSP was generated for 862 QC-valid primary rows only (445 FE, 417 COMMERCIAL). HARD_FAIL and supplementary rows retain blank LSP targets. The physically verified one-time channel swap was asserted for every source row; no temperature offset or slope was fitted.

## Numerical checks

- `LSP_raw` min/median/max: 0.302085964148 / 0.605125887514 / 0.776920969806
- `LSP_raw(T_ref=298.15 K)`: 1
- Strictly decreasing with increasing core temperature: PASS
- Finite and positive: PASS

## Split and normalization

Seed 42 assigned exactly 72/9/9 identities across all 90 measured operating-condition groups. QC-valid candidates occur in 89 groups; `F1000_V5p6_RL49p6025_SINE` has no QC-valid row but remains explicitly assigned in the frozen group manifest. Candidate row counts are train/validation/test = 687/85/90. Population mean/std (`numpy`, `ddof=0`) were fitted on 687 training rows only: `mu_LSP_train=0.60305503339713129`, `sigma_LSP_train=0.083751447010489311`. Validation, test, the Arrhenius reference, and the PIRL Arrhenius residual all use this same frozen scaler; no residual scaler exists. Split hash: `01d097ff8bd1d3d8e09d1ef6affd508e72bfc11b3e81ac74034862103870a166`.

No test result or test metric was used for model selection. No training was run.
