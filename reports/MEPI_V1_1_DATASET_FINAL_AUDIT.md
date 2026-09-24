# MEPI v1.1 Dataset Final Audit

## 1. Raw inventory

- Total: 835387494 bytes in 4231 files; 14 sessions.
- FE finetune/demo sessions: 6/1.
- COMMERCIAL finetune/demo sessions: 6/1.

## 2. Grid completeness

- Expected and observed finetune rows: 900/900.
- Frozen expected groups complete under exact stored voltage labels: 30/90.
- The observed session labels contain 3.2, 4.6, 5.1, 5.7, and 6.3 V outside the frozen grid; no relabeling was inferred. See `grid_completeness.csv` for every missing/extra condition.

## 3. B(t) correction

- Old method: legacy acquisition integrated Scope #2 CH2 Vout with `N_SECONDARY`.
- Correct method: raw Scope #2 CH1 primary Vin integrated with Np=10, Ae=0.0001217268 m^2.
- Reprocessed: 1050 samples; raw primary waveform available: yes for all.
- Exact feature differences are in `B_reprocessing_audit.md/json`.

## 4. QC

- PASS: 0; WARNING: 862; HARD_FAIL: 38.
- Hard-fail reasons: {'VOUT_THD': 38}.
- Warning reasons: {'VIN_REFERENCE_DRIFT': 793, 'VIN_SCOPE1_SCOPE2_MISMATCH': 900, 'UNEXPECTED_VOLTAGE_LEVEL': 450}.
- Per-core/voltage/frequency counts and reasons are explicit in `qc_breakdown.csv`.

## 5. Electrical sanity

Recomputed from raw synchronized waveforms and authoritative Keithley shunt voltage. Summary: `{"P_loss": {"max": 0.8244022551874648, "mean": 0.1849673121787276, "min": 0.03874967127295659, "sd": 0.14156006790347553}, "Pcu_w": {"max": 0.04614471010500114, "mean": 0.003630102413628088, "min": 0.0008567630488500672, "sd": 0.004364314377277044}, "efficiency_percent": {"max": 80.89193836189266, "mean": 68.0313846726533, "min": 42.09433059775269, "sd": 10.19344749741595}, "iin_rms_a": {"max": 0.6635451623548047, "mean": 0.1503750479929309, "min": 0.07193004025871172, "sd": 0.0740307125714241}, "input_vi_phase_deg": {"max": 67.79379505412304, "mean": 24.602736498395913, "min": 6.466865497511396, "sd": 15.240184434848356}, "phase_shift_deg": {"max": -0.08832950937094938, "mean": -1.6608366413462332, "min": -9.255166114330665, "sd": 1.4779779087433975}, "pin_w": {"max": 1.626994597732325, "mean": 0.5650061686727554, "min": 0.19855621302325588, "sd": 0.3188565176151813}, "pout_w": {"max": 0.8807664424925206, "mean": 0.3764087540803996, "min": 0.14070133803742557, "sd": 0.19794245648650916}, "vin_rms_v": {"max": 7.077425634127023, "mean": 4.436109146853434, "min": 2.8385613853230676, "sd": 1.173831915117397}, "vout_rms_v": {"max": 6.60970630691979, "mean": 4.1781589701215776, "min": 2.6418058444937627, "sd": 1.1023397909365018}}`.

The legacy acquisition implementation numerically divides Keithley voltage by `R_SHUNT_OHM = 1.5152`, but its banner/session `iin_dataset_rule` text incorrectly says `/ 0.2508`; the source README also says 0.25 ohm. These retained source files were not rewritten. All searched occurrences and their ACTIVE/HISTORICAL/COMMENT/NEGATIVE TEST/LEGACY DATA classification are in `legacy_inconsistency_audit.csv`.

## 6. Temperature sanity

Ambient/core/rise summary: `{"temp_rise_c": {"max": 0.5, "mean": -1.0444444444444445, "min": -2.75, "sd": 0.6317932018135893}, "temperature_ambient_c": {"max": 29.25, "mean": 27.84722222222222, "min": 26.5, "sd": 0.5397111817292644}, "temperature_core_c": {"max": 28.75, "mean": 26.802777777777777, "min": 25.0, "sd": 0.9694975680962565}}`. There are 852 negative core-minus-ambient readings; session min/max/mean/SD, consecutive jumps and time trends are in `temperature_session_audit.csv`, with core/voltage summaries in `temperature_audit.json`. Temperature quality was not used alone to delete electrically valid samples.

## 7. Train-ready dataset

- Master rows: 900 (FE 450, COMMERCIAL 450).
- Train-ready rows: 0; B1024 shape: (0, 1024); tabular schema: exactly 9 frozen features.
- Target completeness: efficiency 900/900, P_loss 900/900, LSP 0/900.

## 8. Split

No train/validation/test assignment was frozen because cleaning did not yield target-complete rows and the voltage-label grid is unresolved. Test rows were not accessed for model development. Normalization was not fitted.

## 9. Demo

Two demo sessions / 150 rows are isolated in `data/MEPI/demo_manifest.csv`; zero demo rows entered train-ready data.

## 10. Git integration

Added processing code, tests, schemas, reports, manifests, checksums, and compact metadata. Raw waveforms, preview bulk, legacy processed bulk, and archives remain excluded. Git CLI status is unavailable in this workspace because `.git/HEAD` and `.git/config` are not exposed.

## 11. Raw archive

`archive_manifest/` contains a README, file manifest, and SHA-256 list for raw/supporting source files. Regenerable previews and legacy processed B are not required for scientific reconstruction.

## 12. Blockers

1. Exact LSP Arrhenius formulation/constants remain unfrozen/unimplemented; all 900 rows retain `lsp_pending=1`.
2. Stored voltage group labels do not match the frozen six-level cross-core grid. An evidence-backed mapping/correction is required; this build does not infer one.
3. Core-minus-ambient temperature is negative in 852/900 rows; resolve sensor identity/calibration/provenance before using core temperature to derive LSP.

## 13. Final verdict

`TRAIN_READY = FALSE`

Raw data are sufficient for deterministic B/electrical reconstruction, but the required three-target dataset and frozen 90-group split cannot yet be produced without resolving all listed blockers.
