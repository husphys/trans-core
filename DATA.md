# MEPI data availability and layout

This release includes the derived data, deterministic split assignments, normalization artifacts, provenance manifests, and demo waveforms needed to inspect and reproduce the final MEPI v1.5 model workflow without republishing unnecessary duplicate raw archives.

## Included data

| Path | Contents | Size (bytes) |
|---|---|---:|
| `data/MEPI/v1_1/` | Complete checksum-defined v1.1-v4 bundle used by the v1.2 dataset builder, including master/paired/candidate tables and B(t) arrays | 18,741,903 |
| `data/MEPI/v1_2/` | Frozen QC-valid downstream dataset, B(t)_1024 array, sample IDs, exact group split, train-only normalization, schemas, and checksums | 8,816,784 |
| `data/MEPI/v1_3/` | Frozen architecture manifest and checksums | 3,902 |
| `data/splits/mepi_v1/` | Deterministic MagNet train/validation/test assignments | 20,938,500 |
| `data/MEPI/demo_manifest_v2.csv` | Frozen 150-row demo-screening manifest | 343,116 |
| `data/MEPI/demo_waveforms/` | 150 manifest-referenced processed B1024 CSV files in a portable repository layout | 9,149,463 |

The downstream MEPI design begins with 900 primary measured rows in 90 nominal operating-condition groups. After the frozen QC rules, 862 rows remain: 687 train rows in 72 groups, 85 validation rows in 9 groups, and 90 final-test rows in 9 groups.

The demo screening set contains 150 source rows. Four existing `HARD_FAIL` rows are excluded, leaving 146 usable conditions across 15 predefined frequencies. The demo manifest is preserved byte-for-byte, including its original acquisition paths, so its scientific hash remains unchanged. The public runtime first honors the original path and otherwise resolves the included portable copy under `data/MEPI/demo_waveforms/<session_id>/processed/`.

```text
demo manifest SHA256 = 812ea7e87ad69c02702a411d0f1fe6ef241b9753586a50f1d38522d25d7f9556
screening waveform bundle SHA256 = ce27eb62f45a6282ae8b49ceb7c9afd17082f7cf1eb04d8fe9683ebcc6d807d6
```

## External raw data

The following source datasets are intentionally not stored in Git or Git LFS:

| Expected path | Size (bytes) | SHA256 | Reason |
|---|---:|---|---|
| `data/raw/source_datasets/Dataset/magnet_pretrain_186k.h5` | 697,011,639 | `0aca43184e9bd6cc90eec736a89fe1ea32f45b126527cb97918cd92684ac2c19` | Third-party MagNet source dataset; avoid unverified redistribution and a very large duplicate of the externally obtainable source. |
| `data/raw/source_datasets/Dataset/finetune_dataset.csv` | 392,071,276 | `bece39a47e6688cd6b669cefcf51dd11707391d4b6289e5f5a6b94da1052584e` | Raw source export is very large and duplicated locally; the frozen QC-valid tables and waveform arrays used by v1.5 are included. |

The raw MEPI acquisition archive contained 4,530 files totaling 894,890,006 bytes. It includes original Scope #1/Scope #2 waveforms, session tables/configuration, and acquisition records. These files remain external because the final release already includes the exact derived model inputs, while the full raw archive is large and contains redundant acquisition-level material. Its complete file and SHA256 inventories are published under `archive_manifest/`.

The expected raw-session structure is:

```text
data/MEPI/raw/<session_id>/
├── samples.csv
├── config.json
├── raw/
│   ├── <sample_id>_scope1.csv
│   └── <sample_id>_scope2.csv
└── processed/
    └── <sample_id>_B1024.csv
```

Raw files are not required to load the included frozen v1.2 dataset or the two published checkpoints. They are required only to reconstruct the earliest acquisition-to-derived-data stage from instrument exports.

## Historical and redundant scientific artifacts not published

| Excluded path/class | Reason |
|---|---|
| `experiments/xlstm_depth_v1/depth_{2,4,6,10}/best_checkpoint.pt` | Non-selected depth-study checkpoints; the selected depth-8 checkpoint, comparison table, configuration, metrics, and summary are included. |
| `experiments/pretrain_v1/*/best_checkpoint.pt` | Redundant candidate pretraining checkpoints superseded for the final downstream path by the selected depth-8 representation checkpoint. |
| `experiments/finetune_v1_4_xlstm_depth8/` | Historical parent-run binaries superseded by the completed v1.5 continuation; exact parent hashes and resume-state preservation are retained in v1.5 provenance records. |
| `notebooks/32_xlstm_v1_4_finetune.ipynb` | Obsolete v1.4 execution notebook; the v1.4 protocol, source, configuration, validation evidence, and v1.5 continuation provenance remain public. |
| quarantined accidental 47-epoch run | Failed/quarantined run not used for any final MEPI v1.5 claim. |

These exclusions do not alter the final selected configuration, checkpoints, metrics, or frozen hashes.

## Integrity checks

From the repository root:

```bash
(cd data/MEPI/v1_1 && sha256sum -c checksums_v4.sha256)
(cd data/MEPI/v1_2 && sha256sum -c checksums_v1_2.sha256)
(cd data/MEPI/v1_3 && sha256sum -c checksums_v1_3.sha256)
```

See `archive_manifest/RAW_DATA_README.md` and the CSV/SHA256 manifests in that directory for acquisition-level provenance.
