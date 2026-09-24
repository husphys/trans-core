# MEPI v1.5 frequency-screening demonstration

Status: **INFERENCE_COMPLETE**

This is an offline proof-of-concept screening demonstration within the measured demo domain. It is not an automated hardware sweep, an independently validated operating-frequency recommendation, or a new optimization study. No single optimal frequency is declared.

## Provenance and isolation

```text
PROTOCOL_SHA256 = 0f68d6ebd5d16177e5238471639f861dfa91522130ac6a4a3ac48634b9af1e39
FINAL_MODEL_MANIFEST_SHA256 = 607bad9d532777f11a6659459cfa59e6a78e054bbd6ed95abe0e4f3dfffe67c1
SELECTED_CHECKPOINT_SHA256 = 0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33
SCREENING_INPUT_ARTIFACT = data/MEPI/demo_manifest_v2.csv
SCREENING_INPUT_SHA256 = 812ea7e87ad69c02702a411d0f1fe6ef241b9753586a50f1d38522d25d7f9556
SCREENING_WAVEFORM_BUNDLE_SHA256 = ce27eb62f45a6282ae8b49ceb7c9afd17082f7cf1eb04d8fe9683ebcc6d807d6
SCREENING_CONDITION_COUNT = 146
TRAINING_PERFORMED = FALSE
TEST_DATA_USED_FOR_SCREENING = FALSE
TEST_TUNING = FALSE
FINAL_TEST_RERUN = FALSE
TEST_EVALUATION_COUNT = 1
ANY_NAN = FALSE
ANY_INF = FALSE
```

The four source rows with existing `VOUT_THD` hard failures were excluded. Remaining source QC warnings are retained in the per-condition evidence and do not constitute target labels.

## Descriptive frequency-level prediction summary

| Frequency (Hz) | n | Mean predicted efficiency (%) | Mean predicted P_loss | Mean predicted LSP | Mean predicted LSP std |
|---:|---:|---:|---:|---:|---:|
| 1000 | 6 | 65.51509 | 0.1730078 | 0.62883729 | 0.029453454 |
| 1250 | 10 | 61.682512 | 0.20370427 | 0.61591017 | 0.0309071 |
| 1500 | 10 | 62.177871 | 0.19745796 | 0.62927186 | 0.029944666 |
| 1750 | 10 | 64.655362 | 0.16844293 | 0.67742025 | 0.030498467 |
| 2000 | 10 | 63.819211 | 0.17080627 | 0.60735918 | 0.028465601 |
| 2250 | 10 | 66.581305 | 0.1474945 | 0.67346476 | 0.029044074 |
| 2500 | 10 | 67.964186 | 0.14392384 | 0.64257758 | 0.030476573 |
| 2750 | 10 | 66.904665 | 0.1518133 | 0.604607 | 0.029257042 |
| 3000 | 10 | 71.635358 | 0.12013048 | 0.6853793 | 0.02834078 |
| 3250 | 10 | 69.177361 | 0.13035921 | 0.61179724 | 0.029686544 |
| 3500 | 10 | 70.732471 | 0.12736759 | 0.63278468 | 0.028320289 |
| 3750 | 10 | 75.126574 | 0.10749482 | 0.67552173 | 0.031448685 |
| 4000 | 10 | 69.902055 | 0.12706748 | 0.60123321 | 0.030929473 |
| 4250 | 10 | 74.919221 | 0.10717668 | 0.66213909 | 0.030808515 |
| 4500 | 10 | 74.264596 | 0.10806235 | 0.63756855 | 0.029446386 |

The table is descriptive only. It exposes the predicted tradeoff across the predefined measured frequencies without adding a weighted score, tuning a threshold, or declaring an optimum.
