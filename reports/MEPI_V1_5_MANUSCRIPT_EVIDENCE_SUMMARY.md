# MEPI v1.5 manuscript evidence summary

Status: **COMPLETE**

This document aggregates persisted artifacts only. No model was loaded, no dataset or loader was constructed, no inference or training was run, and no scientific result was recomputed or changed during aggregation.

## A. Dataset and split

The inherited MEPI v1.2 scientific dataset contains 900 primary measured rows, of which 862 are QC-valid. The frozen group split uses seed 42.

| Split | Groups | Rows |
|---|---:|---:|
| Train | 72 | 687 |
| Validation | 9 | 85 |
| Test | 9 | 90 |

The split manifest is `data/MEPI/v1_2/split_manifest_v1_2.csv` (SHA256 `01d097ff8bd1d3d8e09d1ef6affd508e72bfc11b3e81ac74034862103870a166`). Sources: `configs/finetune_v1_5.yaml` (SHA256 `6c10585593b786599f8f545f9550fdf96974ee13537663bc5f768943f5fdf8fb`) and `reports/MEPI_V1_3_XLSTM_TRAIN_READINESS.md` (SHA256 `62a9e962d6d9820e6159a019ae068cfd973af6005335397f2930bc4b748a7655`).

## B. Selected xLSTM architecture

The selected downstream backbone is xLSTM depth 8 with latent width 256. It pools by taking the sequence mean after the xLSTM backbone and uses `MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.0, batch_first=True)` plus residual `LayerNorm(256)` for fusion. The pretraining model has 5,969,930 trainable parameters.

Only `waveform_encoder` and `backbone` are transferred, with strict module loading. The source checkpoint is `experiments/xlstm_depth_v1/depth_8/best_checkpoint.pt` (SHA256 `fdbeb87932dd6e20c45104f2ff074e267b414587a5ce9e57e48afb828034055c`). Source: `reports/backbone_transfer_v1_3.json` (SHA256 `26066f81b9ede204bf4e52e9adbc23a4a93cbbb01bdabda666569820fb5844dd`).

## C. Downstream convergence

The v1.4 baseline reached its 50-epoch ceiling with best zero-based epoch 49 and best validation total loss `0.4636393350713393`. Its persisted convergence classification was `A_STILL_CLEARLY_IMPROVING_AT_EPOCH_49`, with absolute validation-loss improvements of `0.0403531095560859` over the last five epochs and `0.09840639619266284` over the last ten epochs. Source: `reports/MEPI_V1_4_BASELINE_VALIDATION_METRICS.json` (SHA256 `0c78748cd507b59770da1e814bc3959b2b919cdafaabfaeba08793d81ed135b3`).

The selected v1.5 configuration completed 100 epochs, with best zero-based epoch 91 and no early stop. On the 85-row validation subset it recorded `validation_L_electrical = 0.0504307309494299`, `validation_L_LSP = 0.1519513356335023`, `VAL_TASK_SCORE = 0.2023820665829322`, and weighted validation total loss `0.24970558706451865`. Source: `experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/l3_1_l4_0p05/completed.json` (SHA256 `88a69f07d5509988face4ebde48a85044b5758afb07bc9fc733e269d8be4ef8a`).

## D. Loss-weight sensitivity

Status: `GRID_COMPLETE_VALIDATION_ONLY`. All 12 predefined configurations completed. Selection minimized `VAL_TASK_SCORE = validation_L_electrical + validation_L_LSP`; weighted validation total loss was not used for cross-configuration ranking, and test data was not accessed for the grid. Source: `experiments/finetune_v1_5_xlstm_depth8/loss_weight_sensitivity/comparison.json` (SHA256 `908c872450e0811ec2e9cc8ca2e193925f9c302100e413e51d66f80d3419245f`).

## E. Selected lambda configuration

The unique validation-only minimum is frozen as:

```text
configuration_id = l3_1_l4_0p05
lambda1 = 1.0
lambda2 = 1.0
lambda3 = 1.0
lambda4 = 0.05
VAL_TASK_SCORE = 0.2023820665829322
selected checkpoint SHA256 = 0315922cab43aad2cde35016f1a60a62f3df4bc94844310e5ef84aee88389b33
```

Sources: `configs/final_model_v1_5.yaml` (SHA256 `4b18d275be0a56f09371f2fd91d3fe03bba72a7cee6ca732014fbbf043dcdf96`) and `experiments/finetune_v1_5_xlstm_depth8/final_model_manifest.json` (SHA256 `607bad9d532777f11a6659459cfa59e6a78e054bbd6ed95abe0e4f3dfffe67c1`).

## F. Final test metrics

Status: `FINAL_TEST_COMPLETE`. The frozen test contains 90 rows in 9 groups and was evaluated exactly once: `TEST_EVALUATION_COUNT = 1`. No numerical failure was recorded.

| Target | MAE | RMSE | R2 | MAPE (%) |
|---|---:|---:|---:|---:|
| efficiency | 1.1456656561957466 | 1.4833748448127801 | 0.9817636936882129 | — |
| P_loss | 0.013692542165517806 | 0.01645941692939988 | 0.9706166407086579 | — |
| LSP_raw | 0.027950685885217454 | 0.03636576957789921 | 0.9162509236930347 | 4.788047790527344 |

Sources: `reports/MEPI_V1_5_FINAL_TEST_RESULTS.json` (SHA256 `aa6178404deea9d48fdee436ba9ef6d033a569df76f84f9ad9e33b8e2ec7c1df`) and `reports/MEPI_V1_5_FINAL_TEST_RESULTS.md` (SHA256 `8254a367d8d7675f5eb8817603ed9364342f34140853f68c86c08c80cdd7ad90`).

## G. Computational profile

The persisted comprehensive benchmark is explicitly benchmark-only, not official scientific evidence. It used an NVIDIA GeForce RTX 3050. The final downstream model has 5,840,389 parameters. The benchmark reported a largest common passing batch size of 512.

| xLSTM HDF5 batch size | Samples/s | Peak allocated GiB | Status |
|---:|---:|---:|---|
| 64 | 596.1309277292313 | 0.6775360107421875 | PASS |
| 128 | 873.502899626077 | 1.2812995910644531 | PASS |
| 256 | 1072.9597094730682 | 2.4889488220214844 | PASS |
| 512 | 1023.6995032508113 | 4.905091762542725 | PASS |

Sources: `reports/benchmark_pretraining_comprehensive.json` (SHA256 `8ec9a636df0bb5e86d24617e74d4988b2cbb467764b0dc34987fdf04c8629904`) and the final-test ledger above.

## H. Frequency-screening summary

Status: `INFERENCE_COMPLETE`. Notebook 35’s saved output records `FREQUENCY_SCREENING_RUN = TRUE`, 150 source rows, 146 usable rows, 4 excluded existing `HARD_FAIL` rows, and 15 candidate frequencies. The persisted JSON contains 146 predictions and all 15 frequency summaries. This remains an offline proof-of-concept; it declares no optimum and used no test data or test tuning.

| frequency_hz | n_samples | mean_predicted_efficiency | mean_predicted_P_loss | mean_predicted_LSP | mean_predicted_LSP_std |
|---:|---:|---:|---:|---:|---:|
| 1000 | 6 | 65.51508967081706 | 0.1730078011751175 | 0.6288372874259949 | 0.029453453814962324 |
| 1250 | 10 | 61.682511520385745 | 0.20370426923036575 | 0.6159101665019989 | 0.030907100168335998 |
| 1500 | 10 | 62.17787055969238 | 0.19745796099305152 | 0.6292718648910522 | 0.029944665953685584 |
| 1750 | 10 | 64.65536155700684 | 0.16844293028116225 | 0.6774202525615692 | 0.030498467400059893 |
| 2000 | 10 | 63.819210815429685 | 0.17080627307295798 | 0.6073591768741607 | 0.028465600778636024 |
| 2250 | 10 | 66.58130493164063 | 0.14749449715018273 | 0.6734647572040557 | 0.029044074312544188 |
| 2500 | 10 | 67.96418647766113 | 0.143923844397068 | 0.6425775766372681 | 0.03047657276733661 |
| 2750 | 10 | 66.9046646118164 | 0.15181330367922782 | 0.6046069979667663 | 0.029257041904164355 |
| 3000 | 10 | 71.63535766601562 | 0.12013048306107521 | 0.6853793025016784 | 0.02834078030709012 |
| 3250 | 10 | 69.17736129760742 | 0.1303592123091221 | 0.6117972373962403 | 0.029686544266362875 |
| 3500 | 10 | 70.73247146606445 | 0.1273675911128521 | 0.6327846765518188 | 0.028320288604069238 |
| 3750 | 10 | 75.1265739440918 | 0.10749482214450837 | 0.675521731376648 | 0.03144868547391435 |
| 4000 | 10 | 69.90205459594726 | 0.12706748247146607 | 0.6012332141399384 | 0.030929472738715887 |
| 4250 | 10 | 74.91922073364258 | 0.10717667862772942 | 0.6621390879154205 | 0.030808515117277467 |
| 4500 | 10 | 74.26459579467773 | 0.10806235074996948 | 0.6375685453414917 | 0.02944638634973239 |

Sources: `reports/MEPI_V1_5_FREQUENCY_SCREENING.json` (SHA256 `5b287be4ad4d457898afcaed5deda3ebddef5022c9fa74964c35f7aa0dce6f5c`), `reports/MEPI_V1_5_FREQUENCY_SCREENING.md` (SHA256 `285b0a146f833f8e4334267a2b1cf1ac1553b8795829c5300f805e21f6a470e3`), `reports/MEPI_V1_5_FREQUENCY_SCREENING_PREDICTIONS.csv` (SHA256 `a10c52185ade8cf972b9518d00dc9bf24a5ae64659b892108ae7803250526210`), and `notebooks/35_xlstm_v1_5_frequency_screening.ipynb` (SHA256 `f18b11a4740fe6ac18206747d98eae5e173745496b78b8482229fb89daec2758`).

## Aggregation safety and manuscript readiness

```text
SCIENTIFIC_TRAINING_RUN = FALSE
FINAL_TEST_RERUN = FALSE
TEST_DATA_ACCESSED_DURING_AGGREGATION = FALSE
FREQUENCY_SCREENING_RERUN = FALSE
MANUSCRIPT_EVIDENCE_COMPLETE = TRUE
```

No additional scientific experiment is required before manuscript revision.
