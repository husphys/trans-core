# Manuscript Architecture Migration v1.3

## Scope and rule

`Manuscript_clean.docx` was inspected without modification. Paragraph and table
indices below are zero-based XML document-order locations used for traceability.
Any BiGRU-specific result is historical evidence only and is classified:

`LEGACY_RESULT_REQUIRES_RETRAINING`

It must not be relabeled, numerically transferred, or interpreted as an xLSTM
result. New xLSTM downstream training and validation must replace it. The frozen
test subset remains closed.

## Architecture statements requiring migration

| Location | Current BiGRU-dependent content | Required treatment |
|---|---|---|
| P8 | Abstract identifies BiGRU and reports final downstream metrics | Replace architecture only after xLSTM retraining; invalidate attached BiGRU metrics |
| P16 | Contribution statement names BiGRU as a supporting module | Replace with frozen xLSTM description |
| P20 | Claims five BiGRU depths and downstream results | Replace with new xLSTM downstream experiment evidence |
| P28 | Justifies BiGRU selection | Superseded by v1.3 validation-selected xLSTM evidence |
| P58 | Overall MEPI architecture names BiGRU | Replace architecture description |
| P72–P75 | BiGRU subsection, motivation and selection | Rewrite for the exact eight-block project xLSTM implementation |
| P88, P90, P92 | BiGRU pooling/output feeding PIRL | Retain only generic pooling/PIRL semantics; replace backbone attribution |
| P114 | Prediction heads described as operating on BiGRU representation | Replace backbone attribution |
| P139 | Downstream initialization says selected BiGRU | Replace with strict waveform-encoder plus xLSTM-backbone transfer |
| P140, P142 | Computational/evaluation scope tied to eight-layer BiGRU | New xLSTM evidence required |
| P162 | Old pretraining ranking and reason for retaining BiGRU | Conflicts with clean measured xLSTM selection; replace from current evidence |
| P166–P167 | Convergence and architecture conclusions centered on BiGRU | Re-evaluate from clean evidence; do not preserve superiority claims |
| P169–P173 | BiGRU downstream depth analysis and depth-8 selection | Entire result chain is legacy and requires xLSTM downstream rerun |
| P179–P182 | Final BiGRU configuration, parameters, checkpoint size and timing | Entire numerical profile is invalid for xLSTM |
| P190, P192, P194–P195 | Discussion justifying BiGRU and interpreting its depth results | Rewrite only after new xLSTM validation evidence |
| P207 | MEPI contribution summary attributes temporal modeling to BiGRU | Replace architecture attribution; re-evaluate associated results |
| P224–P225 | Conclusions identify eight-layer BiGRU and downstream metrics | Replace only after xLSTM fine-tuning; do not carry numbers forward |

## Algorithms, figures and tables

| Artifact | Current content | Status |
|---|---|---|
| Algorithm 2, table T3R1 | Initializes and forwards a `MEPI-BiGRU` model; uses non-strict loading | `LEGACY_RESULT_REQUIRES_RETRAINING`; replace with strict v1.3 transfer boundary |
| Figure 3, P60/P300 | Overall architecture may depict the BiGRU sequence module | Inspect/redraw as xLSTM before publication |
| Figure 5, P77/P302 | Explicit BiGRU architecture figure | Replace with the exact project `XLSTMBlock` stack |
| Table 3, P161/P308, especially T4R3 | Old backbone comparison and BiGRU row | Replace from clean measured pretraining comparison; do not reuse old selection wording |
| Figure 6, P165/P303 | Old MagNet curves | Regenerate only from current measured logs if used in revised manuscript |
| Table 4, P171/P309 and T5R1–T5R5 | BiGRU 2/4/6/8/10 downstream metrics | Entire table is `LEGACY_RESULT_REQUIRES_RETRAINING` |
| Table 5, P175/P310 and T6R1–T6R12 | Loss-weight results downstream of selected BiGRU | Entire table requires new xLSTM fine-tuning/sensitivity evidence |
| Table 6, P181/P311 and T7R2–T7R7 | 2,946,120 parameters; 11.78/11.82 MB; 34:58.865 pretraining; 7:42.294 fine-tuning; 42:41.159 combined | All values are BiGRU-specific and invalid for xLSTM |
| Figure 7, P186/P304 | Prototype inference interface driven by old selected model/results | Preserve only as historical UI evidence until xLSTM inference outputs exist |

## Numerical claims that must not migrate

- BiGRU depth metrics in Table 4, including the depth-8 values MAE 0.04552,
  RMSE 0.05973, R² 0.95775, LSP MAPE 6.89%, Err95 19.1%, and total loss 0.00629.
- Loss-weight sensitivity and selected downstream values in Table 5, including
  MAE 0.04529, RMSE 0.05903, R² 0.95873, LSP MAPE 6.83%, Err95 18.67%, and
  total loss 0.006236.
- BiGRU checkpoint profile: 2,946,120 trainable parameters, 11.78 MB FP32
  storage, 11.82 MB serialized checkpoint, and all reported training times.
- Any abstract, discussion, conclusion, figure, application, or deployment
  claim that attributes those measurements to the v1.3 xLSTM architecture.

## Preserved historical boundary

The manuscript file, old tables, figures, checkpoints, and logs are not deleted
or rewritten in this task. Their provenance remains useful for audit, but they
are not authoritative v1.3 scientific results.

MANUSCRIPT_BIGRU_RESULTS_LEGACY = TRUE
MANUSCRIPT_REWRITE_PERFORMED = FALSE
XLSTM_DOWNSTREAM_RESULTS_AVAILABLE = FALSE
TEST_SUBSET_ACCESSED = FALSE
