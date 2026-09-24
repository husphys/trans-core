# MagNet dataset audit

- Status: **PASS**
- Dataset SHA-256: `0aca43184e9bd6cc90eec736a89fe1ea32f45b126527cb97918cd92684ac2c19`
- Total samples: 186,757
- B waveform shape: `(186757, 1024)`
- Duplicate sample IDs: 0
- Exact duplicate waveforms: 0
- Split seed: 42

## Samples per material

| Material | Samples |
|---|---:|
| 3C90 | 40,713 |
| 3C94 | 40,068 |
| 3E6 | 6,996 |
| 3F4 | 6,564 |
| 77 | 11,444 |
| 78 | 11,380 |
| N27 | 11,396 |
| N30 | 8,978 |
| N49 | 8,602 |
| N87 | 40,616 |

## Exact split counts

| Split | Samples |
|---|---:|
| train | 149,405 |
| validation | 18,676 |
| test | 18,676 |

Normalization is not fitted by this audit. The pretraining runner fits every scaler from the training manifest only and does not load the test manifest.
