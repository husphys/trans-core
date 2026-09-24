# Pretraining throughput benchmark

Status: **BENCHMARK ONLY — NOT OFFICIAL EXPERIMENT EVIDENCE**

Hardware: NVIDIA GeForce RTX 3050, 8 GiB. All measured training steps used CUDA AMP,
the unchanged 1024-point MagNet waveform, the unchanged two operating inputs, the
unchanged target transform and loss, and material-routed temporary heads. The train
manifest contains 149,405 examples.

## Bottleneck profile

The bounded TCN profile used 5 warm-up steps and 100 measured steps at physical batch
48 with four persistent HDF5 workers. It achieved 916.8 samples/s (19.10 steps/s), with
mean times of 0.30 ms data loading, 0.21 ms host-to-device transfer, 19.85 ms forward,
28.56 ms backward, and 3.44 ms optimizer step. Mean sampled GPU utilization was 61.6%;
peak allocated/reserved memory was 0.34/0.41 GiB.

The original failed run logged about 4,000 steps in 2,650 seconds at batch 48, equivalent
to approximately 72.45 samples/s. The optimized common batch-256 TCN benchmark reached
1,432.28 samples/s, a 19.77x throughput increase.

## TCN batch-size sweep

All rows use AMP, four workers, pinned memory, persistent workers, prefetch factor 2,
and direct reads from the authoritative HDF5 file.

| Batch size | DataLoader configuration | Samples/s | Steps/s | Peak GPU allocated/reserved | Estimated min/epoch | Estimated 20 epochs |
|---:|---|---:|---:|---:|---:|---:|
| 64 | HDF5, workers=4, pin, persistent, prefetch=2 | 966.07 | 15.095 | 0.44 / 0.52 GiB | 2.58 | 0.86 h |
| 128 | HDF5, workers=4, pin, persistent, prefetch=2 | 1,242.43 | 9.706 | 0.83 / 0.95 GiB | 2.00 | 0.67 h |
| **256** | **HDF5, workers=4, pin, persistent, prefetch=2** | **1,432.28** | **5.595** | **1.61 / 1.74 GiB** | **1.74** | **0.58 h** |
| 512 | HDF5, workers=4, pin, persistent, prefetch=2 | 1,556.55 | 3.040 | 3.18 / 3.37 GiB | 1.60 | 0.53 h |

Estimates cover training steps only; validation, checkpoint I/O, notebook setup, and final
evaluation add overhead.

## Common-batch memory smoke across all backbones

Every row below is a real MagNet AMP forward/loss/backward/optimizer step at batch 256.
Every loss was finite and every batch routed all 10 material indices.

| Backbone | Samples/s (one-step smoke) | Peak allocated | Peak reserved | Estimated min/epoch | Estimated 20 epochs |
|---|---:|---:|---:|---:|---:|
| TCN | 1,351.22 | 1.62 GiB | 1.74 GiB | 1.84 | 0.61 h |
| LSTM | 1,467.27 | 1.67 GiB | 2.45 GiB | 1.70 | 0.57 h |
| BiLSTM | 1,643.00 | 2.01 GiB | 2.78 GiB | 1.52 | 0.51 h |
| LSTM-Attention | 1,300.98 | 2.02 GiB | 2.78 GiB | 1.91 | 0.64 h |
| GRU | 1,596.05 | 1.75 GiB | 2.64 GiB | 1.56 | 0.52 h |
| BiGRU | 1,490.70 | 2.16 GiB | 3.05 GiB | 1.67 | 0.56 h |
| RWKV | 849.25 | 3.51 GiB | 3.68 GiB | 2.93 | 0.98 h |
| xLSTM | 1,072.96 | 2.49 GiB | 2.63 GiB | 2.32 | 0.77 h |

Batch 512 technically passed all eight one-step smokes, but RWKV reached 6.96 GiB
allocated and 7.24 GiB reserved. That leaves too little operational headroom on an 8 GiB
GPU for a 20-epoch run, so it was not selected. Batch 256 leaves more than 4 GiB of
reserved-memory headroom in the worst case.

## DataLoader and RAM-cache sweep

At batch 512, HDF5 with four workers delivered 1,557.51 samples/s; eight workers delivered
1,547.78 samples/s, while two workers fell to 903.54 samples/s and exposed 239.71 ms mean
data wait. Optional RAM caching delivered 1,574.98, 1,571.00, and 1,585.56 samples/s for
2, 4, and 8 workers respectively. The small gain does not justify approximately 0.6 GiB
of resident waveform cache by default. RAM caching remains available through
`cache_mode: ram` if storage behavior changes.

## Recommended common configuration

- Physical/effective batch size: **256**, identical for all eight backbones; no gradient accumulation.
- DataLoader: **4 workers, pinned memory, persistent workers, prefetch factor 2, HDF5 mode**.
- Transfers: non-blocking when pinned memory is enabled.
- AMP: enabled.
- CUDA runtime: cuDNN benchmark enabled; deterministic algorithms disabled; TF32 enabled;
  float32 matmul precision `high`; fixed seed and all frozen manifests/fingerprints retained.

The selected values are recorded in `experiments/pretrain_v1/common_config.yaml`. The raw
machine-readable measurements are in `reports/benchmark_tcn_profile_hdf5.json` and
`reports/benchmark_pretraining_comprehensive.json`.
