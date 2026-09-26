# MEPI v1.5 ARMv7 deployment audit

## Decision

**ARMV7L DEPLOYMENT: NOT PRACTICALLY SUPPORTED**

Target evidence supplied for this audit is `uname -m = armv7l`, identifying a
32-bit ARM userland. The exact Raspberry Pi OS release, Python version, board
model, and memory capacity were not supplied. The deployment installer now
prints all three available architecture/OS/Python values before taking action.

No `MEPI_PI_ARMV7_DEPLOY` directory or ZIP is produced. Creating one would
misrepresent an unvalidated inference runtime as scientifically equivalent.
The existing `MEPI_PI_DEPLOY` remains the preferred 64-bit package and now
fails closed on `armv7l`.

## Exact runtime dependency audit

| Component | ARMv7 assessment | Source/install class |
|---|---|---|
| Python standard library and raw TCP/SCPI | Architecture-independent Python; no PyVISA | Raspberry Pi OS |
| Tkinter | Available for Debian/Raspberry Pi OS `armhf` | `apt install python3-tk` |
| NumPy | Available for `armhf` | `apt install python3-numpy` |
| Matplotlib | Available for `armhf` | `apt install python3-matplotlib` |
| PyYAML | Available for `armhf` | `apt install python3-yaml` |
| SciPy | Not imported by the packaged runtime | Not required |
| scikit-learn | Not imported by the packaged runtime; frozen scalers are JSON | Not required |
| PyTorch | No official Linux `armv7l` wheel and no Debian `armhf` package | Blocking dependency |
| ONNX Runtime | No official PyPI `armv7l` wheel; ARM32 requires a source/cross-build | Unvalidated alternative |

The exact frozen model is not a small generic LSTM. Its inference path requires
PyTorch checkpoint deserialization and strict state loading plus `Conv1d`,
`GELU`, `MaxPool1d`, `torch.fft.rfft`, adaptive average pooling,
`MultiheadAttention`, `LayerNorm`, linear layers, eight `nn.LSTM` blocks,
`tanh`, `softplus`, tensor power, and CPU tensor operations. The selected and
transfer checkpoints remain byte-identical and are not converted or modified.

## Option A: native PyTorch ARM32

Rejected. The official PyTorch package indexes publish Linux wheels for
`x86_64` and `aarch64`, not `armv7l`. `pip install torch` therefore has no
reproducible official ARM32 wheel matching the package contract. Arbitrary
third-party Raspberry Pi wheels were deliberately excluded because provenance,
operation coverage, Python/ABI compatibility, and frozen-model equivalence
could not be established.

Building current PyTorch from source on the target would be a bespoke toolchain
project, not a reliable no-virtual-environment installation path. It was not
accepted without an actual target build and the mandatory scientific and
performance validation.

## Option B: Raspberry Pi OS/Debian package

Rejected. Debian Bookworm publishes `python3-torch` 1.13.1 only for `amd64`,
`arm64`, `ppc64el`, and `s390x`. Debian Trixie publishes 2.6.0 for `amd64`,
`arm64`, `ppc64el`, `riscv64`, and `s390x`. Neither provides `armhf`. Bookworm's
version is also below the deployment's validated PyTorch 2.5 runtime family.

## Option C: equivalent inference runtime

Not accepted. A bounded export probe used the unchanged frozen checkpoint and
PyTorch 2.5.1 on CPU:

1. The legacy ONNX opset-17 export stopped at the shape-derived
   `adaptive_avg_pool1d` output size.
2. A diagnostic-only export probe replacing that shape expression with its
   mathematically identical fixed value of 256 progressed further, then failed
   because `aten::fft_rfft` is unsupported by that exporter/opset.

No repository or deployment code was changed for either probe. Re-expressing
the FFT or model graph would require a new deployment representation and a
carefully controlled converter. Current ONNX Runtime PyPI releases provide
Linux wheels for `x86_64` and `aarch64`, not `armv7l`. Microsoft's repository
contains an ARM32 source-build Dockerfile, so a custom cross-build is
theoretically possible, but it is not a reproducible prebuilt package for the
reported Pi.

Most importantly, no ARM32 runtime was available on the actual target for the
required multi-record equivalence checks. Therefore no alternative backend can
be called scientifically equivalent.

## Scientific equivalence and performance

ARM32 scientific equivalence: **NOT RUN / NOT CLAIMED**. There is no accepted
ARM32 inference backend. The required comparison of upstream features, all
1024 B(t) values, four predictions, and domain status across multiple replay
records cannot be performed without one. This is a blocker, not a tolerance
waiver; `rtol=1e-7` and `atol=1e-9` remain unchanged.

ARM32 performance: **NOT BENCHMARKED**. Model-load time, inference time, replay
time, memory use, and suitability of the two-second update interval are not
claimed. No scientific processing or GUI interval was changed.

## Frozen-contract verification

This audit did not retrain, fine-tune, select, quantize, simplify, reduce, or
approximately reimplement the model. It did not change the checkpoint, model
weights, eight-layer xLSTM, preprocessing, nine features, train-only scalers,
B(t) reconstruction, 1024-point representation, targets, domain metadata, or
splits. Tkinter, replay, offline screening, transformer profiles, background
worker behavior, and raw port-5025 Keysight SCPI code remain unchanged.

## Required migration

First identify the board:

```bash
cat /proc/device-tree/model; echo
uname -m
getconf LONG_BIT
```

Raspberry Pi 3, 4, 5, 400, and Zero 2 have 64-bit-capable CPUs even when the
current OS reports `armv7l`. For those boards:

1. Back up data and configuration from the current SD card.
2. On another computer, open Raspberry Pi Imager and select the exact board.
3. Select **Raspberry Pi OS (64-bit)** with a desktop, because the Tkinter GUI
   requires a graphical session.
4. Write and boot the new SD card, then update the OS.
5. Confirm `uname -m` reports `aarch64`.
6. Copy only `MEPI_PI_DEPLOY` and follow its `README_PI.md`.

If the board is an original Raspberry Pi, original Pi Zero, or a Raspberry Pi 2
revision with a 32-bit-only CPU, use a 64-bit-capable Raspberry Pi instead. Do
not bypass the architecture guard or install an unverified wheel.

## Primary packaging and platform sources

- PyTorch release files: <https://pypi.org/project/torch/>
- Official PyTorch wheel index: <https://download.pytorch.org/whl/torch/>
- Debian Bookworm `python3-torch` architectures:
  <https://packages.debian.org/bookworm/python3-torch>
- Debian Trixie `python3-torch` architectures:
  <https://packages.debian.org/trixie/python3-torch>
- Debian Bookworm ARMHF packages for NumPy, Matplotlib, Tkinter, and PyYAML:
  <https://packages.debian.org/bookworm/armhf/python3-numpy>,
  <https://packages.debian.org/bookworm/armhf/python3-matplotlib>,
  <https://packages.debian.org/bookworm/armhf/python3-tk>, and
  <https://packages.debian.org/bookworm/armhf/python3-yaml>
- ONNX Runtime release files and ARM32 source-build instructions:
  <https://pypi.org/project/onnxruntime/> and <https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/README.md>
- Raspberry Pi OS architecture guidance: <https://www.raspberrypi.com/documentation/computers/os.html>
