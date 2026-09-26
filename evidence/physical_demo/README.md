# MEPI Physical Demonstration Evidence

## Purpose

This directory preserves implementation and physical-demonstration evidence for reproducibility, audit, and possible future Supporting Information or Supporting Materials. Inclusion here establishes provenance for the archived items; it does not by itself establish a quantitative scientific claim.

The original source was the user-supplied Windows folder `E:\MEPI`, mounted read-only for inspection at `/mnt/e/MEPI`. Originals were not renamed, recompressed, resized, converted, or edited. The repository copy of `IMG_1668.JPG` retains the original bytes and EXIF metadata.

## Physical deployment

The documented deployment uses CPU-only inference on a Raspberry Pi with a touch-oriented MEPI GUI and a Keysight EDUX1052A reached over LAN/Telnet SCPI at `192.168.2.149:5024`. The configured acquisition roles are CH1 = primary Vin and CH2 = secondary Vout. The previously observed physical identity was `KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625`.

The user reports that the GUI subsequently completed live acquisition and MEPI prediction. The archived photograph visibly establishes a benchtop prototype containing a Keysight EDUX1052A displaying two traces, measurement/power hardware and wiring, and a separate screen displaying the MEPI interface. It does not, by itself, establish calibration, signal routing that is not visible, prediction accuracy, or performance guarantees.

Evidence chain:

```text
physical MEPI prototype
  -> Raspberry Pi implementation
  -> Keysight EDUX1052A acquisition
  -> canonical as-demonstrated deployment source
  -> photograph and public demonstration references
  -> cryptographic provenance in manifest.json
```

## Archived software

The two final Python files supplied at the root of `E:\MEPI` were inspected by imports, classes, and functions. They map directly to canonical application modules. They are not duplicated under `evidence/`.

| Artifact | Role | SHA-256 | Relationship to repository implementation |
|---|---|---|---|
| `keysight_lan.py` | As-demonstrated Keysight Telnet/SCPI transport | `652cd4b4013f6d6de3ac2997036e29a5fe9461ddbbb5b0d8203fc1189ab99318` | Canonical source: `apps/mepi_monitor/acquisition/keysight_lan.py`; generated deployment: `deploy/MEPI_PI_DEPLOY/apps/mepi_monitor/acquisition/keysight_lan.py` |
| `main_window.py` | As-demonstrated Raspberry Pi touch-oriented Tk GUI | `7683a9baeb3dff4cb1a8ddc481c6f36dd5cec85a8091bb0f1f3ca69b4379a70d` | Canonical source: `apps/mepi_monitor/ui_tk/main_window.py`; generated deployment: `deploy/MEPI_PI_DEPLOY/apps/mepi_monitor/ui_tk/main_window.py` |

## Physical photographs

| Artifact | Category | What it visibly establishes |
|---|---|---|
| [`IMG_1668.JPG`](images/measurement_setup/IMG_1668.JPG) | Complete measurement setup with GUI visible | A benchtop prototype, Keysight EDUX1052A with two displayed waveforms, additional measurement/power hardware, transformer/electronics, wiring, and a screen displaying the MEPI interface. Connections and numerical validity not legible or independently verified are not inferred. |

`IMG_1668.JPG` is 3,222,968 bytes, 4032 × 3024 pixels, SHA-256 `ff20afa211875c8615d8189afd568bb4a216336bcb29d61ae5f910c597f7a868`. Preserved EXIF identifies an Apple iPhone 12 Pro Max and `DateTimeOriginal=2026:09:26 18:25:27`, offset `+07:00`.

No separate GUI-only or live-prediction-only image was present in `E:\MEPI`; the single photograph contains the physical setup and a visible GUI screen.

## Video demonstrations

| Video | URL | Demonstrated function | Verification status |
|---|---|---|---|
| `Software_interface` | <https://youtu.be/T-xyvZ_ODOQ> | User-described software-interface/GUI demonstration | Public title and availability verified through YouTube oEmbed on 2026-09-26 UTC; video content not independently reviewed in this environment |
| `prototype test` | <https://youtu.be/ZZc24mB8yK4> | User-described physical prototype operation | Public title and availability verified through YouTube oEmbed on 2026-09-26 UTC; video content not independently reviewed in this environment |

See [`videos/README.md`](videos/README.md) for the external-evidence record and limitations. The videos are referenced, not downloaded or re-uploaded.

## Source-folder inventory and exclusions

The root of `E:\MEPI` contained seven files plus `MEPI_DATA/`:

| Source item | Size | SHA-256 / inventory | Disposition |
|---|---:|---|---|
| `IMG_1668.JPG` | 3,222,968 bytes | `ff20afa211875c8615d8189afd568bb4a216336bcb29d61ae5f910c597f7a868` | Archived unchanged as physical evidence |
| `keysight_lan.py` | 13,761 bytes | `652cd4b4013f6d6de3ac2997036e29a5fe9461ddbbb5b0d8203fc1189ab99318` | Integrated into canonical and generated deployment source |
| `main_window.py` | 27,329 bytes | `7683a9baeb3dff4cb1a8ddc481c6f36dd5cec85a8091bb0f1f3ca69b4379a70d` | Integrated into canonical and generated deployment source |
| `README.txt` | 1,012 bytes | `627969c05dd95e4f0c5e91e0ed80d1628a36bededa7f3244359519d9ec7d2a04` | Not archived: describes a distinct Windows v4.1 acquisition bundle |
| `mepi_acquisition_config.json` | 859 bytes | `bd4bc95c3b879e6483a7be2cb07f8c1fe6d9f235fb72c33adde8153865802abb` | Not archived: configuration for the distinct Windows acquisition workflow |
| `mepi_data_acquisition.py` | 83,198 bytes | `d32256dc0f63c385afe5a1a658cf2011cab0cf7b2159820b56b3e894d6fef6c6` | Not integrated: Windows Scope1/Scope2/Keithley/GPIB dataset acquisition program, not a Pi deployment module |
| `read_temp.ino` | 921 bytes | `f1fa2c1e91dc3309fd512f832ab0f115944d24b31c33a2c49383351ac6f7615a` | Not archived: separate temperature-board firmware, not demonstrated as a Pi runtime module |
| `MEPI_DATA/` | 894,890,006 bytes | 4,530 files: 3,390 CSV, 1,125 SVG, 15 JSON | Not archived or copied into `data/MEPI`; it is a large measurement-data tree outside this physical-demo evidence scope |

These exclusions prevent accidental duplication of datasets and historical acquisition software and do not alter the originals in `E:\MEPI`.

## Evidence-to-claim boundary

Supported when tied to the specific artifact and its verification status:

- a physical MEPI prototype and Raspberry Pi-oriented interface exist;
- a Keysight EDUX1052A and two displayed waveforms are visible in the physical setup;
- the canonical repository contains the user-identified as-demonstrated Pi transport and GUI source;
- the user reports successful live acquisition and MEPI prediction on the deployed system;
- public demonstration references were available at the audit date.

Not established by the photograph, user report, or demonstration videos alone:

- model accuracy or generalization beyond evaluated data;
- hard real-time guarantees or benchmark performance;
- lifetime/RUL validity;
- calibrated predictive uncertainty;
- superiority over other methods;
- measurement traceability beyond separately documented calibration evidence;
- long-term field reliability;
- independent validation of every displayed numerical value.
