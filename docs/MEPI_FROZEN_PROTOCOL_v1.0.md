# MEPI-FROZEN-PROTOCOL v1.0

**Status:** FROZEN  
**Date:** 2026-09-18  
**Purpose:** Single source of truth for MEPI manuscript revision, code implementation, experiment reruns, fine-tuning data acquisition, and deployment.

## 1. Overall research pipeline

MEPI follows a transfer-learning workflow:

MagNet public dataset  
→ representation pretraining  
→ backbone screening  
→ transfer pretrained waveform encoder and selected backbone  
→ fine-tuning on the experimentally measured transformer dataset  
→ full MEPI with PIRL and MTPH  
→ evaluation  
→ edge/offline deployment.

No later manuscript or code revision may introduce a contradictory pipeline unless this protocol is explicitly unfrozen.

---

## 2. Pretraining dataset

Use the MagNet training dataset containing approximately 186,757 samples from 10 magnetic materials.

### Pretraining model inputs

Each sample uses:

- Flux-density waveform \(B(t)\), standardized to 1024 samples.
- Excitation frequency \(f\).
- Temperature \(T\).

Therefore:

\[
X_B=B(t)_{1024}
\]

and

\[
X_{tab}^{pre}=[f,T]
\]

### Pretraining target

The regression target is volumetric/core loss:

\[
y=P_v
\]

The target may be log-transformed and normalized according to the implemented training protocol.

### Material identity

Material/Core ID is **not a model input feature**.

Material identity may only be retained as dataset metadata or used internally by the training pipeline if required for material-specific temporary loss heads.

### Pretraining architecture

The pretraining stage contains:

- waveform encoder;
- small operating-condition encoder for \(f,T\);
- fusion stage;
- candidate sequence backbone;
- temporary core-loss regression head.

Candidate backbones remain those already investigated in the manuscript:

- TCN;
- LSTM;
- BiLSTM;
- LSTM-Attention;
- GRU;
- BiGRU;
- RWKV;
- xLSTM.

PIRL and MTPH are **not used during MagNet pretraining**.

The purpose of pretraining is:

1. generic magnetic waveform representation learning;
2. backbone screening;
3. initialization of the downstream MEPI model.

After pretraining, temporary MagNet-specific prediction heads are discarded.

The transferred components are:

- pretrained waveform encoder;
- selected sequence backbone.

The MagNet-specific tabular encoder does not need to be transferred because the fine-tuning tabular feature space differs from the MagNet feature space.

---

## 3. Experimental fine-tuning objects

Only two physical transformer-core specimens are used:

1. Fe-based nanocrystalline core fabricated by the research group;
2. commercial reference core.

The two cores have identical physical geometry.

Core dimensions are therefore experimental constants and are **not model input features**.

`core_type` is retained only as metadata for grouping, analysis and reporting. It is not a predictive model input.

---

## 4. Fine-tuning excitation protocol

Use sinusoidal excitation only.

Use the same electrical operating grid for both cores.

The principal controlled variables are:

- excitation frequency;
- input voltage level.

The load is fixed at:

\[
R_L=50\ \Omega
\]

unless the experimental protocol is explicitly unfrozen and revised before acquisition.

Each operating condition is repeated five times for each core.

An operating-condition group is defined by:

\[
(f_{set},V_{in,set},R_L,\text{waveform})
\]

Both cores and all repeated measurements belonging to the same operating condition must remain in the same train, validation or test subset.

---

## 5. Raw experimental measurements

The acquisition system uses:

- digital oscilloscope;
- Keithley instruments for voltage/current measurement;
- core-temperature sensor;
- ambient-temperature sensor.

For each measurement sample, save the raw quantities required to reconstruct the targets and derived features.

Required measured or recorded quantities include:

- frequency;
- input voltage;
- input current;
- output voltage;
- output current;
- phase-related measurement where available;
- raw voltage waveform required to derive \(B(t)\);
- core temperature;
- ambient temperature.

Raw waveform files must be retained and must never be overwritten by processed/resampled signals.

---

## 6. Fine-tuning waveform input

The principal waveform modality is:

\[
B(t)
\]

not a categorical waveform label.

Derive \(B(t)\) from the measured winding voltage using:

\[
B(t)=\frac{1}{N_sA_e}\int v_s(t)\,dt
\]

Processing sequence:

raw waveform  
→ offset/drift correction  
→ integration  
→ complete-cycle extraction  
→ resampling  
→ 1024-point \(B(t)\).

Final waveform model input:

\[
X_B^{fine}=B(t)_{1024}
\]

---

## 7. Fine-tuning tabular model inputs

The frozen fine-tuning tabular input contains **nine features**:

1. `frequency_hz`
2. `vin_rms_v`
3. `phase_shift_deg`
4. `temperature_ambient_c`
5. `B_peak_t`
6. `B_rms`
7. `B_thd_percent`
8. `dBdt_max`
9. `form_factor`

Therefore:

\[
X_{tab}^{fine}
=
[
f,
V_{in,rms},
\phi,
T_{ambient},
B_{peak},
B_{rms},
B_{THD},
(dB/dt)_{max},
FF
]
\]

These nine features replace previous inconsistent claims of 12 or 22 model-input features.

No later manuscript section may claim a 22-dimensional MEPI tabular input unless this protocol is explicitly unfrozen.

---

## 8. Quantities retained but excluded from model input

The following quantities are retained in the dataset but are not predictive inputs:

- `core_type`
- `waveform_type`
- `temperature_core_c`
- `iin_rms_a`
- `vout_rms_v`
- `iout_rms_a`
- `vin_peak_v`
- `temp_rise_c`
- `input_power_w`
- `output_power_w`
- `Pcu_w`
- `P_loss`
- `efficiency_percent`
- `LSP`

They are used for metadata, quality control, target construction or reporting.

Particularly important:

`temperature_core_c` is not used as a direct model input because the current LSP target is constructed from core temperature. Using the same temperature directly as an input would create an undesirable shortcut for LSP prediction.

---

## 9. Fine-tuning targets

MEPI predicts three downstream quantities:

\[
\hat{\eta},\qquad
\hat{P}_{loss},\qquad
\widehat{LSP}
\]

### Efficiency

\[
\eta=
100\frac{P_{out}}{P_{in}}
\]

### Copper loss

\[
P_{cu}
=
I_{in,rms}^{2}R_p+
I_{out,rms}^{2}R_s
\]

### Loss-related target

\[
P_{loss}
=
P_{in}-P_{out}-P_{cu}
\]

### LSP

LSP is derived from the measured core temperature using the Arrhenius-informed formulation defined in the manuscript.

The direct target quantities and quantities used to calculate them must not be included as predictive input features.

---

## 10. Fine-tuning MEPI architecture

Fine-tuning uses:

\[
B(t)_{1024}
+
9\text{-dimensional tabular vector}
\]

Processing:

\(B(t)\)  
→ pretrained waveform encoder

Nine tabular features  
→ newly initialized fine-tuning tabular encoder

Both representations  
→ fusion  
→ pretrained selected backbone  
→ PIRL  
→ MTPH  
→ efficiency, loss-related output and LSP.

The full 9-feature fine-tuning tabular encoder is not claimed to have been pretrained on MagNet.

---

## 11. Dataset partition

Use three subsets:

\[
80\%\ train + 10\%\ validation + 10\%\ test
\]

Partition at the operating-condition-group level.

All five repetitions for both cores belonging to one operating condition must be assigned to the same subset.

No repeated operating condition may cross train, validation or test.

Normalization and preprocessing parameters are fitted using the training subset only.

Validation is used for:

- checkpoint selection;
- architecture selection;
- backbone depth selection;
- loss-weight selection;
- other model-development decisions.

The test set is reserved for the final evaluation only.

Do not use the term `held-out` in the manuscript. Use only:

- training set;
- validation set;
- test set.

---

## 12. Manuscript consistency rules

Every manuscript revision must comply with this protocol.

Specifically:

- MagNet pretraining input = \(B(t)+f+T\).
- Material ID is not a model input.
- Full 9-feature MEPI tabular encoder belongs to fine-tuning.
- Fine-tuning waveform modality = \(B(t)\).
- Fine-tuning tabular dimensionality = 9.
- Fine-tuning outputs = efficiency, loss-related quantity and LSP.
- Two experimental cores only.
- Same physical geometry.
- Sinusoidal excitation.
- Fixed 50-ohm load under the frozen protocol.
- Five repeats per operating condition per core.
- Group-wise 80/10/10 train/validation/test split.
- No target leakage.
- No claim of 22 model input features.
- No claim that the full downstream MIE was identically pretrained on MagNet.
- No `held-out` terminology.

If old text, code or figures conflict with this protocol, the old material must be revised rather than altering this protocol implicitly.

---

## 13. Experiment rerun policy

Only experiments already forming part of the manuscript are rerun unless explicitly approved otherwise.

Rerun:

- MagNet preprocessing;
- candidate backbone pretraining;
- backbone comparison;
- selected BiGRU initialization;
- downstream fine-tuning;
- BiGRU depth experiment;
- loss-weight sensitivity experiment;
- final train/validation/test evaluation;
- final inference used for the frequency-screening demonstration.

Do not introduce unrelated new ablation studies or auxiliary experiments.

---

## 14. Change-control rule

This file is the authoritative MEPI specification.

Any proposed change must be classified as either:

### CONSISTENCY FIX
A code/manuscript change required to comply with this frozen protocol.

or

### PROTOCOL CHANGE
A modification to the scientific pipeline itself.

A PROTOCOL CHANGE must not be performed automatically.

It requires an explicit instruction containing:

`UNFREEZE MEPI-FROZEN-PROTOCOL v1.0`

followed by the exact item to change.

Until that instruction is given, conflicting suggestions, previous manuscript wording, exploratory discussions and alternative pipelines must be ignored.
