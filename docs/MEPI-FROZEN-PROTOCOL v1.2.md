# MEPI-FROZEN-PROTOCOL v1.2

**Status:** FROZEN  
**Date:** 2026-09-24  
**Supersedes:** MEPI-FROZEN-PROTOCOL v1.1 (2026-09-22)  
**Purpose:** Narrow amendment resolving temperature-channel provenance, the primary session set, the LSP target, and its train-only normalization.

MEPI-FROZEN-PROTOCOL v1.1 was explicitly unfrozen for this amendment. All
v1.1 requirements not named below remain unchanged, including the MagNet
representation-pretraining pipeline, architecture, Scope #2 CH1 primary-Vin
source for `B(t)_1024`, (N_p=10), (A_e=1.217268\times10^{-4}\;m^2), the
nine tabular inputs, three downstream targets, QC policy, electrical formulas,
raw-data immutability, leakage controls, group-wise 80/10/10 partitioning,
validation-only model development, and test-only final evaluation.

## v1.2 Amendment from v1.1

### 1. Corrected temperature mapping

The experimentally verified physical order is:

- board field 1 = physical Core sensor;
- board field 2 = physical Ambient sensor.

Historical acquisition labels were reversed. Preserve them as
`temperature_ambient_raw_legacy_c` and `temperature_core_raw_legacy_c`, then
apply exactly one correction:

```text
temperature_ambient_c = temperature_core_raw_legacy_c
temperature_core_c    = temperature_ambient_raw_legacy_c
temp_rise_c            = temperature_core_c - temperature_ambient_c
```

Every corrected row must carry mapping provenance and a guard that rejects an
attempted second swap. Future acquisition software must expose `ambient` from
physical field 2 and `core` from physical field 1. This mapping correction is
not an offset, slope, or sensor-calibration fit; no such calibration may be
inferred from transformer measurements.

### 2. Primary and supplementary session definition

The primary paired grid contains both FE and COMMERCIAL cores at nominal
voltages 2.8, 3.3, 3.9, 4.5, 5.0, and 5.6 V, frequencies 1000 through 4500 Hz
in 250-Hz increments, and five repeats per core and condition. The measured
design therefore contains 900 rows in 90 nominal voltage-frequency groups.

The experimenter-verified new COMMERCIAL 5.6-V session is the primary
COMMERCIAL 5.6-V condition. The historical COMMERCIAL approximately 6.x-V
session is preserved with
`condition_role = supplementary_excluded_from_primary` and must not enter the
primary candidate dataset, train/validation/test subsets, normalization, or
main paired-grid metrics. Actual measured `vin_rms_v` values are never replaced
by nominal group values.

### 3. Exact LSP equation

Define corrected absolute core temperature and the raw LSP target in float64:

\[
T_{core,K}=\texttt{temperature\_core\_c}+273.15
\]

\[
\log(LSP_{raw})=\frac{E_{a,eff}}{R}
\left(\frac{1}{T_{core,K}}-\frac{1}{T_{ref}}\right)
\]

\[
LSP_{raw}=\exp(\log(LSP_{raw})).
\]

No epsilon is added. `LSP_raw` is finite, positive, dimensionless, equals one
at (T_{core,K}=T_{ref}), and decreases monotonically as core temperature
increases. It is a **relative thermal-stress proxy**, not actual lifetime,
remaining useful life, time-to-failure, or service hours. LSP is constructed
only for QC-valid primary rows; HARD_FAIL rows remain in the master evidence
without an LSP training target.

### 4. Ea_eff and T_ref

Freeze:

```text
R      = 8.314462618 J mol^-1 K^-1
Ea_eff = 125000.0 J/mol
T_ref  = 298.15 K
```

`Ea_eff` is a **literature-informed effective thermal-aging sensitivity
parameter**. It is not experimentally identified from this dataset, not a
measured activation energy of the nanocrystalline core, and not a material
constant established by this experiment.

### 5. Train-only LSP normalization

First split the 90 operating-condition group identities using the existing
deterministic seed 42 into exactly 72 training, 9 validation, and 9 test
groups. Both cores and all QC-valid repeats from a nominal voltage-frequency
condition remain together; incomplete groups caused by QC exclusion remain
usable. No group may cross subsets.

Only after assignment, fit the project's standard scaler on training
`LSP_raw` values. It uses the population standard deviation (`numpy.std` with
`ddof=0`):

\[
LSP_z=\frac{LSP_{raw}-\mu_{LSP,train}}{\sigma_{LSP,train}}.
\]

Validation and test use the unchanged training mean and scale. They are never
refitted. Reported physical-space LSP metrics must inverse-transform to
`LSP_raw` unless explicitly labeled normalized-space.

### 6. Arrhenius residual normalization

There is no separately fitted Arrhenius-residual scaler. Normalize the
Arrhenius reference with the same training LSP scaler:

\[
LSP_{Arr,z}=\frac{LSP_{Arr,raw}-\mu_{LSP,train}}
{\sigma_{LSP,train}}
\]

and define the dimensionless residual

\[
r_{Arr}=LSP_{aux,z}-LSP_{Arr,z}.
\]

The latent physics-residual projection and Arrhenius-related regularization
must use this residual. No validation or test statistic may participate in its
normalization.

## Frozen change control

This v1.2 document is now the authoritative protocol. A scientific change
requires an explicit instruction containing:

`UNFREEZE MEPI-FROZEN-PROTOCOL v1.2`

Consistency fixes required to implement this text do not change the protocol.
