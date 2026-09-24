# B(t) Reprocessing Audit

Reprocessed 1050 samples (900 finetune + 150 demo) from raw Scope #2 CH1 using Np=10 and Ae=0.0001217268 m^2. Raw primary waveforms were available for every sample. Legacy CH2/Vout-derived B1024 files were comparison-only and never used as ground truth.

Algorithm: validate synchronous finite raw channels; least-squares remove DC/linear baseline; trapezoidal integration; remove integration drift; extract one centered complete cycle; resample periodically to 1024 points; remove residual B offset; compute peak, RMS, harmonics 2-5 THD, periodic maximum derivative, and RMS/mean-absolute form factor.

- B_peak_t: old mean 0.33583067, new mean 0.35796881, mean absolute change 0.022138143, changed 1050/1050
- B_rms: old mean 0.23651736, new mean 0.25203813, mean absolute change 0.01552077, changed 1050/1050
- B_thd_percent: old mean 0.77689336, new mean 0.87356435, mean absolute change 0.17858212, changed 1050/1050
- dBdt_max: old mean 5769.228, new mean 6639.8082, mean absolute change 1934.9102, changed 1050/1050
- form_factor: old mean 1.1101125, new mean 1.1096157, mean absolute change 0.00052116457, changed 1050/1050

Sanity: Bpeak-up-with-measured-Vin pair fraction=1.0000; Bpeak-down-with-frequency group fraction=1.0000; max repeat CV=0.8531%; abrupt periodic discontinuities (>10% peak)=0; Bpeak >1 T rows=15.

No remeasurement is indicated solely for B(t): every sample retains usable raw Scope #2 CH1. Dataset-level blockers are documented separately.
