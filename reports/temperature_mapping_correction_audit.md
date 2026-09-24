# Temperature Mapping Correction Audit

Verified correction: `SWAPPED_CORE_ROOM_CHANNELS_PHYSICALLY_VERIFIED`. All 14 session configs use the same legacy `TempRoom,TempCore` label and no session-specific wiring override was found. Corrected 900 finetune and 150 demo rows without modifying raw/session evidence.

Finetune before swap: ambient mean 27.8472 C, core mean 26.8028 C, rise mean -1.04444 C; rise signs negative/zero/positive = 852/32/16.

Finetune after swap: ambient mean 26.8028 C, core mean 27.8472 C, rise mean 1.04444 C; rise signs negative/zero/positive = 16/32/852. Negative post-swap values were retained.

Session diagnostics include jumps, timestamp-based rates and quantization. Audit-only >1 C/s rate screen count: 0; this screen does not alter QC. No independent reference-thermometer calibration file was found, so absolute calibration remains pending.
