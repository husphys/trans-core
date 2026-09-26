# MEPI EDUX1052A Telnet initialization audit

Date: 2026-09-26
Source baseline: `0955d7228af0a1468618b69a93a877b9c341dd5a` (`main`)
Status: **BUILD VALIDATION PASS; CUSTOM PHYSICAL TRANSPORT PENDING**

## Scope

This correction is limited to the Telnet session initialization used by the Raspberry Pi deployment. `_frequency()`, `_waveform()`, `acquire()`, `ScopeCapture`, preprocessing, B(t), features, model/checkpoints/scalers, domain logic, datasets, training, results, and manuscript are unchanged.

## Packet-trace-grounded correction

The physical EDUX1052A does not send its banner immediately after TCP connect. The working Linux Telnet session first sends an empty CRLF; the scope then negotiates `IAC WILL 1` and `IAC WILL 3`, and the client answers `IAC DO 1` and `IAC DO 3` before the banner/prompt and echoed SCPI exchange.

The custom transport now:

- sends exactly one initial CRLF wakeup after TCP establishment;
- uses a separate, bounded 30-second connection/handshake window rather than the ordinary 6-second SCPI query timeout;
- accepts only observed server WILL options 1 and 3 with DO;
- rejects unsupported server WILL options with DONT and refuses unrequested client-side options;
- restores the normal query timeout before issuing authoritative `*IDN?`;
- closes cleanly after initialization failure;
- preserves command echo/prompt cleanup, Telnet-control filtering, reconnect, and length-driven IEEE waveform-block parsing.

The banner is not treated as device identity; the returned `*IDN?` value remains authoritative.

## Files changed

- `apps/mepi_monitor/acquisition/keysight_lan.py`
- `apps/mepi_monitor/tests/test_keysight_telnet.py`
- `scripts/build_mepi_pi_deploy.py`
- `scripts/pi_deploy_templates/README_PI.md`
- `reports/MEPI_EDUX1052A_TELNET_INITIALIZATION_AUDIT.md`
- regenerated corresponding files and manifest under `deploy/MEPI_PI_DEPLOY/`
- rebuilt `deploy/MEPI_PI_DEPLOY.zip`

## Validation boundary

Mock tests cover delayed initialization without a real multi-second sleep, total handshake deadline, observed option acceptance, unsupported option rejection, wakeup CRLF, command echo, identity, numeric response, normal query timeout, reconnect, and IEEE waveform framing. Mock success is not physical transport success.

User-supplied evidence establishes that Linux Telnet identity/frequency/VRMS on port 5024 is **PASS**. The custom MEPI transport remains **PENDING** until rerun on the physical Pi. Physical CH1/CH2 waveform acquisition and physical live MEPI inference remain **NOT YET VERIFIED**.
