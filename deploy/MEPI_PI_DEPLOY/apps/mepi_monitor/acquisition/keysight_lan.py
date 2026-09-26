"""Keysight EDUX raw-LAN acquisition adapted from the project acquisition script.

Only Scope #2 semantics are used: CH1 primary Vin and CH2 secondary Vout from
one synchronized acquisition.  No Keithley or second oscilloscope is required.
"""

from __future__ import annotations

import socket
import statistics
import time
from dataclasses import dataclass

import numpy as np

SCPI_PORT = 5025
WAVEFORM_POINTS = 5000


@dataclass(frozen=True)
class ScopeCapture:
    time_s: np.ndarray
    vin_v: np.ndarray
    vout_v: np.ndarray
    frequency_hz: float
    frequency_source: str


class _SCPISocket:
    def __init__(self, ip: str, port: int = SCPI_PORT, timeout_s: float = 6.0) -> None:
        self.ip, self.port, self.timeout_s = ip, int(port), float(timeout_s)
        self.socket: socket.socket | None = None
        self.buffer = b""

    def connect(self) -> str:
        self.close()
        self.socket = socket.create_connection((self.ip, self.port), self.timeout_s)
        self.socket.settimeout(self.timeout_s)
        return self.query("*IDN?")

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
        self.socket, self.buffer = None, b""

    def write(self, command: str) -> None:
        if self.socket is None:
            raise RuntimeError("Scope is not connected")
        self.socket.sendall((command.strip() + "\n").encode("ascii"))

    def query_bytes(self, command: str) -> bytes:
        self.write(command)
        while b"\n" not in self.buffer:
            if self.socket is None:
                raise RuntimeError("Scope is not connected")
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError("Scope closed the raw-LAN connection")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.rstrip(b"\r")

    def query(self, command: str) -> str:
        return self.query_bytes(command).decode("ascii", "ignore").strip()

    def qfloat(self, command: str) -> float:
        value = float(self.query(command))
        if not np.isfinite(value) or abs(value) >= 1e30:
            raise ValueError(f"Invalid scope response to {command}: {value}")
        return value


class KeysightLanScope:
    def __init__(self, ip: str, *, port: int = SCPI_PORT, timeout_s: float = 6.0) -> None:
        self.ip = ip
        self.scpi = _SCPISocket(ip, port, timeout_s)
        self.idn: str | None = None

    def connect(self) -> str:
        self.idn = self.scpi.connect()
        return self.idn

    def disconnect(self) -> None:
        self.scpi.close()
        self.idn = None

    @property
    def connected(self) -> bool:
        return self.idn is not None

    def _frequency(self) -> float:
        readings = []
        for _ in range(5):
            try:
                value = self.scpi.qfloat(":MEASure:FREQuency? CHANnel1")
                if value > 0.0:
                    readings.append(value)
            except (ValueError, OSError):
                pass
            time.sleep(0.08)
        if not readings:
            raise RuntimeError("Keysight CH1 frequency measurement unavailable")
        result = float(statistics.median(readings))
        if (max(readings) - min(readings)) / result > 0.05:
            raise RuntimeError("Keysight CH1 frequency measurement is unstable")
        return result

    @staticmethod
    def _parse_ascii_waveform(raw: bytes) -> np.ndarray:
        payload = raw.strip()
        if payload.startswith(b"#") and len(payload) >= 2:
            digits = int(chr(payload[1]))
            if digits:
                start = 2 + digits
                count = int(payload[2:start].decode("ascii"))
                payload = payload[start : start + count]
        values = np.fromstring(payload.decode("ascii", "ignore"), sep=",")
        if values.size < 100 or not np.isfinite(values).all():
            raise RuntimeError("Keysight returned an incomplete/non-finite waveform")
        return values

    def _waveform(self, channel: int) -> tuple[np.ndarray, np.ndarray]:
        self.scpi.write(":WAVeform:FORMat ASCii")
        self.scpi.write(":WAVeform:POINts:MODE NORMal")
        self.scpi.write(f":WAVeform:POINts {WAVEFORM_POINTS}")
        self.scpi.write(f":WAVeform:SOURce CHANnel{channel}")
        increment = self.scpi.qfloat(":WAVeform:XINCrement?")
        try:
            origin = self.scpi.qfloat(":WAVeform:XORigin?")
        except ValueError:
            origin = 0.0
        values = self._parse_ascii_waveform(self.scpi.query_bytes(":WAVeform:DATA?"))
        return origin + np.arange(values.size) * increment, values

    def acquire(self) -> ScopeCapture:
        if not self.connected:
            raise RuntimeError("Connect to the Keysight scope before acquisition")
        frequency = self._frequency()
        self.scpi.write(":DIGitize CHANnel1,CHANnel2")
        try:
            try:
                self.scpi.query("*OPC?")
            except Exception:
                pass
            t1, vin = self._waveform(1)
            t2, vout = self._waveform(2)
        finally:
            try:
                self.scpi.write(":RUN")
            except Exception:
                pass
        count = min(t1.size, vin.size, t2.size, vout.size)
        if not np.allclose(t1[:count], t2[:count], rtol=0.0, atol=1e-12):
            raise RuntimeError("Keysight CH1/CH2 time bases differ")
        return ScopeCapture(t1[:count], vin[:count], vout[:count], frequency, "Keysight CH1 measurement")
