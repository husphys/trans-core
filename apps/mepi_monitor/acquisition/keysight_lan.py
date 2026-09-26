"""Keysight EDUX Telnet/SCPI acquisition adapted from the project acquisition script.

Only Scope #2 semantics are used: CH1 primary Vin and CH2 secondary Vout from
one synchronized acquisition.  No Keithley or second oscilloscope is required.
"""

from __future__ import annotations

import socket
import statistics
import time
from dataclasses import dataclass

import numpy as np

SCPI_PORT = 5024
WAVEFORM_POINTS = 5000
TELNET_HANDSHAKE_TIMEOUT_S = 30.0

_IAC = 255
_DONT = 254
_DO = 253
_WONT = 252
_WILL = 251
_SB = 250
_SE = 240
_PROMPT = b">>"
_ACCEPTED_SERVER_WILL_OPTIONS = frozenset({1, 3})


@dataclass(frozen=True)
class ScopeCapture:
    time_s: np.ndarray
    vin_v: np.ndarray
    vout_v: np.ndarray
    frequency_hz: float
    frequency_source: str


class _SCPISocket:
    def __init__(
        self,
        ip: str,
        port: int = SCPI_PORT,
        timeout_s: float = 6.0,
        handshake_timeout_s: float = TELNET_HANDSHAKE_TIMEOUT_S,
    ) -> None:
        self.ip, self.port = ip, int(port)
        self.timeout_s = float(timeout_s)
        self.handshake_timeout_s = float(handshake_timeout_s)
        if self.timeout_s <= 0.0 or self.handshake_timeout_s <= 0.0:
            raise ValueError("Keysight timeouts must be positive")
        self.socket: socket.socket | None = None
        self.buffer = b""
        self._telnet_state = "data"
        self._telnet_command: int | None = None

    def connect(self) -> str:
        self.close()
        try:
            self.socket = socket.create_connection(
                (self.ip, self.port), self.handshake_timeout_s
            )
            self.socket.settimeout(self.handshake_timeout_s)
            self.buffer = b""
            self._telnet_state = "data"
            self._telnet_command = None
            # The physical EDUX1052A trace shows negotiation begins only after
            # the Linux Telnet client sends an empty CRLF line.
            self.socket.sendall(b"\r\n")
            try:
                self._read_until_prompt(
                    "Telnet initialization",
                    overall_timeout_s=self.handshake_timeout_s,
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    "Keysight Telnet initialization timed out after "
                    f"{self.handshake_timeout_s:g} seconds following CRLF wakeup"
                ) from exc
            # The banner is session UI only. *IDN? below is authoritative.
            self.socket.settimeout(self.timeout_s)
            return self.query("*IDN?")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
        self.socket, self.buffer = None, b""
        self._telnet_state = "data"
        self._telnet_command = None

    def _send_telnet_response(self, command: int, option: int) -> None:
        if self.socket is None:
            raise RuntimeError("Scope is not connected")
        if command == _WILL:
            response = _DO if option in _ACCEPTED_SERVER_WILL_OPTIONS else _DONT
        elif command in {_DO, _DONT}:
            response = _WONT
        else:
            response = _DONT
        self.socket.sendall(bytes((_IAC, response, option)))

    def _decode_telnet(self, chunk: bytes) -> bytes:
        """Remove Telnet controls while preserving application bytes exactly."""

        output = bytearray()
        for value in chunk:
            if self._telnet_state == "data":
                if value == _IAC:
                    self._telnet_state = "iac"
                else:
                    output.append(value)
            elif self._telnet_state == "iac":
                if value == _IAC:
                    output.append(_IAC)
                    self._telnet_state = "data"
                elif value in {_DO, _DONT, _WILL, _WONT}:
                    self._telnet_command = value
                    self._telnet_state = "negotiation"
                elif value == _SB:
                    self._telnet_state = "subnegotiation"
                else:
                    self._telnet_state = "data"
            elif self._telnet_state == "negotiation":
                if self._telnet_command is not None:
                    self._send_telnet_response(self._telnet_command, value)
                self._telnet_command = None
                self._telnet_state = "data"
            elif self._telnet_state == "subnegotiation":
                if value == _IAC:
                    self._telnet_state = "subnegotiation_iac"
            elif self._telnet_state == "subnegotiation_iac":
                self._telnet_state = "data" if value == _SE else "subnegotiation"
        return bytes(output)

    def _receive(self, context: str) -> None:
        if self.socket is None:
            raise RuntimeError("Scope is not connected")
        try:
            chunk = self.socket.recv(65536)
        except socket.timeout as exc:
            raise TimeoutError(f"Timed out waiting for Keysight {context}") from exc
        if not chunk:
            raise RuntimeError("Scope closed the Telnet/SCPI connection")
        self.buffer += self._decode_telnet(chunk)

    def _read_until_prompt(
        self, context: str, *, overall_timeout_s: float | None = None
    ) -> bytes:
        deadline = (
            None if overall_timeout_s is None else time.monotonic() + overall_timeout_s
        )
        while _PROMPT not in self.buffer:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"Timed out waiting for Keysight {context}")
                if self.socket is None:
                    raise RuntimeError("Scope is not connected")
                self.socket.settimeout(remaining)
            self._receive(context)
        response, self.buffer = self.buffer.split(_PROMPT, 1)
        return response

    @staticmethod
    def _clean_text_response(raw: bytes, command: str, *, allow_empty: bool = False) -> bytes:
        """Strip only whole Telnet UI lines; never alter bytes inside a payload."""

        command_bytes = command.strip().encode("ascii")
        cleaned: list[bytes] = []
        for line in raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n"):
            value = line.strip()
            if not value or value == command_bytes:
                continue
            if value.startswith(b"Welcome to Keysight InfiniiVision Oscilloscope"):
                continue
            cleaned.append(value)
        response = b"\n".join(cleaned)
        if not response and not allow_empty:
            raise RuntimeError(f"Keysight returned an empty response to {command}")
        return response

    def _send_command(self, command: str) -> None:
        if self.socket is None:
            raise RuntimeError("Scope is not connected")
        if self.buffer.strip():
            raise RuntimeError("Unexpected unread data before the next Keysight command")
        self.buffer = b""
        self.socket.sendall(command.strip().encode("ascii") + b"\r\n")

    def write(self, command: str) -> None:
        self._send_command(command)
        response = self._clean_text_response(
            self._read_until_prompt(command), command, allow_empty=True
        )
        if response:
            raise RuntimeError(f"Unexpected Keysight response to {command}: {response!r}")

    def query_bytes(self, command: str) -> bytes:
        self._send_command(command)
        while True:
            block_marker = self.buffer.find(b"#")
            prompt_marker = self.buffer.find(_PROMPT)
            if block_marker >= 0 and (prompt_marker < 0 or block_marker < prompt_marker):
                prefix = self.buffer[:block_marker]
                if self._clean_text_response(prefix, command, allow_empty=True):
                    raise RuntimeError("Unexpected text before Keysight IEEE waveform block")
                while len(self.buffer) < block_marker + 2:
                    self._receive(command)
                digits_byte = self.buffer[block_marker + 1]
                if not 48 <= digits_byte <= 57 or digits_byte == 48:
                    raise RuntimeError("Unsupported Keysight IEEE waveform block header")
                digits = digits_byte - 48
                header_end = block_marker + 2 + digits
                while len(self.buffer) < header_end:
                    self._receive(command)
                try:
                    count = int(self.buffer[block_marker + 2 : header_end].decode("ascii"))
                except ValueError as exc:
                    raise RuntimeError("Invalid Keysight IEEE waveform block length") from exc
                payload_end = header_end + count
                while len(self.buffer) < payload_end:
                    self._receive(command)
                result = self.buffer[block_marker:payload_end]
                self.buffer = self.buffer[payload_end:]
                trailer = self._read_until_prompt(command)
                if self._clean_text_response(trailer, command, allow_empty=True):
                    raise RuntimeError("Unexpected text after Keysight IEEE waveform block")
                return result
            if prompt_marker >= 0:
                return self._clean_text_response(self._read_until_prompt(command), command)
            self._receive(command)

    def query(self, command: str) -> str:
        return self.query_bytes(command).decode("ascii", "strict").strip()

    def qfloat(self, command: str) -> float:
        value = float(self.query(command))
        if not np.isfinite(value) or abs(value) >= 1e30:
            raise ValueError(f"Invalid scope response to {command}: {value}")
        return value


class KeysightLanScope:
    def __init__(
        self,
        ip: str,
        *,
        port: int = SCPI_PORT,
        timeout_s: float = 6.0,
        handshake_timeout_s: float = TELNET_HANDSHAKE_TIMEOUT_S,
    ) -> None:
        self.ip = ip
        self.scpi = _SCPISocket(ip, port, timeout_s, handshake_timeout_s)
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
