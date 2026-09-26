from __future__ import annotations

import socket
import threading
import time
from contextlib import AbstractContextManager

import numpy as np
import pytest

from apps.mepi_monitor.acquisition.keysight_lan import KeysightLanScope, _SCPISocket
from apps.mepi_monitor.ui_tk.controller import test_scope_connection as _test_scope_connection

_IAC = 255
_DO = 253
_WILL = 251


class MockKeysightTelnetServer(AbstractContextManager["MockKeysightTelnetServer"]):
    def __init__(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(0.1)
        self.host, self.port = self.listener.getsockname()
        self.commands: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "MockKeysightTelnetServer":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self.listener.close()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _read_command(connection: socket.socket) -> str | None:
        data = bytearray()
        while True:
            value = connection.recv(1)
            if not value:
                return None
            if value[0] == _IAC:
                command = connection.recv(1)
                if not command:
                    return None
                if command[0] in {251, 252, 253, 254}:
                    if not connection.recv(1):
                        return None
                continue
            if value == b"\n":
                return data.rstrip(b"\r").decode("ascii")
            data.extend(value)

    def _serve(self, connection: socket.socket) -> None:
        connection.settimeout(1.0)
        connection.sendall(bytes((_IAC, _WILL)))
        connection.sendall(bytes((1, _IAC, _DO, 3)))
        connection.sendall(
            b"Welcome to Keysight InfiniiVision Oscilloscope EDUX1052A - CN63260332\r\n>>"
        )
        while not self._stop.is_set():
            try:
                command = self._read_command(connection)
            except (ConnectionError, OSError, socket.timeout):
                return
            if command is None:
                return
            self.commands.append(command)
            if command == "*IDN?":
                response = (
                    b"*IDN?\r\n"
                    b"KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625\r\n>>"
                )
            elif command == ":MEASure:FREQuency? CHANnel1":
                response = b"+2.129E+03\r\n>>"
            elif command == ":MEASure:VRMS? CHANnel1":
                response = b"+3.86E+00\r\n>>"
            elif command == ":WAVeform:DATA?":
                payload = b",".join(str(index / 10).encode("ascii") for index in range(128))
                count = str(len(payload)).encode("ascii")
                response = b"#" + str(len(count)).encode("ascii") + count + payload + b"\r\n>>"
            elif command == ":NOResponse?":
                time.sleep(0.25)
                continue
            elif command.endswith("?"):
                response = b"0\r\n>>"
            else:
                response = b"\r\n>>"
            midpoint = max(1, len(response) // 2)
            connection.sendall(response[:midpoint])
            connection.sendall(response[midpoint:])

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self.listener.accept()
            except (OSError, socket.timeout):
                continue
            with connection:
                self._serve(connection)


def test_telnet_banner_negotiation_repeated_queries_and_reconnect() -> None:
    with MockKeysightTelnetServer() as server:
        transport = _SCPISocket(server.host, server.port, timeout_s=0.5)
        expected_id = "KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625"
        assert transport.connect() == expected_id
        assert transport.qfloat(":MEASure:FREQuency? CHANnel1") == pytest.approx(2129.0)
        assert transport.qfloat(":MEASure:VRMS? CHANnel1") == pytest.approx(3.86)
        assert transport.qfloat(":MEASure:VRMS? CHANnel1") == pytest.approx(3.86)
        transport.close()
        assert transport.connect() == expected_id
        transport.close()
        assert server.commands.count("*IDN?") == 2
        assert server.commands.count(":MEASure:VRMS? CHANnel1") == 2


def test_telnet_ieee_waveform_framing_is_preserved() -> None:
    with MockKeysightTelnetServer() as server:
        transport = _SCPISocket(server.host, server.port, timeout_s=0.5)
        transport.connect()
        raw = transport.query_bytes(":WAVeform:DATA?")
        assert raw.startswith(b"#")
        values = KeysightLanScope._parse_ascii_waveform(raw)
        np.testing.assert_allclose(values, np.arange(128) / 10.0)
        assert transport.qfloat(":MEASure:VRMS? CHANnel1") == pytest.approx(3.86)
        transport.close()


def test_telnet_write_prompt_timeout_and_connection_helper() -> None:
    with MockKeysightTelnetServer() as server:
        assert _test_scope_connection(server.host, server.port, timeout_s=0.5).startswith(
            "KEYSIGHT TECHNOLOGIES,EDUX1052A"
        )
        transport = _SCPISocket(server.host, server.port, timeout_s=0.1)
        transport.connect()
        transport.write(":WAVeform:FORMat ASCii")
        with pytest.raises(TimeoutError, match="Timed out waiting for Keysight"):
            transport.query(":NOResponse?")
        transport.close()
        time.sleep(0.3)
        transport.timeout_s = 0.5
        assert transport.connect().startswith("KEYSIGHT TECHNOLOGIES,EDUX1052A")
        transport.close()


def test_keysight_scope_contract_acquires_two_channel_capture() -> None:
    with MockKeysightTelnetServer() as server:
        scope = KeysightLanScope(server.host, port=server.port, timeout_s=0.5)
        scope.connect()
        capture = scope.acquire()
        scope.disconnect()
        assert capture.frequency_hz == pytest.approx(2129.0)
        assert capture.frequency_source == "Keysight CH1 measurement"
        assert capture.time_s.shape == (128,)
        np.testing.assert_allclose(capture.vin_v, np.arange(128) / 10.0)
        np.testing.assert_allclose(capture.vout_v, np.arange(128) / 10.0)
        assert ":DIGitize CHANnel1,CHANnel2" in server.commands
        assert server.commands.count(":WAVeform:DATA?") == 2
