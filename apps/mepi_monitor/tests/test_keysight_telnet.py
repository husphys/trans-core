from __future__ import annotations

import socket
import threading
import time
from contextlib import AbstractContextManager

import numpy as np
import pytest

from apps.mepi_monitor.acquisition import keysight_lan
from apps.mepi_monitor.acquisition.keysight_lan import KeysightLanScope, _SCPISocket
from apps.mepi_monitor.ui_tk.controller import test_scope_connection as _test_scope_connection

_IAC = 255
_DONT = 254
_DO = 253
_WONT = 252
_WILL = 251


class MockKeysightTelnetServer(AbstractContextManager["MockKeysightTelnetServer"]):
    """Mock the wakeup and negotiation order in the physical EDUX1052A trace."""

    def __init__(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(0.1)
        self.host, self.port = self.listener.getsockname()
        self.commands: list[str] = []
        self.negotiation_responses: list[tuple[int, int]] = []
        self.wakeups = 0
        self.errors: list[BaseException] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "MockKeysightTelnetServer":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self.listener.close()
        self._thread.join(timeout=2.0)
        if args[0] is None and self.errors:
            raise AssertionError(f"Mock Telnet server errors: {self.errors!r}")

    @staticmethod
    def _read_event(connection: socket.socket) -> tuple[str, object] | None:
        data = bytearray()
        while True:
            value = connection.recv(1)
            if not value:
                return None
            if value[0] == _IAC:
                command = connection.recv(1)
                if not command:
                    return None
                if command[0] in {_WILL, _WONT, _DO, _DONT}:
                    option = connection.recv(1)
                    if not option:
                        return None
                    return "negotiation", (command[0], option[0])
                continue
            if value == b"\n":
                return "line", data.rstrip(b"\r").decode("ascii")
            data.extend(value)

    def _expect_negotiation(self, connection: socket.socket) -> tuple[int, int]:
        event = self._read_event(connection)
        if event is None or event[0] != "negotiation":
            raise AssertionError(f"Expected Telnet negotiation, got {event!r}")
        response = event[1]
        assert isinstance(response, tuple)
        self.negotiation_responses.append(response)
        return response

    def _serve(self, connection: socket.socket) -> None:
        connection.settimeout(1.0)
        wakeup = self._read_event(connection)
        if wakeup != ("line", ""):
            raise AssertionError(f"Expected initial CRLF wakeup, got {wakeup!r}")
        self.wakeups += 1

        connection.sendall(bytes((_IAC, _WILL, 1)))
        if self._expect_negotiation(connection) != (_DO, 1):
            raise AssertionError("Client did not accept observed WILL option 1 with DO")

        connection.sendall(
            bytes((_IAC, _WILL, 3, _IAC, _WILL, 42))
            + b"Welcome to Keysight InfiniiVision Oscilloscope EDUX1052A - CN63260332\r\n>>"
        )
        responses = {self._expect_negotiation(connection), self._expect_negotiation(connection)}
        if responses != {(_DO, 3), (_DONT, 42)}:
            raise AssertionError(f"Unexpected Telnet option responses: {responses!r}")

        while not self._stop.is_set():
            try:
                event = self._read_event(connection)
            except (ConnectionError, OSError, socket.timeout):
                return
            if event is None:
                return
            if event[0] == "negotiation":
                response = event[1]
                assert isinstance(response, tuple)
                self.negotiation_responses.append(response)
                continue
            command = str(event[1])
            self.commands.append(command)
            echo = command.encode("ascii") + b"\r\n"
            if command == "*IDN?":
                response = b"\r\n>>" + echo + (
                    b"KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625\r\n>>\r\n"
                )
            elif command == ":MEASure:FREQuency? CHANnel1":
                response = echo + b"+2.129E+03\r\n>>\r\n"
            elif command == ":MEASure:VRMS? CHANnel1":
                response = echo + b"+3.86E+00\r\n>>\r\n"
            elif command == ":WAVeform:DATA?":
                payload = b",".join(str(index / 10).encode("ascii") for index in range(128))
                count = str(len(payload)).encode("ascii")
                response = (
                    echo
                    + b"#"
                    + str(len(count)).encode("ascii")
                    + count
                    + payload
                    + b"\r\n>>\r\n"
                )
            elif command == ":NOResponse?":
                time.sleep(0.25)
                continue
            elif command.endswith("?"):
                response = echo + b"0\r\n>>\r\n"
            else:
                response = echo + b">>\r\n"
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
                try:
                    self._serve(connection)
                except BaseException as exc:
                    self.errors.append(exc)


class DelayedHandshakeSocket:
    """Simulate a seven-second device delay without sleeping."""

    simulated_delay_s = 7.0

    def __init__(self) -> None:
        self.timeout = 0.0
        self.timeout_history: list[float] = []
        self.sent: list[bytes] = []
        self.recv_count = 0
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = float(timeout)
        self.timeout_history.append(self.timeout)

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self, _size: int) -> bytes:
        if self.recv_count == 0:
            assert self.timeout > self.simulated_delay_s
            result = (
                bytes((_IAC, _WILL, 1, _IAC, _WILL, 3))
                + b"Welcome to Keysight InfiniiVision Oscilloscope EDUX1052A - CN63260332\r\n>>\r\n"
            )
        elif self.recv_count == 1:
            assert self.timeout == pytest.approx(6.0)
            result = (
                b"*IDN?\r\n"
                b"KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625\r\n>>\r\n"
            )
        else:
            raise AssertionError("Unexpected extra receive")
        self.recv_count += 1
        return result

    def close(self) -> None:
        self.closed = True


def test_delayed_handshake_uses_separate_timeout_without_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = DelayedHandshakeSocket()
    connect_timeouts: list[float] = []

    def create_connection(_address: tuple[str, int], timeout: float) -> DelayedHandshakeSocket:
        connect_timeouts.append(float(timeout))
        return fake

    monkeypatch.setattr(keysight_lan.socket, "create_connection", create_connection)
    transport = _SCPISocket(
        "192.168.2.149", timeout_s=6.0, handshake_timeout_s=20.0
    )
    assert transport.connect().startswith("KEYSIGHT TECHNOLOGIES,EDUX1052A")
    assert connect_timeouts == [20.0]
    assert fake.sent[0] == b"\r\n"
    assert bytes((_IAC, _DO, 1)) in fake.sent
    assert bytes((_IAC, _DO, 3)) in fake.sent
    assert b"*IDN?\r\n" in fake.sent
    assert fake.timeout_history[-1] == pytest.approx(6.0)


def test_initialization_has_bounded_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = DelayedHandshakeSocket()
    fake.recv = lambda _size: b"\r\n"  # type: ignore[method-assign]
    clock = iter((100.0, 100.1, 131.0))
    monkeypatch.setattr(keysight_lan.socket, "create_connection", lambda *_args: fake)
    monkeypatch.setattr(keysight_lan.time, "monotonic", lambda: next(clock))
    transport = _SCPISocket(
        "192.168.2.149", timeout_s=6.0, handshake_timeout_s=30.0
    )
    with pytest.raises(
        TimeoutError,
        match="Telnet initialization timed out after 30 seconds following CRLF wakeup",
    ):
        transport.connect()
    assert fake.closed


def test_physical_negotiation_command_echo_queries_and_reconnect() -> None:
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
        assert server.wakeups == 2
        assert server.negotiation_responses.count((_DO, 1)) == 2
        assert server.negotiation_responses.count((_DO, 3)) == 2
        assert server.negotiation_responses.count((_DONT, 42)) == 2
        assert server.commands.count("*IDN?") == 2
        assert server.commands.count(":MEASure:VRMS? CHANnel1") == 2


def test_delayed_second_initialization_prompt_cannot_terminate_idn_query() -> None:
    with MockKeysightTelnetServer() as server:
        transport = _SCPISocket(server.host, server.port, timeout_s=0.5)
        assert transport.connect() == (
            "KEYSIGHT TECHNOLOGIES,EDUX1052A,CN63260332,02.12.2021071625"
        )
        assert server.commands == ["*IDN?"]
        transport.close()


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


def test_normal_query_timeout_remains_bounded_and_connection_helper_uses_telnet() -> None:
    with MockKeysightTelnetServer() as server:
        assert _test_scope_connection(server.host, server.port, timeout_s=0.5).startswith(
            "KEYSIGHT TECHNOLOGIES,EDUX1052A"
        )
        transport = _SCPISocket(server.host, server.port, timeout_s=0.1)
        transport.connect()
        transport.write(":WAVeform:FORMat ASCii")
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Timed out waiting for Keysight"):
            transport.query(":NOResponse?")
        assert time.monotonic() - started < 0.5
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
