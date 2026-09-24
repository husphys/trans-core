"""Verified MEPI temperature-board channel parser.

The COM5 protocol is unchanged: 9600 baud, transmit ``b"T"``, receive two
temperatures.  Physical verification established that the board reply order is
TempCore,TempRoom.  The software API intentionally returns room first and core
second so downstream acquisition code can continue using ``tamb, tcore``.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

COM_PORT = "COM5"
BAUD_RATE = 9600
QUERY = b"T"
PHYSICAL_REPLY_ORDER = ("TempCore", "TempRoom")
API_RETURN_ORDER = ("TempRoom", "TempCore")

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def parse_temperature_board_reply(reply: str | bytes) -> tuple[float | None, float | None]:
    """Return ``(temp_room, temp_core)`` from a physical Core,Room reply."""

    text = reply.decode("ascii", "ignore") if isinstance(reply, bytes) else str(reply)
    numbers = _NUMBER.findall(text)
    if len(numbers) < 2:
        return None, None

    # VERIFIED PHYSICAL BOARD CHANNEL ORDER:
    # board reply = TempCore,TempRoom
    # function return = TempRoom,TempCore
    temp_core = float(numbers[0])
    temp_room = float(numbers[1])
    return temp_room, temp_core


class TempBoard:
    """Active COM5 reader with a drop-in ``read() -> (room, core)`` API."""

    def __init__(
        self,
        port: str = COM_PORT,
        *,
        timeout_s: float = 2.0,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.port = port
        self.timeout_s = timeout_s
        self.serial_factory = serial_factory
        self.ser: Any | None = None

    def connect(self) -> None:
        if self.serial_factory is None:
            import serial

            factory = serial.Serial
            serial_kwargs = {
                "bytesize": serial.EIGHTBITS,
                "parity": serial.PARITY_NONE,
                "stopbits": serial.STOPBITS_ONE,
            }
        else:
            factory = self.serial_factory
            serial_kwargs = {}
        self.ser = factory(
            port=self.port,
            baudrate=BAUD_RATE,
            timeout=self.timeout_s,
            **serial_kwargs,
        )
        time.sleep(0.4)
        self._flush_input()

    def _flush_input(self) -> None:
        if self.ser is None:
            return
        if hasattr(self.ser, "reset_input_buffer"):
            self.ser.reset_input_buffer()

    def read(self) -> tuple[float | None, float | None]:
        if self.ser is None:
            raise RuntimeError("Temperature board is not connected")
        self._flush_input()
        self.ser.write(QUERY)
        time.sleep(0.10)
        return parse_temperature_board_reply(self.ser.readline())

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None
