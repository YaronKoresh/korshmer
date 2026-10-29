from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from korshmer.midi.controller_mapping import DEFAULT_CONTROLLER_TYPE
from korshmer.midi.message_validator import (
    validate_cc,
    validate_channel,
    validate_pitch_bend,
    validate_velocity,
)

LED_ON_VELOCITY: int = 127
LED_OFF_VELOCITY: int = 0


FADER_CC = "cc"
FADER_PITCH_BEND = "pitch_bend"


VOLUME_CC: int = 7
VOLUME_CHANNEL: int = 0


class LedSpec:
    __slots__ = ("kind", "channel", "number")

    def __init__(self, kind: str, channel: int, number: int):
        if kind not in ("note", "cc"):
            raise ValueError(f"LED kind must be 'note' or 'cc', got {kind!r}")
        self.kind = kind
        self.channel = validate_channel(channel)
        self.number = validate_velocity(number)

    def __repr__(self) -> str:
        return f"LedSpec({self.kind!r}, {self.channel}, {self.number})"

    def __eq__(self, other) -> bool:
        return isinstance(other, LedSpec) and (
            (self.kind, self.channel, self.number)
            == (other.kind, other.channel, other.number)
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.channel, self.number))


LED_MESSAGES: dict[str, dict[str, LedSpec]] = {
    "pioneer_ddj": {
        "play_1": LedSpec("note", 0, 0x0B),
        "cue_1": LedSpec("note", 0, 0x0C),
        "play_2": LedSpec("note", 1, 0x0B),
        "cue_2": LedSpec("note", 1, 0x0C),
    },
}


FADER_PROTOCOLS: dict[str, str] = {}


class MidiOutputPort:
    def __init__(self, port):
        self._port = port

    def send(self, message: dict) -> None:
        import mido

        self._port.send(mido.Message(**message))

    def close(self) -> None:
        self._port.close()


def find_output_name(input_name: str, output_names: list[str]) -> str | None:
    if input_name in output_names:
        return input_name
    lowered = input_name.lower()
    for name in output_names:
        low = name.lower()
        if low in lowered or lowered in low:
            return name
    return None


def open_midi_output(device_name: str):
    try:
        import mido

        name = find_output_name(device_name, list(mido.get_output_names()))
        if name is None:
            return None
        return MidiOutputPort(mido.open_output(name))
    except Exception:
        logging.debug("Could not open MIDI output", exc_info=True)
        return None


class ControllerOutput:
    def __init__(
        self,
        open_output: Callable[[str], Any] = open_midi_output,
        led_map: dict[str, dict[str, LedSpec]] | None = None,
        fader_protocols: dict[str, str] | None = None,
    ):
        self._open_output = open_output
        self._led_map = led_map if led_map is not None else LED_MESSAGES
        self._fader_protocols = (
            fader_protocols if fader_protocols is not None else FADER_PROTOCOLS
        )
        self._lock = threading.RLock()
        self._port = None
        self.device_name: str | None = None
        self.controller_type: str = DEFAULT_CONTROLLER_TYPE
        self._led_state: dict[str, bool] = {}
        self._received_leds: dict[str, bool] = {}
        self._received_volume: int | None = None

    @property
    def is_open(self) -> bool:
        return self._port is not None

    def open(self, device_name: str, controller_type: str) -> bool:
        self.close()
        try:
            port = self._open_output(device_name)
        except Exception:
            logging.debug("MIDI output open failed", exc_info=True)
            port = None
        if port is None:
            return False
        with self._lock:
            self._port = port
            self.device_name = device_name
            self.controller_type = controller_type
            states = dict(self._led_state)
        for button, on in states.items():
            self._send_led(button, on)
        return True

    def close(self) -> None:
        with self._lock:
            port, self._port = self._port, None
            self.device_name = None
            self.controller_type = DEFAULT_CONTROLLER_TYPE
        if port is not None:
            try:
                port.close()
            except Exception:
                logging.debug("MIDI output close failed", exc_info=True)

    def on_device_connected(
        self, device_name: str, controller_type: str
    ) -> None:
        self.open(device_name, controller_type)

    def on_device_disconnected(self, device_name: str) -> None:
        if device_name == self.device_name:
            self.close()

    def _send(self, message: dict) -> bool:
        with self._lock:
            port = self._port
        if port is None:
            return False
        try:
            port.send(message)
            return True
        except Exception:
            logging.debug("MIDI output send failed", exc_info=True)
            return False

    def _led_spec(self, button: str) -> LedSpec | None:
        return self._led_map.get(self.controller_type, {}).get(button)

    def _send_led(self, button: str, on: bool) -> bool:
        spec = self._led_spec(button)
        if spec is None:
            return False
        velocity = LED_ON_VELOCITY if on else LED_OFF_VELOCITY
        if spec.kind == "note":
            message = {
                "type": "note_on" if on else "note_off",
                "channel": spec.channel,
                "note": spec.number,
                "velocity": velocity,
            }
        else:
            message = {
                "type": "control_change",
                "channel": spec.channel,
                "control": spec.number,
                "value": velocity,
            }
        return self._send(message)

    def set_led(self, button: str, on: bool) -> bool:
        on = bool(on)
        with self._lock:
            self._led_state[button] = on
            echoed = self._received_leds.pop(button, None) == on
        if echoed:
            return False
        return self._send_led(button, on)

    def note_received(self, button: str, on: bool) -> None:
        with self._lock:
            self._received_leds[button] = bool(on)

    def volume_received(self, value: float) -> None:
        with self._lock:
            self._received_volume = self._volume_to_cc(value)

    @staticmethod
    def _volume_to_cc(value) -> int:
        value = float(value)
        if value != value:
            raise ValueError("volume must be a number, got NaN")
        return int(round(min(1.0, max(0.0, value)) * 127))

    def send_volume(self, value: float) -> bool:
        try:
            cc_value = self._volume_to_cc(value)
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._received_volume == cc_value:
                self._received_volume = None
                return False
            protocol = self._fader_protocols.get(self.controller_type, FADER_CC)
        if protocol == FADER_PITCH_BEND:
            bend = validate_pitch_bend(
                int(round(min(1.0, max(0.0, float(value))) * 16383))
            )
            return self._send(
                {
                    "type": "pitchwheel",
                    "channel": VOLUME_CHANNEL,
                    "pitch": bend - 8192,
                }
            )
        controller, cc_value = validate_cc(VOLUME_CC, cc_value)
        return self._send(
            {
                "type": "control_change",
                "channel": VOLUME_CHANNEL,
                "control": controller,
                "value": cc_value,
            }
        )

    def connect_engine(self, engine) -> None:
        engine.volumeChanged.connect(self.send_volume)
