from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

from korshmer.midi.event_processor import MidiEvent
from korshmer.midi.message_validator import (
    validate_cc,
    validate_channel,
    validate_midi_note,
    validate_note_message,
)
from korshmer.midi_processing.mpe_support import MPEControllerMapper

DEFAULT_CONTROLLER_TYPE: str = "generic"


DEVICE_NAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "roli_seaboard": ("seaboard", "roli", "lightpad", "blocks"),
    "linnstrument": ("linnstrument", "linn"),
    "haken_continuum": ("continuum", "haken"),
}


DJ_CONTROLLER_TYPES: tuple[str, ...] = (
    "pioneer_ddj",
    "numark_mixtrack",
    "traktor_kontrol",
)


DJ_NAME_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "pioneer_ddj": (
        re.compile(r"\bddj\b|\bddj-"),
        re.compile(r"pioneer\s+dj"),
    ),
    "numark_mixtrack": (re.compile(r"mixtrack"),),
    "traktor_kontrol": (
        re.compile(r"traktor"),
        re.compile(r"\bkontrol\s+(?:[sdxfz]\d)\b(?!\d)"),
    ),
}


_BASE_CC_ACTIONS: dict[int, str] = {
    1: "modulation",
    7: "volume",
    10: "pan",
    11: "expression",
    64: "sustain",
}


_PIONEER_DDJ_CC_ACTIONS: dict[int, str] = {
    0x00: "tempo_fader",
    0x04: "trim",
    0x07: "eq_high",
    0x0B: "eq_mid",
    0x0C: "headphone_mix",
    0x0D: "headphone_gain",
    0x0F: "eq_low",
    0x13: "channel_fader",
    0x17: "filter",
    0x18: "filter",
    0x1F: "crossfader",
    0x21: "jog_bend",
    0x22: "jog",
    0x23: "jog",
    0x29: "jog_search",
    0x40: "browse",
}


_NUMARK_MIXTRACK_CC_ACTIONS: dict[int, str] = {
    0x01: "eq_high",
    0x02: "eq_mid",
    0x03: "eq_low",
    0x04: "filter",
    0x05: "channel_fader",
    0x07: "crossfader",
    0x08: "headphone_gain",
    0x09: "headphone_mix",
    0x0A: "master_gain",
    0x11: "jog",
    0x17: "browse",
    0x18: "tempo_fader",
    0x1A: "strip",
}


_TRAKTOR_KONTROL_CC_ACTIONS: dict[int, str] = dict(_BASE_CC_ACTIONS)


def _build_cc_maps() -> dict[str, dict[int, str]]:
    maps = {DEFAULT_CONTROLLER_TYPE: dict(_BASE_CC_ACTIONS)}
    maps["pioneer_ddj"] = dict(_PIONEER_DDJ_CC_ACTIONS)
    maps["numark_mixtrack"] = dict(_NUMARK_MIXTRACK_CC_ACTIONS)
    maps["traktor_kontrol"] = dict(_TRAKTOR_KONTROL_CC_ACTIONS)
    maps[DEFAULT_CONTROLLER_TYPE][74] = "timbre"
    for name, settings in MPEControllerMapper.CONTROLLER_PRESETS.items():
        table = dict(_BASE_CC_ACTIONS)
        table[settings["timbre_cc"]] = "timbre"
        table[settings["slide_cc"]] = "slide"
        maps[name] = table
    return maps


CC_ACTIONS: dict[str, dict[int, str]] = _build_cc_maps()


ACTION_NAMES: tuple[str, ...] = tuple(
    sorted({a for table in CC_ACTIONS.values() for a in table.values()})
)


def validate_action_name(action) -> str:
    if not isinstance(action, str) or action not in ACTION_NAMES:
        raise ValueError(f"Unknown controller action: {action!r}")
    return action


class ControllerAction(NamedTuple):
    action: str
    channel: int
    value: int


MAX_MACRO_ACTIONS: int = 16


TRIGGER_CC = "cc"
TRIGGER_NOTE = "note"


@dataclass(frozen=True)
class Macro:
    name: str
    trigger: tuple
    actions: tuple

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Macro name must be a non-empty string")
        object.__setattr__(self, "trigger", _check_trigger(self.trigger))
        object.__setattr__(self, "actions", _check_macro_actions(self))


def _check_trigger(trigger) -> tuple:
    if isinstance(trigger, (list, tuple)) and trigger:
        if trigger[0] == TRIGGER_CC and len(trigger) == 2:
            return (TRIGGER_CC, validate_cc(trigger[1], 0)[0])
        if trigger[0] == TRIGGER_NOTE and len(trigger) == 3:
            return (
                TRIGGER_NOTE,
                validate_channel(trigger[1]),
                validate_midi_note(trigger[2], "note"),
            )
    raise ValueError(f"Invalid macro trigger: {trigger!r}")


def _check_macro_actions(macro) -> tuple:
    actions = macro.actions
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("A macro needs at least one action")
    if len(actions) > MAX_MACRO_ACTIONS:
        raise ValueError(
            f"A macro has at most {MAX_MACRO_ACTIONS} actions, "
            f"got {len(actions)}"
        )
    checked = []
    for item in actions:
        try:
            action, value = item
        except (TypeError, ValueError):
            raise ValueError(f"Invalid macro action: {item!r}") from None
        if action == macro.name:
            raise ValueError("A macro cannot refer to itself")
        action = validate_action_name(action)
        if value is not None:
            value = validate_cc(0, value)[1]
        checked.append((action, value))
    return tuple(checked)


def as_action_list(result) -> list:
    if result is None:
        return []
    if isinstance(result, ControllerAction):
        return [result]
    return list(result)


def list_input_device_names() -> list[str]:
    try:
        import mido

        return list(mido.get_input_names())
    except Exception:
        return []


def detect_controller_type(device_name) -> str:
    if not isinstance(device_name, str):
        return DEFAULT_CONTROLLER_TYPE
    lowered = device_name.lower()
    for controller_type, patterns in DJ_NAME_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            return controller_type
    for controller_type, keywords in DEVICE_NAME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return controller_type
    return DEFAULT_CONTROLLER_TYPE


class ControllerMapping:
    def __init__(self, controller_type: str = DEFAULT_CONTROLLER_TYPE):
        if controller_type not in CC_ACTIONS:
            controller_type = DEFAULT_CONTROLLER_TYPE
        self.controller_type = controller_type
        self.cc_actions = dict(CC_ACTIONS[controller_type])
        self.macros: dict[str, Macro] = {}

    @classmethod
    def for_device(cls, device_name) -> ControllerMapping:
        return cls(detect_controller_type(device_name))

    def set_cc_action(self, controller, action) -> None:
        controller, _ = validate_cc(controller, 0)
        self.cc_actions[controller] = validate_action_name(action)

    def remove_cc_action(self, controller) -> bool:
        controller, _ = validate_cc(controller, 0)
        return self.cc_actions.pop(controller, None) is not None

    def assign_macro(self, macro: Macro) -> None:
        for other in self.macros.values():
            if other.name != macro.name and other.trigger == macro.trigger:
                raise ValueError(
                    f"Trigger {macro.trigger!r} already starts "
                    f"macro {other.name!r}"
                )
        self.macros[macro.name] = macro

    def remove_macro(self, name: str) -> bool:
        return self.macros.pop(name, None) is not None

    def _macro_for(self, trigger: tuple) -> Macro | None:
        for macro in self.macros.values():
            if macro.trigger == trigger:
                return macro
        return None

    @staticmethod
    def _expand(macro: Macro, channel: int, value: int) -> list:
        return [
            ControllerAction(action, channel, value if v is None else v)
            for action, v in macro.actions
        ]

    def reset_cc_actions(self) -> None:
        self.cc_actions = dict(CC_ACTIONS[self.controller_type])

    def replace_cc_actions(self, table) -> None:
        checked = {}
        for controller, action in dict(table).items():
            controller, _ = validate_cc(controller, 0)
            checked[controller] = validate_action_name(action)
        self.cc_actions = checked

    def to_dict(self) -> dict:
        return {
            "controller_type": self.controller_type,
            "cc_actions": {
                str(cc): action
                for cc, action in sorted(self.cc_actions.items())
            },
            "macros": [
                {
                    "name": m.name,
                    "trigger": list(m.trigger),
                    "actions": [list(a) for a in m.actions],
                }
                for m in self.macros.values()
            ],
        }

    def _load_macros(self, entries) -> None:
        if not isinstance(entries, list):
            return
        loaded = ControllerMapping(self.controller_type)
        for entry in entries:
            try:
                loaded.assign_macro(
                    Macro(
                        entry["name"],
                        tuple(entry["trigger"]),
                        tuple(tuple(a) for a in entry["actions"]),
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue
        self.macros = loaded.macros

    def load_dict(self, data) -> None:
        if not isinstance(data, dict):
            return
        self._load_macros(data.get("macros"))
        table = data.get("cc_actions")
        if not isinstance(table, dict):
            return
        loaded = {}
        for controller, action in table.items():
            try:
                controller, _ = validate_cc(int(controller), 0)
                loaded[controller] = validate_action_name(action)
            except (TypeError, ValueError):
                continue
        self.cc_actions = loaded

    def map_cc(self, channel, controller, value):
        channel = validate_channel(channel)
        controller, value = validate_cc(controller, value)
        macro = self._macro_for((TRIGGER_CC, controller))
        if macro is not None:
            return self._expand(macro, channel, value)
        action = self.cc_actions.get(controller)
        if action is None:
            return None
        return ControllerAction(action, channel, value)

    def map_note(self, channel, note, velocity=0):
        channel, note, velocity = validate_note_message(channel, note, velocity)
        if velocity > 0:
            macro = self._macro_for((TRIGGER_NOTE, channel, note))
            if macro is not None:
                return self._expand(macro, channel, velocity)
        name = "note_on" if velocity > 0 else "note_off"
        return ControllerAction(name, channel, note)

    def map_message(self, message: dict):
        kind = message.get("type")
        if kind == "control_change":
            return self.map_cc(
                message.get("channel"),
                message.get("control"),
                message.get("value"),
            )
        if kind in ("note_on", "note_off"):
            velocity = message.get("velocity", 0) if kind == "note_on" else 0
            action = self.map_note(
                message.get("channel"), message.get("note"), velocity
            )
            return action
        return None


def note_message_to_event(
    message: dict, time: float, note_id: int
) -> MidiEvent:
    kind = message.get("type")
    if kind not in ("note_on", "note_off"):
        raise ValueError(f"Not a note message: {kind!r}")
    velocity = message.get("velocity", 0) if kind == "note_on" else 0
    channel, note, velocity = validate_note_message(
        message.get("channel"), message.get("note"), velocity
    )
    etype = "noteon" if kind == "note_on" and velocity > 0 else "noteoff"
    return MidiEvent(float(time), etype, channel, note, velocity, note_id)
