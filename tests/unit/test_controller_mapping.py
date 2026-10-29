import pytest

from korshmer.midi.controller_mapping import (
    CC_ACTIONS,
    DEFAULT_CONTROLLER_TYPE,
    ControllerAction,
    ControllerMapping,
    detect_controller_type,
    list_input_device_names,
    note_message_to_event,
)
from korshmer.midi.event_processor import MidiEventQueue
from korshmer.midi_processing.mpe_support import MPEControllerMapper


def test_known_device_names_map_to_presets():
    assert detect_controller_type("ROLI Seaboard Block") == "roli_seaboard"
    assert detect_controller_type("LinnStrument MIDI 1") == "linnstrument"
    assert detect_controller_type("Haken Audio Continuum") == "haken_continuum"


def test_detected_types_are_preset_keys():
    for name in ("ROLI Seaboard Block", "LinnStrument", "Continuum"):
        assert detect_controller_type(name) in (
            MPEControllerMapper.CONTROLLER_PRESETS
        )


@pytest.mark.parametrize("name", ["Unknown Keys 49", "", None, 5])
def test_unknown_device_gets_default(name):
    assert detect_controller_type(name) == DEFAULT_CONTROLLER_TYPE


def test_presets_are_not_extended():
    assert DEFAULT_CONTROLLER_TYPE not in MPEControllerMapper.CONTROLLER_PRESETS
    assert set(MPEControllerMapper.CONTROLLER_PRESETS) == {
        "roli_seaboard",
        "linnstrument",
        "haken_continuum",
    }


def test_every_type_has_a_cc_table():
    for name in MPEControllerMapper.CONTROLLER_PRESETS:
        assert name in CC_ACTIONS
    assert DEFAULT_CONTROLLER_TYPE in CC_ACTIONS


def test_cc_74_maps_to_timbre():
    mapping = ControllerMapping.for_device("ROLI Seaboard Block")
    assert mapping.map_cc(0, 74, 64) == ControllerAction("timbre", 0, 64)


def test_unmapped_cc_returns_none():
    assert ControllerMapping().map_cc(0, 99, 10) is None


def test_cc_number_128_is_rejected():
    with pytest.raises(ValueError):
        ControllerMapping().map_cc(0, 128, 10)


def test_cc_value_and_channel_are_validated():
    mapping = ControllerMapping()
    with pytest.raises(ValueError):
        mapping.map_cc(0, 74, 128)
    with pytest.raises(ValueError):
        mapping.map_cc(16, 74, 10)


def test_note_messages():
    mapping = ControllerMapping()
    assert mapping.map_message(
        {"type": "note_on", "channel": 1, "note": 60, "velocity": 90}
    ) == ControllerAction("note_on", 1, 60)
    assert mapping.map_message(
        {"type": "note_on", "channel": 1, "note": 60, "velocity": 0}
    ) == ControllerAction("note_off", 1, 60)
    assert mapping.map_message(
        {"type": "note_off", "channel": 1, "note": 60, "velocity": 40}
    ) == ControllerAction("note_off", 1, 60)
    with pytest.raises(ValueError):
        mapping.map_message({"type": "note_on", "channel": 1, "note": 128})


def test_map_message_ignores_other_types():
    assert ControllerMapping().map_message({"type": "pitchwheel"}) is None


def test_note_message_to_event_feeds_queue():
    queue = MidiEventQueue()
    queue.push(
        note_message_to_event(
            {"type": "note_on", "channel": 2, "note": 61, "velocity": 80},
            0.5,
            7,
        )
    )
    queue.push(
        note_message_to_event(
            {"type": "note_off", "channel": 2, "note": 61}, 1.0, 7
        )
    )
    events = queue.sorted_events()
    assert [e.etype for e in events] == ["noteon", "noteoff"]
    assert events[0].velocity == 80 and events[1].velocity == 0
    with pytest.raises(ValueError):
        note_message_to_event({"type": "control_change"}, 0.0, 1)


def test_device_listing_without_backend_is_a_list(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mido", None)
    assert list_input_device_names() == []


def test_device_listing_uses_backend(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(get_input_names=lambda: ["A", "B"])
    monkeypatch.setitem(sys.modules, "mido", fake)
    assert list_input_device_names() == ["A", "B"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Pioneer DDJ-400", "pioneer_ddj"),
        ("DDJ-400 MIDI 1", "pioneer_ddj"),
        ("Numark Mixtrack Pro 3", "numark_mixtrack"),
        ("Traktor Kontrol S4 MK3", "traktor_kontrol"),
        ("Kontrol S2 MK2", "traktor_kontrol"),
        ("Kontrol X1 MK2", "traktor_kontrol"),
    ],
)
def test_dj_controllers_are_detected(name, expected):
    assert detect_controller_type(name) == expected
    assert expected != DEFAULT_CONTROLLER_TYPE


@pytest.mark.parametrize(
    "name",
    ["Komplete Kontrol S49 MK2", "Komplete Kontrol S88", "Blocks Pioneer"],
)
def test_dj_patterns_do_not_capture_other_devices(name):
    assert detect_controller_type(name) not in (
        "pioneer_ddj",
        "numark_mixtrack",
        "traktor_kontrol",
    )


def test_blocks_name_still_goes_to_roli():
    assert detect_controller_type("Blocks Pioneer") == "roli_seaboard"


def test_ddj_cc_actions():
    mapping = ControllerMapping.for_device("Pioneer DDJ-400")
    assert mapping.controller_type == "pioneer_ddj"
    assert mapping.map_cc(6, 0x1F, 64) == ControllerAction("crossfader", 6, 64)
    assert mapping.map_cc(0, 0x13, 100) == ControllerAction(
        "channel_fader", 0, 100
    )
    assert mapping.map_cc(1, 0x00, 64) == ControllerAction("tempo_fader", 1, 64)
    assert mapping.map_cc(0, 0x22, 65) == ControllerAction("jog", 0, 65)


def test_every_mapped_dj_cc_returns_an_action():
    for name in ("Pioneer DDJ-400", "Numark Mixtrack Pro 3"):
        mapping = ControllerMapping.for_device(name)
        assert mapping.cc_actions
        for cc, action in mapping.cc_actions.items():
            assert mapping.map_cc(0, cc, 64).action == action


def test_mixtrack_cc_actions():
    mapping = ControllerMapping.for_device("Numark Mixtrack Pro 3")
    assert mapping.map_cc(0, 7, 10) == ControllerAction("crossfader", 0, 10)
    assert mapping.map_cc(1, 0x11, 70) == ControllerAction("jog", 1, 70)


def test_traktor_kontrol_uses_base_actions():
    mapping = ControllerMapping.for_device("Traktor Kontrol S4 MK3")
    assert mapping.map_cc(0, 7, 90) == ControllerAction("volume", 0, 90)


def test_dj_types_have_no_mpe_preset():
    for name in ("pioneer_ddj", "numark_mixtrack", "traktor_kontrol"):
        assert name not in MPEControllerMapper.CONTROLLER_PRESETS
