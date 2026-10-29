import math

import pytest

from korshmer.midi.message_validator import (
    MAX_SYSEX_LENGTH,
    validate_cc,
    validate_channel,
    validate_note_message,
    validate_pitch_bend,
    validate_sysex,
)
from korshmer.midi_processing.midi_file_enhancement import SysExHandler

GOOD = b"\xf0\x7e\x00\xf7"


def test_sysex_valid():
    assert validate_sysex(GOOD) == GOOD
    assert validate_sysex(bytearray(GOOD)) == GOOD


@pytest.mark.parametrize(
    "bad",
    [b"", b"\xf0", b"\x00\x01\xf7", b"\xf0\x01", b"\xf0\x80\xf7"],
)
def test_sysex_malformed(bad):
    with pytest.raises(ValueError):
        validate_sysex(bad)


def test_sysex_too_long_and_wrong_type():
    with pytest.raises(ValueError):
        validate_sysex(b"\xf0" + b"\x01" * MAX_SYSEX_LENGTH + b"\xf7")
    with pytest.raises(TypeError):
        validate_sysex("text")


def test_add_sysex_rejects_bad_input_and_keeps_list_empty():
    h = SysExHandler()
    big = b"\xf0" + b"\x01" * 1000001 + b"\xf7"
    for t, d, exc in [
        (0.5, big, ValueError),
        (math.nan, GOOD, ValueError),
        (-5, GOOD, ValueError),
        (math.inf, GOOD, ValueError),
        (1.0, "text", TypeError),
    ]:
        with pytest.raises(exc):
            h.add_sysex(t, d)
    assert h.sysex_messages == []


def test_add_sysex_valid_is_stored_and_sorted():
    h = SysExHandler()
    h.add_sysex(2.0, GOOD)
    h.add_sysex(1.0, GOOD)
    assert [t for t, _ in h.sysex_messages] == [1.0, 2.0]
    assert h.get_sysex_at_time(1.0) == [GOOD]


def test_channel_range():
    assert validate_channel(0) == 0
    assert validate_channel(15) == 15
    for bad in (-1, 16, 1.5, math.nan, None, True, "1"):
        with pytest.raises(ValueError):
            validate_channel(bad)


def test_cc_and_pitch_bend_range():
    assert validate_cc(7, 127) == (7, 127)
    for c, v in [(128, 0), (-1, 0), (7, 128), (7, -1), (7, math.nan)]:
        with pytest.raises(ValueError):
            validate_cc(c, v)
    assert validate_pitch_bend(0) == 0
    assert validate_pitch_bend(16383) == 16383
    for bad in (-1, 16384, math.nan):
        with pytest.raises(ValueError):
            validate_pitch_bend(bad)


def test_note_message():
    assert validate_note_message(9, 60, 100) == (9, 60, 100)
    assert validate_note_message(0, 60) == (0, 60, 0)
    for args in [(16, 60, 1), (0, 128, 1), (0, 60, 128)]:
        with pytest.raises(ValueError):
            validate_note_message(*args)


def test_fluidsynth_wrapper_rejects_before_native_call():
    from korshmer.core.fluidsynth import FluidSynth

    calls = []

    class Lib:
        def __getattr__(self, name):
            return lambda *a: calls.append(name)

    fs = FluidSynth.__new__(FluidSynth)
    fs._lib = Lib()
    fs.synth = object()
    fs.audio_driver = None
    fs.settings = None
    with pytest.raises(ValueError):
        fs.note_on(16, 60, 100)
    with pytest.raises(ValueError):
        fs.cc(0, 200, 1)
    with pytest.raises(ValueError):
        fs.pitch_bend(0, 99999)
    assert calls == []
    fs.note_on(0, 60, 100)
    assert calls == ["fluid_synth_noteon"]
