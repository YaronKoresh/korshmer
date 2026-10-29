import pytest

from korshmer.midi_processing.midi_file_enhancement import MIDI2Support


@pytest.fixture
def midi2():
    return MIDI2Support()


def test_note_off_velocity_is_scaled_to_midi2_range(midi2):
    (event,) = midi2.convert_midi1_to_midi2(
        [{"type": "note_off", "velocity": 64}]
    )
    assert event["velocity"] == 64 * 512


def test_note_off_velocity_round_trips(midi2):
    original = [{"type": "note_off", "velocity": 127}]
    back = midi2.convert_midi2_to_midi1(midi2.convert_midi1_to_midi2(original))
    assert back == original


def test_out_of_range_velocity_stays_within_midi2_range(midi2):
    (event,) = midi2.convert_midi1_to_midi2(
        [{"type": "note_on", "velocity": 200}]
    )
    assert event["velocity"] == 127 * 512


def test_out_of_range_midi2_values_stay_within_midi1_range(midi2):
    events = midi2.convert_midi2_to_midi1(
        [
            {"type": "note_on", "velocity": 70000},
            {"type": "control_change", "value": 70000},
            {"type": "pitch_bend", "value": 10**7},
        ]
    )
    assert events[0]["velocity"] == 127
    assert events[1]["value"] == 127
    assert events[2]["value"] == 16383


def test_quiet_midi2_note_on_does_not_become_note_off(midi2):
    (event,) = midi2.convert_midi2_to_midi1(
        [{"type": "note_on", "velocity": 100}]
    )
    assert event["velocity"] == 1


def test_midi2_note_on_zero_velocity_stays_zero(midi2):
    (event,) = midi2.convert_midi2_to_midi1(
        [{"type": "note_on", "velocity": 0}]
    )
    assert event["velocity"] == 0


@pytest.mark.parametrize(
    "event",
    [
        {"type": "note_on"},
        {"type": "note_off", "velocity": None},
        {"type": "pitch_bend"},
        {"type": "control_change", "value": "x"},
        {"type": "note_on", "velocity": float("nan")},
    ],
)
def test_missing_or_non_numeric_values_raise_value_error(midi2, event):
    with pytest.raises(ValueError):
        midi2.convert_midi1_to_midi2([event])
    with pytest.raises(ValueError):
        midi2.convert_midi2_to_midi1([event])


def test_valid_events_round_trip(midi2):
    original = [
        {"type": "note_on", "velocity": 127},
        {"type": "pitch_bend", "value": -8192},
        {"type": "pitch_bend", "value": 8191},
        {"type": "control_change", "value": 127},
        {"type": "program_change", "program": 3},
    ]
    back = midi2.convert_midi2_to_midi1(midi2.convert_midi1_to_midi2(original))
    for got, want in zip(back, original, strict=True):
        for key, value in want.items():
            assert got[key] == value
