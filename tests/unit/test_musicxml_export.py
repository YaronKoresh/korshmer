import xml.etree.ElementTree as ET

import pytest

from korshmer.midi_processing.score_notation import (
    Measure,
    MusicXMLExporter,
    ScoreNote,
)

DIVISIONS = 480


def _note(pitch, start, duration=1.0):
    return ScoreNote(
        pitch=pitch, duration=duration, start_time=start, velocity=80
    )


def _measure(notes, number=1, time_signature=(4, 4), key=0):
    return Measure(number, notes, time_signature, key, [])


def _export(measures):
    return ET.fromstring(MusicXMLExporter().export_to_musicxml(measures))


def _voice_lengths(measure_element):
    totals = {}
    for note in measure_element.findall("note"):
        if note.find("chord") is not None:
            continue
        voice = note.findtext("voice")
        totals[voice] = totals.get(voice, 0) + int(note.findtext("duration"))
    return totals


def test_first_measure_declares_divisions_key_time_and_clef():
    root = _export([_measure([_note(60, 0.0)], key=2)])
    attributes = root.find("part/measure/attributes")
    assert attributes.findtext("divisions") == str(DIVISIONS)
    assert attributes.findtext("key/fifths") == "2"
    assert attributes.findtext("time/beats") == "4"
    assert attributes.findtext("time/beat-type") == "4"
    assert attributes.findtext("clef/sign") == "G"


def test_chord_notes_share_one_position():
    root = _export([_measure([_note(60, 0.0), _note(64, 0.0), _note(67, 0.0)])])
    measure = root.find("part/measure")
    notes = [n for n in measure.findall("note") if n.find("rest") is None]
    assert [n.find("chord") is not None for n in notes] == [False, True, True]
    assert _voice_lengths(measure) == {"1": 4 * DIVISIONS}


def test_gap_before_a_note_becomes_a_rest():
    root = _export([_measure([_note(72, 2.0)])])
    notes = root.find("part/measure").findall("note")
    assert notes[0].find("rest") is not None
    assert notes[0].findtext("duration") == str(2 * DIVISIONS)
    assert notes[1].findtext("pitch/step") == "C"
    assert notes[1].findtext("pitch/octave") == "5"


def test_empty_measure_is_one_full_rest():
    root = _export([_measure([])])
    notes = root.find("part/measure").findall("note")
    assert len(notes) == 1
    assert notes[0].find("rest") is not None
    assert notes[0].findtext("duration") == str(4 * DIVISIONS)


@pytest.mark.parametrize(
    ("time_signature", "beats"), [((3, 4), 3), ((6, 8), 3), ((2, 2), 4)]
)
def test_every_measure_adds_up_to_its_time_signature(time_signature, beats):
    notes = [_note(60, 0.0, 0.5), _note(62, 1.0, 0.5), _note(64, 2.0, 0.5)]
    root = _export([_measure(notes, time_signature=time_signature)])
    assert _voice_lengths(root.find("part/measure")) == {
        "1": round(beats * DIVISIONS)
    }


def test_overlapping_notes_go_to_a_second_voice_with_a_backup():
    root = _export([_measure([_note(60, 0.0, 4.0), _note(67, 1.0, 1.0)])])
    measure = root.find("part/measure")
    assert _voice_lengths(measure) == {
        "1": 4 * DIVISIONS,
        "2": 4 * DIVISIONS,
    }
    backup = measure.find("backup")
    assert backup.findtext("duration") == str(4 * DIVISIONS)


def test_note_running_past_the_bar_is_cut_at_the_bar_line():
    root = _export([_measure([_note(60, 3.0, 3.0)])])
    assert _voice_lengths(root.find("part/measure")) == {"1": 4 * DIVISIONS}


def test_note_starting_after_the_bar_is_not_written_in_it():
    root = _export([_measure([_note(60, 4.0)])])
    names = [n.find("rest") is not None for n in root.iter("note")]
    assert names == [True]


def test_note_types_and_dots_follow_the_duration():
    root = _export(
        [
            _measure(
                [
                    _note(60, 0.0, 1.5),
                    _note(62, 1.5, 0.5),
                    _note(64, 2.0, 2.0),
                ]
            )
        ]
    )
    notes = root.find("part/measure").findall("note")
    assert [n.findtext("type") for n in notes] == [
        "quarter",
        "eighth",
        "half",
    ]
    assert notes[0].find("dot") is not None
    assert notes[1].find("dot") is None


def test_flat_key_spells_black_keys_with_flats():
    root = _export([_measure([_note(70, 0.0)], key=-2)])
    pitch = root.find("part/measure/note/pitch")
    assert pitch.findtext("step") == "B"
    assert pitch.findtext("alter") == "-1"


def test_sharp_key_spells_black_keys_with_sharps():
    root = _export([_measure([_note(70, 0.0)], key=2)])
    pitch = root.find("part/measure/note/pitch")
    assert pitch.findtext("step") == "A"
    assert pitch.findtext("alter") == "1"


def test_later_measures_repeat_attributes_only_when_they_change():
    first = _measure([_note(60, 0.0)], number=1)
    second = _measure([_note(60, 0.0)], number=2)
    third = _measure([_note(60, 0.0)], number=3, time_signature=(3, 4))
    root = _export([first, second, third])
    measures = root.findall("part/measure")
    assert measures[1].find("attributes/time") is None
    assert measures[1].find("attributes/divisions") is None
    assert measures[2].findtext("attributes/time/beats") == "3"


@pytest.mark.parametrize("key", [8, -8, 1.5, "C", True, None])
def test_invalid_key_signature_is_rejected(key):
    with pytest.raises(ValueError, match="Key signature"):
        MusicXMLExporter().export_to_musicxml([_measure([], key=key)])


def test_pitch_octave_and_semitone_of_every_note_are_kept():
    pitches = [0, 21, 60, 61, 127]
    notes = [_note(p, float(i % 4), 1.0) for i, p in enumerate(pitches)]
    root = _export([_measure(notes)])
    seen = []
    for note in root.iter("note"):
        pitch = note.find("pitch")
        if pitch is None:
            continue
        step = "CDEFGAB".index(pitch.findtext("step"))
        semitone = (0, 2, 4, 5, 7, 9, 11)[step] + int(
            pitch.findtext("alter") or 0
        )
        seen.append(12 * (int(pitch.findtext("octave")) + 1) + semitone)
    assert sorted(seen) == sorted(pitches)


def test_measure_without_changes_has_no_empty_attributes_element():
    first = _measure([_note(60, 0.0)], number=1)
    second = _measure([_note(60, 0.0)], number=2)
    root = _export([first, second])
    assert root.findall("part/measure")[1].find("attributes") is None
