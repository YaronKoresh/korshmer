import pretty_midi
import pytest

from korshmer import midi_algorithms
from korshmer.midi_processors import (
    process_arpeggiator_advanced,
)


@pytest.fixture(autouse=True)
def processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path / "data"))


def _write(path, notes, tempo=120.0, program=0, is_drum=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program, is_drum=is_drum)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(90, pitch, start, end))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def _chord_song(tmp_path):
    notes = [(60, 0.0, 2.0), (64, 0.0, 2.0), (67, 0.0, 2.0)]
    return _write(tmp_path / "chord.mid", notes, tempo=120.0)


def test_advanced_arpeggio_spreads_a_chord_over_its_length(tmp_path):
    out = process_arpeggiator_advanced(_chord_song(tmp_path), "up", "1/16")
    notes = pretty_midi.PrettyMIDI(out).instruments[0].notes
    assert [n.pitch for n in notes[:6]] == [60, 64, 67, 60, 64, 67]
    assert notes[1].start == pytest.approx(0.125, abs=1e-3)
    assert len(notes) == 16
    assert max(n.end for n in notes) <= 2.0 + 1e-6


def test_advanced_arpeggio_down_pattern_and_rate(tmp_path):
    out = process_arpeggiator_advanced(_chord_song(tmp_path), "Down", "1/8")
    notes = pretty_midi.PrettyMIDI(out).instruments[0].notes
    assert [n.pitch for n in notes[:3]] == [67, 64, 60]
    assert notes[1].start == pytest.approx(0.25, abs=1e-3)


def test_advanced_arpeggio_keeps_single_notes_and_drums(tmp_path):
    single = _write(tmp_path / "line.mid", [(60, 0.0, 0.5), (62, 0.5, 1.0)])
    out = process_arpeggiator_advanced(single, "up", "1/16")
    notes = pretty_midi.PrettyMIDI(out).instruments[0].notes
    assert [(n.pitch, round(n.start, 3)) for n in notes] == [
        (60, 0.0),
        (62, 0.5),
    ]
    drums = _write(
        tmp_path / "kit.mid",
        [(36, 0.0, 0.1), (42, 0.0, 0.1)],
        is_drum=True,
    )
    out = process_arpeggiator_advanced(drums, "up", "1/16")
    assert len(pretty_midi.PrettyMIDI(out).instruments[0].notes) == 2


@pytest.mark.parametrize("pattern", ["sideways", None, 3])
def test_advanced_arpeggio_rejects_unknown_patterns(tmp_path, pattern):
    with pytest.raises(ValueError, match="pattern"):
        process_arpeggiator_advanced(_chord_song(tmp_path), pattern, "1/16")


@pytest.mark.parametrize("rate", ["1/3", "fast", None])
def test_advanced_arpeggio_rejects_unknown_rates(tmp_path, rate):
    with pytest.raises(ValueError, match="rate"):
        process_arpeggiator_advanced(_chord_song(tmp_path), "up", rate)
