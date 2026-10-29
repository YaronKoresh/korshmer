import pretty_midi
import pytest

from korshmer.midi_processing import MIDIBatchProcessor


def _song(path, pitch=60):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, pitch, 0.0, 1.0))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def _pitches(path):
    return [
        n.pitch
        for inst in pretty_midi.PrettyMIDI(path).instruments
        for n in inst.notes
    ]


@pytest.fixture(autouse=True)
def _processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("korshmer.midi_algorithms.DATA_DIR", str(tmp_path))


def test_operation_is_applied_to_every_file(tmp_path):
    first = _song(tmp_path / "a.mid", 60)
    second = _song(tmp_path / "b.mid", 70)
    batch = MIDIBatchProcessor()
    batch.add_operation("transpose", {"semitones": 5})
    out = batch.process_files([first, second])
    assert [_pitches(p) for p in out] == [[65], [75]]
    assert _pitches(first) == [60]


def test_operations_are_chained_in_order(tmp_path):
    src = _song(tmp_path / "a.mid", 60)
    batch = MIDIBatchProcessor()
    batch.add_operation("transpose", {"semitones": 2})
    batch.add_operation("transpose", {"semitones": 3})
    assert _pitches(batch.process_files([src])[0]) == [65]


def test_no_operations_keeps_paths(tmp_path):
    src = _song(tmp_path / "a.mid")
    assert MIDIBatchProcessor().process_files([src]) == [src]


def test_unknown_operation_and_bad_params_are_rejected():
    batch = MIDIBatchProcessor()
    with pytest.raises(ValueError):
        batch.add_operation("no_such_operation")
    with pytest.raises(ValueError):
        batch.add_operation("../x")
    with pytest.raises(ValueError):
        batch.add_operation("transpose", {"wrong": 1})
    assert batch.operations == []


def test_unreadable_file_is_skipped_and_others_processed(tmp_path):
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not midi")
    good = _song(tmp_path / "a.mid", 60)
    batch = MIDIBatchProcessor()
    batch.add_operation("transpose", {"semitones": 1})
    out = batch.process_files([str(bad), good])
    assert len(out) == 1
    assert _pitches(out[0]) == [61]
