import io

import pretty_midi
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from korshmer.midi_processors import get_chord_suggestion, process_humanize
from korshmer.ui.viewmodels import arranger_vm
from korshmer.ui.viewmodels.arranger_vm import ArrangerViewModel

EXPECTED_ERRORS = (EOFError, OSError, ValueError)
MTHD = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0"

FUZZ = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


def _valid_midi_bytes():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate((60, 64, 67)):
        inst.notes.append(
            pretty_midi.Note(
                velocity=90, pitch=pitch, start=i * 0.5, end=i * 0.5 + 0.4
            )
        )
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


VALID = _valid_midi_bytes()


def _parse(tmp_path, data):
    path = tmp_path / "fuzz.mid"
    path.write_bytes(data)
    return ArrangerViewModel().parse_midi(str(path), 120.0, 100.0, 10.0)


@FUZZ
@given(data=st.binary(max_size=200))
def test_parse_midi_random_bytes(tmp_path_factory, data):
    tmp_path = tmp_path_factory.mktemp("fuzz")
    try:
        _parse(tmp_path, data)
    except EXPECTED_ERRORS:
        pass


@FUZZ
@given(body=st.binary(max_size=60))
def test_parse_midi_valid_header_random_track(tmp_path_factory, body):
    tmp_path = tmp_path_factory.mktemp("fuzz")
    try:
        _parse(tmp_path, MTHD + b"MTrk" + body)
    except EXPECTED_ERRORS:
        pass


@FUZZ
@given(
    edits=st.lists(
        st.tuples(st.integers(0, len(VALID) - 1), st.integers(0, 255)),
        min_size=1,
        max_size=4,
    )
)
def test_parse_midi_mutated_valid_file(tmp_path_factory, edits):
    tmp_path = tmp_path_factory.mktemp("fuzz")
    data = bytearray(VALID)
    for idx, val in edits:
        data[idx] = val
    try:
        _parse(tmp_path, bytes(data))
    except EXPECTED_ERRORS:
        pass


def test_parse_midi_valid_file_still_parses(tmp_path):
    notes = _parse(tmp_path, VALID)
    assert sorted(n.pitch for n in notes) == [60, 64, 67]


@FUZZ
@given(data=st.binary(max_size=200))
def test_midi_processors_random_bytes(tmp_path_factory, data):
    path = tmp_path_factory.mktemp("fuzz") / "fuzz.mid"
    path.write_bytes(data)
    for func in (get_chord_suggestion, lambda p: process_humanize(p, 0.01, 5)):
        try:
            func(str(path))
        except EXPECTED_ERRORS:
            pass


def test_unexpected_exception_is_not_swallowed(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(arranger_vm, "load_pretty_midi", boom)
    with pytest.raises(RuntimeError):
        _parse(tmp_path, VALID)
