import io

import pretty_midi
import pytest

from korshmer.midi_generation import engine
from korshmer.midi_generation.composer import (
    ROLE_KEYS,
    ComposeOptions,
    compose,
)

EPS = 1e-9


def _overlaps(inst):
    by_pitch = {}
    for n in inst.notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    found = 0
    for notes in by_pitch.values():
        notes.sort(key=lambda n: n.start)
        found += sum(
            1
            for a, b in zip(notes, notes[1:], strict=False)
            if b.start < a.end - EPS
        )
    return found


def _inst(*spans):
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in spans:
        inst.notes.append(pretty_midi.Note(80, pitch, start, end))
    return inst


def _round_trip(instruments):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    pm.instruments = instruments
    buffer = io.BytesIO()
    pm.write(buffer)
    buffer.seek(0)
    return pretty_midi.PrettyMIDI(buffer).instruments


def test_trim_ends_a_note_where_the_next_one_of_the_same_pitch_starts():
    inst = _inst((60, 0.0, 1.0), (60, 0.5, 1.5), (62, 0.2, 2.0))
    engine.trim_overlaps(inst)
    spans = sorted((n.pitch, n.start, n.end) for n in inst.notes)
    assert spans == [(60, 0.0, 0.5), (60, 0.5, 1.5), (62, 0.2, 2.0)]


def test_trim_keeps_notes_that_do_not_overlap_unchanged():
    inst = _inst((60, 0.0, 0.5), (60, 0.5, 1.0), (64, 0.0, 3.0))
    before = sorted((n.pitch, n.start, n.end) for n in inst.notes)
    engine.trim_overlaps(inst)
    assert sorted((n.pitch, n.start, n.end) for n in inst.notes) == before


def test_trim_merges_notes_of_the_same_pitch_that_start_together():
    inst = _inst((60, 1.0, 1.5), (60, 1.0, 2.5), (60, 3.0, 3.5))
    engine.trim_overlaps(inst)
    spans = sorted((n.pitch, n.start, n.end) for n in inst.notes)
    assert spans == [(60, 1.0, 2.5), (60, 3.0, 3.5)]


def test_trim_handles_a_chain_of_overlapping_notes_and_an_empty_instrument():
    chain = _inst(*((60, i * 0.25, i * 0.25 + 0.96) for i in range(8)))
    engine.trim_overlaps(chain)
    assert _overlaps(chain) == 0
    assert all(n.end - n.start == pytest.approx(0.25) for n in chain.notes[:-1])
    empty = _inst()
    engine.trim_overlaps(empty)
    assert empty.notes == []


@pytest.mark.parametrize("style", ["Pop", "Rock", "Jazz", "EDM", "Hip-Hop"])
@pytest.mark.parametrize("seed", range(3))
def test_composed_instruments_never_overlap_notes_of_one_pitch(style, seed):
    comp = compose(
        ComposeOptions(
            root="C",
            scale="Minor",
            bpm=120.0,
            minutes=6,
            style=style,
            seed=seed,
            roles=ROLE_KEYS,
            texture_batches=0,
        )
    )
    for inst in comp.instruments:
        assert _overlaps(inst) == 0, inst.name


def test_saved_midi_keeps_every_composed_note_length():
    comp = compose(
        ComposeOptions(
            root="C",
            scale="Minor",
            bpm=120.0,
            minutes=3,
            seed=1,
            roles=("chords", "pad", "melody"),
            texture_batches=0,
        )
    )
    back = _round_trip(comp.instruments)
    for original, loaded in zip(comp.instruments, back, strict=True):
        a = sorted((n.start, n.pitch, n.end) for n in original.notes)
        b = sorted((n.start, n.pitch, n.end) for n in loaded.notes)
        assert len(a) == len(b)
        for (_s1, _p1, e1), (_s2, _p2, e2) in zip(a, b, strict=True):
            assert e2 == pytest.approx(e1, abs=0.01)


def test_build_song_instruments_never_overlap_notes_of_one_pitch():
    pm = engine.build_song(
        "C",
        "Minor",
        parts=["drums", "bass", "chords", "melody", "pad", "arp", "counter"],
        seed=2,
    )
    assert pm.instruments
    for inst in pm.instruments:
        assert _overlaps(inst) == 0, inst.name
