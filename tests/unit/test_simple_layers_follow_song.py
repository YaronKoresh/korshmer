import numpy as np
import pretty_midi
import pytest

from korshmer.midi_algorithms import generate_pad_layer, generate_simple_bass

CHORDS = {"C": (0, 4, 7), "F": (5, 9, 0), "G": (7, 11, 2), "Am": (9, 0, 4)}
LAYOUT = ["C", "F", "G", "C", "Am", "F", "G", "G"]


def _song(bpm):
    bar = 240.0 / bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(0)
    for index, name in enumerate(LAYOUT):
        for k, pc in enumerate(CHORDS[name]):
            pitch = 48 + pc + (12 if k == 0 else 0)
            piano.notes.append(
                pretty_midi.Note(90, pitch, index * bar, (index + 1) * bar)
            )
    pm.instruments.append(piano)
    return pm, bar


def _layer(pm, name):
    return next(i for i in pm.instruments if name in i.name)


@pytest.mark.parametrize("bpm", [90.0, 120.0, 140.0])
def test_bass_plays_the_root_of_every_bar_at_any_tempo(bpm):
    np.random.seed(1)
    pm, bar = _song(bpm)
    pm = generate_simple_bass(pm, bpm=bpm)
    bass = _layer(pm, "Bass")

    by_bar = {}
    for n in bass.notes:
        by_bar.setdefault(int((n.start + 1e-6) // bar), set()).add(n.pitch % 12)
    assert sorted(by_bar) == list(range(len(LAYOUT)))
    for index, name in enumerate(LAYOUT):
        assert by_bar[index] <= {CHORDS[name][0]}


@pytest.mark.parametrize("bpm", [90.0, 140.0])
def test_pad_holds_the_triad_of_every_bar_at_any_tempo(bpm):
    np.random.seed(1)
    pm, bar = _song(bpm)
    pm = generate_pad_layer(pm, bpm=bpm)
    pad = _layer(pm, "Pad")

    for index, name in enumerate(LAYOUT):
        pcs = {
            n.pitch % 12
            for n in pad.notes
            if int((n.start + 1e-6) // bar) == index
        }
        assert pcs == set(CHORDS[name])


def test_layers_end_with_the_song():
    np.random.seed(1)
    pm, bar = _song(100.0)
    end = pm.get_end_time()
    pm = generate_simple_bass(pm, bpm=100.0)
    pm = generate_pad_layer(pm, bpm=100.0)

    for name in ("Bass", "Pad"):
        assert max(n.end for n in _layer(pm, name).notes) <= end + 1e-6
