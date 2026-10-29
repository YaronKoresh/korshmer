import pretty_midi

from korshmer.constants import SCALES
from korshmer.midi_algorithms import generate_melody_algo

BPM = 120.0
BAR = 240.0 / BPM
CHORDS = {0: (0, 4, 7), 1: (2, 5, 9), 3: (5, 9, 0), 4: (7, 11, 2), 5: (9, 0, 4)}
LAYOUT = [0, 3, 4, 0, 5, 3, 1, 4]


def _song(layout=LAYOUT):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    piano = pretty_midi.Instrument(0)
    for bar, degree in enumerate(layout):
        for k, pc in enumerate(CHORDS[degree]):
            piano.notes.append(
                pretty_midi.Note(
                    90,
                    48 + pc + (12 if k == 0 else 0),
                    bar * BAR,
                    (bar + 1) * BAR,
                )
            )
    pm.instruments.append(piano)
    return pm


def _layer(pm, name):
    return next(i for i in pm.instruments if name in i.name)


def test_bass_layer_plays_the_root_of_each_bar_of_the_song():
    pm = generate_melody_algo(_song(), "Major", 0, bpm=BPM)
    bass = _layer(pm, "Bass")

    for note in bass.notes:
        bar = int(note.start // BAR)
        assert note.pitch % 12 == CHORDS[LAYOUT[bar]][0]


def test_pad_layer_holds_the_scale_triad_of_each_bar():
    pm = generate_melody_algo(_song(), "Major", 0, bpm=BPM)
    pad = _layer(pm, "Atmosphere")

    for bar, degree in enumerate(LAYOUT):
        pcs = {n.pitch % 12 for n in pad.notes if int(n.start // BAR) == bar}
        assert pcs == set(CHORDS[degree])


def test_arp_and_lead_notes_on_strong_beats_stay_in_the_bar_chord():
    pm = generate_melody_algo(_song(), "Major", 0, bpm=BPM)
    arp = _layer(pm, "Arp")

    for note in arp.notes:
        bar = int(note.start // BAR)
        assert note.pitch % 12 in CHORDS[LAYOUT[bar]]


def test_song_without_notes_keeps_the_default_progression():
    empty = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    pm = generate_melody_algo(empty, "Major", 0, bpm=BPM)
    bass = _layer(pm, "Bass")

    assert bass.notes
    first_bar = {n.pitch % 12 for n in bass.notes if n.start < BAR}
    assert first_bar == {SCALES["Major"][5]}
