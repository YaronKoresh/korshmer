import random

import pretty_midi
import pytest

from korshmer import midi_algorithms as algo

BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
STEP = BEAT / 4
CHORDS = [48, 53, 55, 48, 48, 53, 55, 48]


def _song(shift):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    drums = pretty_midi.Instrument(0, is_drum=True)
    for bar in range(len(CHORDS)):
        for beat in range(4):
            start = bar * BAR + beat * BEAT + shift
            drums.notes.append(
                pretty_midi.Note(
                    100, 36 if beat % 2 == 0 else 38, start, start + 0.1
                )
            )
    pm.instruments.append(drums)
    chords = pretty_midi.Instrument(0)
    for bar, root in enumerate(CHORDS):
        for pitch in (root, root + 4, root + 7):
            chords.notes.append(
                pretty_midi.Note(
                    90, pitch, bar * BAR + shift, (bar + 1) * BAR + shift
                )
            )
    pm.instruments.append(chords)
    return pm


def _off_grid(times, shift, unit):
    worst = 0.0
    for t in times:
        pos = ((t - shift) / unit) % 1.0
        worst = max(worst, min(pos, 1.0 - pos) * unit)
    return worst


@pytest.mark.parametrize("shift", [0.07, 0.2, 1.0])
def test_ghost_notes_land_on_the_sixteenths_of_a_shifted_song(shift):
    random.seed(1)
    pm = algo.add_ghost_notes(_song(shift), prob=0.3, bpm=BPM)
    ghosts = [n.start for n in pm.instruments[0].notes if n.velocity < 45]
    assert ghosts
    assert _off_grid(ghosts, shift, STEP) < 0.005


@pytest.mark.parametrize("shift", [0.07, 0.2, 1.0])
def test_drum_fill_crash_sits_on_a_bar_line_of_a_shifted_song(shift):
    pm = algo.add_drum_fill(_song(shift), 4, BPM)
    crashes = [n.start + 0.05 for n in pm.instruments[0].notes if n.pitch == 49]
    assert crashes
    assert _off_grid(crashes, shift, BAR) < 0.005


@pytest.mark.parametrize("shift", [0.07, 1.0])
def test_followed_progression_ignores_the_shift(shift):
    scale = algo.SCALES["Major"]
    straight = algo._followed_progression(_song(0.0), BPM, 0, scale)
    shifted = algo._followed_progression(_song(shift), BPM, 0, scale)
    assert shifted == straight


@pytest.mark.parametrize("shift", [0.07, 1.0])
def test_melody_layers_follow_the_bar_lines_of_a_shifted_song(shift):
    random.seed(2)
    pm = algo.generate_melody_algo(_song(shift), "Major", 0, bpm=BPM)
    pads = [i for i in pm.instruments if i.name == "Layer 4: Atmosphere"][0]
    assert _off_grid([n.start for n in pads.notes], shift, BAR) < 0.005
    assert min(n.start for n in pads.notes) == pytest.approx(shift, abs=0.02)


@pytest.mark.parametrize("shift", [0.07, 0.2, 1.0])
def test_click_track_accents_the_bar_lines_of_a_shifted_song(shift):
    pm = algo.generate_click_track(_song(shift), bpm=BPM)
    click = pm.instruments[-1]
    assert _off_grid([n.start for n in click.notes], shift, BEAT) < 0.005
    accents = [n.start for n in click.notes if n.velocity == 110]
    assert _off_grid(accents, shift, BAR) < 0.005
    assert any(abs(a - shift) < 0.005 for a in accents)


def test_unshifted_song_keeps_the_grid_from_zero():
    pm = algo.generate_click_track(_song(0.0), bpm=BPM)
    assert pm.instruments[-1].notes[0].start == 0.0
    assert pm.instruments[-1].notes[0].velocity == 110
