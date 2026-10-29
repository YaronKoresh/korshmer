import bisect
import copy

import numpy as np
import pretty_midi
import pytest

from korshmer.midi_algorithms import generate_pad_layer, generate_simple_bass
from korshmer.midi_generation import ComposeOptions, compose, song_match

BPM = 100.0
BAR = 240.0 / BPM
SEED = 3


def _options(**changes):
    values = {
        "root": "D",
        "scale": "Minor",
        "bpm": BPM,
        "minutes": 0.5,
        "style": "Pop",
        "structure": "Verse / Chorus",
        "mood": "Auto",
        "seed": SEED,
        "roles": ("drums", "bass", "chords", "pad", "melody"),
        "texture_batches": 0,
        "mode": "New song",
        "match": "Strong",
        "existing_tracks": 0,
    }
    values.update(changes)
    return ComposeOptions(**values)


def _song():
    comp = compose(_options())
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    pm.instruments = comp.instruments
    return pm


def _shifted(pm, seconds):
    out = copy.deepcopy(pm)
    for inst in out.instruments:
        for note in inst.notes:
            note.start += seconds
            note.end += seconds
    return out


def _aligned_share(layers, song):
    starts = sorted(
        n.start for i in song.instruments if not i.is_drum for n in i.notes
    )
    hit = total = 0
    for inst in layers:
        if inst.is_drum:
            continue
        for note in inst.notes:
            total += 1
            at = bisect.bisect_left(starts, note.start)
            gap = min(
                abs(starts[k] - note.start)
                for k in (at - 1, at)
                if 0 <= k < len(starts)
            )
            hit += gap < 0.03
    return hit / total


def test_offset_of_a_song_that_starts_on_a_bar_is_zero():
    assert song_match.estimate_offset(_song(), BPM) == 0.0


@pytest.mark.parametrize("seconds", [0.15, 0.3, 0.45])
def test_offset_is_found_for_a_song_that_starts_between_bar_lines(seconds):
    found = song_match.estimate_offset(_shifted(_song(), seconds), BPM)
    assert found == pytest.approx(seconds, abs=0.02)


def test_offset_of_an_empty_song_is_zero():
    assert song_match.estimate_offset(pretty_midi.PrettyMIDI(), BPM) == 0.0


@pytest.mark.parametrize("seconds", [0.15, 0.3, 0.45])
def test_layers_land_on_the_beat_of_a_shifted_song(seconds):
    base = _song()
    straight = compose(
        _options(mode="Layer over editor", seed=9, style="Pop"),
        existing=base,
    )
    song = _shifted(base, seconds)
    layered = compose(
        _options(mode="Layer over editor", seed=9, style="Pop"),
        existing=song,
    )
    reference = _aligned_share(straight.instruments, base)
    assert _aligned_share(layered.instruments, song) >= reference - 0.05


def test_layers_keep_the_bar_count_of_the_shifted_song():
    base = _song()
    straight = compose(
        _options(mode="Layer over editor", seed=9), existing=base
    )
    layered = compose(
        _options(mode="Layer over editor", seed=9),
        existing=_shifted(base, 0.3),
    )
    assert layered.total_bars == straight.total_bars


def test_simple_bass_and_pad_follow_a_shifted_song():
    np.random.seed(1)
    shift = 0.3
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    piano = pretty_midi.Instrument(0)
    for index, root in enumerate([48, 53, 55, 48]):
        for pitch in (root, root + 4, root + 7):
            piano.notes.append(
                pretty_midi.Note(
                    90,
                    pitch,
                    index * BAR + shift,
                    (index + 1) * BAR + shift,
                )
            )
        piano.notes.append(
            pretty_midi.Note(
                70, root + 12, index * BAR + shift, index * BAR + shift + 0.3
            )
        )
        piano.notes.append(
            pretty_midi.Note(
                70,
                root + 12,
                index * BAR + shift + BAR / 4,
                index * BAR + shift + BAR / 4 + 0.3,
            )
        )
    pm.instruments.append(piano)
    pm = generate_simple_bass(pm, bpm=BPM)
    pm = generate_pad_layer(pm, bpm=BPM)
    for inst in pm.instruments[1:]:
        assert min(n.start for n in inst.notes) == pytest.approx(
            shift, abs=0.02
        )
