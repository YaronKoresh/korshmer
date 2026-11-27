import statistics

import pytest

from korshmer.midi_generation.composer import ComposeOptions, compose
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 110.0
BAR = 240.0 / BPM
SEEDS = range(1, 7)
LEAP = 5


def _edge_moves(style, seed):
    song = compose(
        ComposeOptions(
            "A",
            "Minor",
            bpm=BPM,
            minutes=1.0,
            style=style,
            seed=seed,
            texture_batches=0,
        )
    )
    lead = sorted(
        (
            note
            for inst in song.instruments
            if inst.name == "Melody" or inst.name.startswith("Melody ")
            if "answer" not in inst.name
            for note in inst.notes
        ),
        key=lambda note: note.start,
    )
    moves = []
    for earlier, later in zip(lead, lead[1:], strict=False):
        if later.start - earlier.end > BAR / 2:
            continue
        if int(earlier.start / BAR + 1e-6) != int(later.start / BAR + 1e-6):
            moves.append(abs(later.pitch - earlier.pitch))
    return moves


@pytest.mark.parametrize("style", SONG_STYLES)
def test_the_tune_moves_on_into_the_next_bar_instead_of_jumping(style):
    moves = [move for seed in SEEDS for move in _edge_moves(style, seed)]
    assert len(moves) >= 100
    leaps = sum(move >= LEAP for move in moves) / len(moves)
    assert leaps <= 0.42
    assert statistics.mean(moves) <= 4.6


def test_a_bar_placed_a_third_away_never_cancels_the_chord_change():
    from korshmer.midi_generation.engine import HarmonicContext, _lead_for

    ctx = HarmonicContext(root=9, scale="Minor", bpm=BPM)
    lead = _lead_for(ctx, 70)
    first = lead.place(0, [0, 2, 4], lead.mid)
    second = lead.place(2, [0, 2, 4], first[-1])
    assert (second[0] - first[0]) % 7 != 0
