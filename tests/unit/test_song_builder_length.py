import pytest

from korshmer.midi_generation.engine import (
    SONG_STRUCTURES,
    HarmonicContext,
    build_song,
    generate_bass,
    generate_drums,
)

PART_SETS = [
    ["melody", "bass", "drums"],
    ["bass", "drums"],
    ["drums", "bass", "chords", "melody", "pad"],
]


def _ends(pm):
    return {i.name: i.get_end_time() for i in pm.instruments}


@pytest.mark.parametrize("structure", list(SONG_STRUCTURES))
@pytest.mark.parametrize("parts", PART_SETS)
def test_roles_end_within_one_bar_of_each_other(structure, parts):
    pm = build_song("C", "Major", structure=structure, seed=1, parts=parts)
    bar = 60.0 / 120.0 * 4
    ends = _ends(pm)
    assert ends
    assert max(ends.values()) - min(ends.values()) <= bar * 1.05


def test_bass_does_not_outlast_the_song_length():
    sections = len(SONG_STRUCTURES["Pop (Verse/Chorus)"])
    pm = build_song("C", "Major", seed=1, parts=["melody", "bass", "drums"])
    song_end = sections * 8 * (60.0 / 120.0 * 4)
    assert _ends(pm)["Bass"] <= song_end + 0.01


def test_bass_following_late_drums_keeps_requested_bars():
    ctx = HarmonicContext(root=0, scale="Major", bpm=120.0)
    drums = generate_drums(ctx, "Rock", bars=4, seed=1, start_bar=24)
    bass = generate_bass(ctx, "Pop", drums, seed=1, bars=4, start_bar=24)
    bar = ctx.bar
    assert max(n.end for n in bass.notes) <= 28 * bar + 0.01
    assert min(n.start for n in bass.notes) >= 24 * bar - 0.01
