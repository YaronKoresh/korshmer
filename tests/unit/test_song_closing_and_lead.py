import pretty_midi
import pytest

from korshmer.midi_generation import ComposeOptions, compose, engine, song_match
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 120.0
BAR = 60.0 / BPM * 4
EPS = 1e-6
SEEDS = (1, 2, 3, 4)
FULL_BAND = ("drums", "bass", "chords", "pad", "melody")


def _compose(style, seed, roles, mode="New song", existing=None):
    return compose(
        ComposeOptions(
            root="A",
            scale="Minor",
            bpm=BPM,
            minutes=2.0,
            style=style,
            seed=seed,
            roles=roles,
            texture_batches=0,
            mode=mode,
            match="Strong",
        ),
        existing=existing,
    )


def _midi(comp):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    pm.instruments = comp.instruments
    return pm


def _voice(name, pitches, start):
    inst = pretty_midi.Instrument(program=0, name=name)
    for k, pitch in enumerate(pitches):
        begin = start + k * 0.5
        inst.notes.append(pretty_midi.Note(90, pitch, begin, begin + 0.4))
    return inst


def _rhythm(notes, bar):
    return tuple(
        round((n.start - bar * BAR) / BAR * 16)
        for n in notes
        if bar * BAR - EPS <= n.start < (bar + 1) * BAR - EPS
    )


def test_find_lead_pools_the_numbered_voices_of_one_instrument():
    pm = pretty_midi.PrettyMIDI()
    pm.instruments = [
        _voice("Lead", [72, 74, 76, 77] * 20, 0.0),
        _voice("Lead 2", [84, 86, 88] * 4, 100.0),
        _voice("Bass", [40] * 30, 0.0),
    ]
    assert len(song_match.find_lead(pm)) == 92


def test_find_lead_keeps_unnamed_instruments_apart():
    pm = pretty_midi.PrettyMIDI()
    pm.instruments = [
        _voice("", [72, 74, 76, 77] * 20, 0.0),
        _voice("", [84, 86, 88] * 4, 100.0),
    ]
    assert len(song_match.find_lead(pm)) == 12


@pytest.mark.parametrize("style", SONG_STYLES)
def test_find_lead_hears_the_whole_generated_melody(style):
    for seed in SEEDS:
        comp = _compose(style, seed, ("melody",))
        total = sum(len(inst.notes) for inst in comp.instruments)
        assert len(song_match.find_lead(_midi(comp))) == total


@pytest.mark.parametrize("style", SONG_STYLES)
def test_layer_over_a_generated_song_harmonises_the_whole_melody(style):
    for seed in (7, 8):
        song = _compose(style, seed, FULL_BAND)
        pm = _midi(song)
        lead = song_match.find_lead(pm)
        layer = _compose(
            style,
            seed,
            ("melody",),
            mode="Layer over editor",
            existing=pm,
        )
        harmony = sum(len(inst.notes) for inst in layer.instruments)
        assert len(lead) >= 100
        assert harmony >= 0.9 * len(lead)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_the_generated_melody_is_a_single_line(style):
    for seed in SEEDS:
        for inst in _compose(style, seed, ("melody",)).instruments:
            notes = sorted(inst.notes, key=lambda n: n.start)
            for note, following in zip(notes, notes[1:], strict=False):
                assert note.end <= following.start + EPS


@pytest.mark.parametrize("style", SONG_STYLES)
def test_song_closes_on_five_then_one(style):
    for seed in SEEDS:
        comp = _compose(style, seed, ("chords",))
        final = comp.sections[-1]
        prog = final.progression
        last = (comp.start_bar + final.start_bar + final.bars - 1) % len(prog)
        assert prog[last] == 0
        assert prog[last - 1] == 4


def test_build_song_choruses_share_one_hook_and_the_last_bar_closes():
    pm = engine.build_song(
        "A",
        "Minor",
        style="Pop",
        bpm=BPM,
        seed=3,
        parts=["melody"],
        bars_per_section=8,
    )
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    choruses = {_rhythm(notes, bar) for bar in (16, 32, 48)}
    assert len(choruses) == 1
    assert _rhythm(notes, 0) == _rhythm(notes, 16)
    assert _rhythm(notes, 8) != _rhythm(notes, 16)
    preset = engine.PROGRESSION_PRESETS[engine._STYLES["Pop"][0]]
    ctx = engine.HarmonicContext(
        root=9, scale="Minor", bpm=BPM, progression=list(preset)
    )
    final_root = ctx.degree_pitch(ctx.chord_at(63), 4)
    assert (notes[-1].pitch - final_root) % 12 == 0


def test_build_song_hook_follows_the_seed():
    def chorus(seed):
        pm = engine.build_song(
            "A",
            "Minor",
            style="Pop",
            bpm=BPM,
            seed=seed,
            parts=["melody"],
            bars_per_section=8,
        )
        notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
        return tuple(
            (n.pitch, round(n.start / BAR * 16))
            for n in notes
            if 16 * BAR <= n.start < 17 * BAR
        )

    assert chorus(3) == chorus(3)
    assert len({chorus(seed) for seed in range(8)}) >= 5
