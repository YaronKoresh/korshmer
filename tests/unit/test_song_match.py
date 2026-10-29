import pretty_midi
import pytest

from korshmer.constants import SCALES
from korshmer.midi_generation import song_match
from korshmer.midi_generation.composer import MODES, ComposeOptions, compose

BPM = 120.0
BAR = 240.0 / BPM
ROOT = 0
MAJOR = SCALES["Major"]

C, F, G, AM = (0, 4, 7), (5, 9, 0), (7, 11, 2), (9, 0, 4)
CHORDS = {0: C, 3: F, 4: G, 5: AM}
LAYOUT = [0, 3, 4, 0] * 2 + [5, 3, 4, 4] * 2 + [0, 3, 4, 0] * 2
LOUD = range(8, 16)


def _song(layout=LAYOUT, loud=LOUD):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    piano = pretty_midi.Instrument(0)
    drums = pretty_midi.Instrument(0, is_drum=True)
    for bar, degree in enumerate(layout):
        for k, pc in enumerate(CHORDS[degree]):
            octave = 48 + (12 if k == 0 else 0)
            piano.notes.append(
                pretty_midi.Note(
                    100 if bar in loud else 40,
                    octave + pc,
                    bar * BAR,
                    (bar + 1) * BAR,
                )
            )
        if bar in loud:
            for beat in range(8):
                drums.notes.append(
                    pretty_midi.Note(
                        110,
                        38,
                        bar * BAR + beat * BAR / 8,
                        bar * BAR + beat * BAR / 8 + 0.05,
                    )
                )
    pm.instruments += [piano, drums]
    return pm


def test_chord_degrees_follow_the_song():
    analysis = song_match.analyze_song(_song(), BPM, ROOT, MAJOR)

    assert analysis.bars == len(LAYOUT)
    assert analysis.degrees == LAYOUT


def test_parts_are_named_by_loudness():
    analysis = song_match.analyze_song(_song(), BPM, ROOT, MAJOR)
    names = [(name, start, bars) for name, start, bars in analysis.sections]

    assert names[0][0] == "Verse" or names[0][0] == "Intro"
    chorus = [s for s in names if s[0] == "Chorus"]
    assert chorus and chorus[0][1] == 8 and chorus[0][2] == 8
    assert sum(bars for _n, _s, bars in names) == analysis.bars


def test_progression_per_match_level():
    analysis = song_match.analyze_song(_song(), BPM, ROOT, MAJOR)

    assert song_match.progression_for(analysis, "Strong") == LAYOUT
    assert song_match.progression_for(analysis, "Weak") is None
    medium = song_match.progression_for(analysis, "Medium")
    assert len(medium) == len(LAYOUT)
    assert medium[0::2] == medium[1::2]


def test_unknown_match_level_is_rejected():
    with pytest.raises(ValueError, match="match"):
        song_match.validate_match("Huge")


def test_auto_is_a_valid_match_and_the_default():
    assert song_match.validate_match(song_match.AUTO_MATCH) == "Auto"
    assert song_match.DEFAULT_MATCH == song_match.AUTO_MATCH
    assert ComposeOptions(root="C", scale="Major").match == "Auto"


def _heard(degrees):
    return song_match.SongAnalysis(len(degrees), degrees=degrees)


@pytest.mark.parametrize(
    ("degrees", "level"),
    [
        ([0, 3, 4, 0, 5, 3, 4, 4], "Strong"),
        ([0, 3, 0, 3, 4, 4, 0, 0], "Strong"),
        ([0, 3, None, 4, 0], "Strong"),
        ([None, None, 0, 3, 4, 0, None, None], "Strong"),
        ([0, 3, 0, 0, 4, 4, 0, 0], "Medium"),
        ([0, 0, 3, 3, 4, 4, 0, 0], "Medium"),
        ([0] * 16, "Medium"),
        ([0, 0, None, 4, 4, None, 3, 3], "Medium"),
        ([0, None, None, None, 3], "Medium"),
        ([0, None, None, None, None, 3], "Weak"),
        ([None, None, None, 4, None, None, None, None], "Weak"),
        ([None, None, None, None], "Weak"),
        ([], "Weak"),
    ],
)
def test_auto_match_reads_how_much_plays_and_how_fast_chords_move(
    degrees, level
):
    assert song_match.auto_match(_heard(degrees)) == level


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [
        ([0, 3, 4, 0], [0, 3, 4, 0]),
        ([0, 3, 0, 0, 4, 4, 0, 0], [0, 0, 0, 0, 4, 4, 0, 0]),
        ([None] * 4, None),
    ],
)
def test_auto_progression_follows_the_level_it_picks(degrees, expected):
    analysis = _heard(degrees)

    assert (
        song_match.progression_for(analysis, song_match.AUTO_MATCH) == expected
    )


def test_auto_match_layer_equals_the_strong_layer_when_chords_move_every_bar():
    def notes(composition):
        return [
            (inst.name, [(n.start, n.pitch) for n in inst.notes])
            for inst in composition.instruments
        ]

    assert notes(_layer(song_match.AUTO_MATCH)) == notes(_layer("Strong"))
    assert notes(_layer(song_match.AUTO_MATCH)) != notes(_layer("Medium"))


def _layer(match, roles=("bass", "chords"), **kw):
    options = ComposeOptions(
        root="C",
        scale="Major",
        bpm=BPM,
        seed=3,
        roles=roles,
        texture_batches=0,
        mode="Layer over editor",
        match=match,
        **kw,
    )
    return compose(options, existing=_song())


def _bass_roots_by_bar(composition):
    out = {}
    for inst in composition.instruments:
        if "Bass" not in inst.name:
            continue
        for n in inst.notes:
            out.setdefault(int(n.start // BAR), []).append(n.pitch % 12)
    return out


def test_layer_mode_is_registered_and_covers_the_song():
    assert "Layer over editor" in MODES
    composition = _layer("Strong")

    assert composition.start_bar == 0
    assert composition.total_bars == len(LAYOUT)
    end = max(n.end for i in composition.instruments for n in i.notes)
    assert end <= len(LAYOUT) * BAR + 1e-6


def test_strong_match_bass_plays_the_chord_of_each_bar():
    composition = _layer("Strong")
    roots = _bass_roots_by_bar(composition)

    assert roots
    for bar, pcs in roots.items():
        chord = set(CHORDS[LAYOUT[bar]])
        assert set(pcs) <= chord | {(p + 7) % 12 for p in chord}
        assert CHORDS[LAYOUT[bar]][0] in pcs


def test_layer_keeps_one_key_and_follows_the_loud_parts():
    composition = _layer("Strong")
    names = [s.name for s in composition.sections]

    assert "Chorus" in names
    chorus = next(s for s in composition.sections if s.name == "Chorus")
    assert chorus.start_bar == 8
    assert all(s.key.startswith("C ") for s in composition.sections)


def test_weak_match_still_stays_in_the_key():
    composition = _layer("Weak", roles=("bass",))
    pcs = {n.pitch % 12 for i in composition.instruments for n in i.notes}

    assert pcs <= {0, 2, 4, 5, 7, 9, 11}


def test_layer_needs_notes_in_the_editor():
    options = ComposeOptions(
        root="C", scale="Major", bpm=BPM, mode="Layer over editor"
    )
    with pytest.raises(ValueError, match="needs notes"):
        compose(options, existing=pretty_midi.PrettyMIDI())


def test_harmonic_context_from_midi_reads_the_chords_of_the_song():
    from korshmer.midi_generation.engine import HarmonicContext

    ctx = HarmonicContext.from_midi(_song(), BPM, ROOT, "Major")

    assert ctx.progression == LAYOUT
    assert [ctx.chord_at(b) for b in range(len(LAYOUT))] == LAYOUT


def test_harmonic_context_keeps_a_given_progression_and_default_for_empty():
    from korshmer.midi_generation.engine import HarmonicContext

    given = HarmonicContext.from_midi(_song(), BPM, ROOT, "Major", [0, 4])
    empty = HarmonicContext.from_midi(
        pretty_midi.PrettyMIDI(), BPM, ROOT, "Major"
    )

    assert given.progression == [0, 4]
    assert empty.progression == [0, 4, 5, 3]


def test_strong_layer_bass_is_given_the_kicks_of_the_song(monkeypatch):
    from korshmer.midi_generation import engine

    song = _song()
    kick = pretty_midi.Instrument(0, is_drum=True)
    for bar in range(len(LAYOUT)):
        kick.notes.append(
            pretty_midi.Note(100, 36, bar * BAR + 0.01, bar * BAR + 0.1)
        )
    song.instruments.append(kick)
    seen = []
    real = engine.generate_bass

    def spy(ctx, style, drums, **kw):
        seen.append(drums)
        return real(ctx, style, drums, **kw)

    monkeypatch.setattr(engine, "generate_bass", spy)
    options = ComposeOptions(
        root="C",
        scale="Major",
        bpm=BPM,
        seed=3,
        roles=("bass",),
        texture_batches=0,
        mode="Layer over editor",
        match="Strong",
    )
    compose(options, existing=song)

    assert seen
    for drums in seen:
        assert drums is not None
        assert all(n.pitch in (36, 38) for n in drums.notes)
        assert any(n.pitch == 36 for n in drums.notes)


def _melody(bars=4):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    lead = pretty_midi.Instrument(0)
    pitches = [60, 62, 64, 65, 67, 65, 64, 62]
    for i in range(bars * 8):
        start = i * BAR / 8
        lead.notes.append(
            pretty_midi.Note(90, pitches[i % 8], start, start + BAR / 8 * 0.9)
        )
    pm.instruments.append(lead)
    return pm


def test_continued_melody_hits_chord_tones_on_the_beat():
    options = ComposeOptions(
        root="C",
        scale="Major",
        bpm=BPM,
        minutes=1.0,
        seed=5,
        roles=("melody",),
        texture_batches=0,
        mode="Continue editor",
    )
    composition = compose(options, existing=_melody())

    notes = [
        n
        for inst in composition.instruments
        if "continued" in inst.name
        for n in inst.notes
    ]
    assert notes
    beat = BAR / 4
    checked = 0
    for n in notes:
        offset = n.start % beat
        if min(offset, beat - offset) > 0.06:
            continue
        index = int(n.start // BAR) - composition.start_bar
        chord = composition.bar_chords[index]
        assert n.pitch % 12 in chord
        checked += 1
    assert checked


def test_layer_adds_no_drums_when_the_song_has_drums():
    with_drums = _layer("Strong", roles=("drums", "bass"))
    names = [i.name for i in with_drums.instruments]

    assert all(not i.is_drum for i in with_drums.instruments)
    assert any("Bass" in n for n in names)


def test_layer_adds_drums_when_the_song_has_none():
    song = _song()
    song.instruments = [i for i in song.instruments if not i.is_drum]
    options = ComposeOptions(
        root="C",
        scale="Major",
        bpm=BPM,
        seed=3,
        roles=("drums", "bass"),
        texture_batches=0,
        mode="Layer over editor",
        match="Strong",
    )
    composition = compose(options, existing=song)

    assert any(i.is_drum and i.notes for i in composition.instruments)


def test_a_song_of_even_loudness_is_one_verse_not_all_chorus():
    analysis = song_match.analyze_song(
        _song(loud=range(len(LAYOUT))), BPM, ROOT, MAJOR
    )

    assert {name for name, _s, _b in analysis.sections} == {"Verse"}
    assert sum(b for _n, _s, b in analysis.sections) == analysis.bars
