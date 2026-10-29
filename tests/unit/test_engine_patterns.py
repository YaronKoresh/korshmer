import pretty_midi
import pytest

from korshmer.constants import SCALES
from korshmer.midi_generation import engine, genre_feel, hooks
from korshmer.midi_generation.engine import SONG_STYLES, HarmonicContext

TOMS = {genre_feel.LOW_TOM, genre_feel.MID_TOM, genre_feel.HIGH_TOM}
RIFF = [(0, 3, 0), (4, 2, 0), (8, 4, 0)]
HOOK = hooks.Hook(
    ((0, 3, 0), (3, 3, 2), (6, 2, 1), (8, 4, 4), (12, 4, 2)), 0.9, 74
)


def _ctx(progression=(0, 5, 2, 6), scale="Minor"):
    return HarmonicContext(
        root=9, scale=scale, bpm=120.0, progression=list(progression)
    )


BAR = _ctx().bar
STEP = BAR / 16


def _step(note):
    return round(note.start / STEP)


def _signature(inst):
    return [(n.pitch, round(n.start, 4), round(n.end, 4)) for n in inst.notes]


def _groups(inst):
    grouped = {}
    for n in inst.notes:
        grouped.setdefault(_step(n), []).append(n.pitch)
    return [sorted(p) for _, p in sorted(grouped.items())]


def _bar_notes(inst, bar):
    return [n for n in inst.notes if bar * BAR <= n.start < (bar + 1) * BAR]


def test_every_style_names_known_bass_patterns():
    for feel in genre_feel.FEELS.values():
        assert set(feel.bass) <= set(engine._BASS_BARS)
    for style in SONG_STYLES:
        assert engine._STYLES[style][2] in engine._BASS_BARS


@pytest.mark.parametrize("pattern", sorted(engine._BASS_BARS))
def test_bass_pattern_stays_in_the_bass_register(pattern):
    inst = engine.generate_bass(
        _ctx(), rhythm=pattern, bars=8, seed=1, riff=RIFF
    )
    assert inst.notes
    for n in inst.notes:
        assert 30 <= n.pitch <= 60
        assert n.end > n.start >= 0


def test_bass_patterns_have_their_own_shape():
    shapes = set()
    for pattern in engine._BASS_BARS:
        inst = engine.generate_bass(
            _ctx(), rhythm=pattern, bars=1, seed=0, riff=RIFF
        )
        shapes.add(
            tuple(
                (_step(n), round((n.end - n.start) / STEP, 1), n.pitch)
                for n in inst.notes
            )
        )
    assert len(shapes) == len(engine._BASS_BARS)


def test_offbeat_bass_never_plays_on_the_beat():
    inst = engine.generate_bass(_ctx(), rhythm="offbeat", bars=8, seed=2)
    assert {_step(n) % 4 for n in inst.notes} == {2}


def test_pump_bass_leaves_every_beat_open():
    inst = engine.generate_bass(_ctx(), rhythm="pump", bars=8, seed=2)
    assert {_step(n) % 4 for n in inst.notes} == {1, 2, 3}


def test_walking_bass_steps_into_the_next_root():
    ctx = _ctx()
    inst = engine.generate_bass(ctx, rhythm="walking", bars=8, seed=3)
    notes = sorted(inst.notes, key=lambda n: n.start)
    assert [_step(n) % 16 for n in notes] == [0, 4, 8, 12] * 8
    for b in range(7):
        bar = notes[4 * b : 4 * b + 4]
        following = notes[4 * b + 4]
        assert bar[0].pitch % 12 == ctx.degree_pitch(ctx.chord_at(b), 2) % 12
        assert 1 <= abs(bar[3].pitch - following.pitch) <= 2


def test_walking_bass_over_a_held_chord_climbs_to_the_fifth():
    ctx = _ctx((0,))
    inst = engine.generate_bass(ctx, rhythm="walking", bars=3, seed=3)
    notes = sorted(inst.notes, key=lambda n: n.start)
    assert notes[3].pitch - notes[0].pitch == 7


def test_808_bass_plays_with_the_kick():
    ctx = _ctx()
    drums = engine.generate_drums(ctx, "HipHop", bars=4, seed=1)
    inst = engine.generate_bass(ctx, drums=drums, rhythm="808", bars=4, seed=1)
    kicks = {_step(n) for n in drums.notes if n.pitch == genre_feel.KICK}
    assert len(kicks) > 4
    assert {_step(n) for n in inst.notes} == kicks


def test_808_bass_holds_each_note_until_the_next_kick():
    ctx = _ctx()
    drums = engine.generate_drums(ctx, "HipHop", bars=2, seed=1)
    inst = engine.generate_bass(ctx, drums=drums, rhythm="808", bars=2, seed=1)
    first = [n for n in inst.notes if _step(n) == 0][0]
    assert first.end - first.start == pytest.approx(
        7 * STEP * genre_feel.feel_for("Pop").bass_gate
    )


def test_bass_locks_onto_nearby_kicks():
    ctx = _ctx()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for b in range(2):
        start = b * BAR + 0.02
        drums.notes.append(pretty_midi.Note(100, 36, start, start + 0.1))
    inst = engine.generate_bass(
        ctx, drums=drums, rhythm="eighths", bars=2, seed=1
    )
    starts = {round(n.start, 4) for n in inst.notes}
    assert {0.02, round(BAR + 0.02, 4)} <= starts


def test_riff_bass_without_a_riff_falls_back_to_driving():
    riff = engine.generate_bass(_ctx(), rhythm="riff", bars=4, seed=5)
    driving = engine.generate_bass(_ctx(), rhythm="driving", bars=4, seed=5)
    assert _signature(riff) == _signature(driving)


def test_riff_bass_follows_the_riff_rhythm():
    inst = engine.generate_bass(
        _ctx(), rhythm="riff", riff=RIFF, bars=4, seed=5
    )
    assert {_step(n) % 16 for n in inst.notes} == {0, 4, 8}


def test_quiet_sections_soften_the_bass():
    loud = engine.generate_bass(_ctx(), rhythm="driving", bars=4, seed=5)
    soft = engine.generate_bass(
        _ctx(), rhythm="driving", bars=4, seed=5, energy=0.3
    )
    assert max(n.velocity for n in soft.notes) < max(
        n.velocity for n in loud.notes
    )


def test_power_chords_are_root_fifth_octave_and_low():
    inst = engine.generate_chords(
        _ctx(), "Rock", bars=8, seed=1, power=True, energy=0.5
    )
    chords = _groups(inst)
    assert len(chords) == 8
    for low, fifth, top in chords:
        assert fifth - low == 7
        assert top - low == 12
        assert 45 <= low < 57


def test_jazz_shell_voicing_leaves_out_the_root():
    ctx = _ctx()
    inst = engine.generate_chords(ctx, "Jazz", bars=8, seed=1, energy=0.5)
    for bar in range(8):
        pitches = [n.pitch for n in _bar_notes(inst, bar)]
        degree = ctx.chord_at(bar)
        expected = {ctx.degree_pitch(degree + k, 4) % 12 for k in (2, 6, 8)}
        assert {p % 12 for p in pitches} == expected
        assert ctx.degree_pitch(degree, 4) % 12 not in expected


def test_edm_stack_voicing_doubles_the_root_on_top():
    inst = engine.generate_chords(_ctx(), "EDM", bars=8, seed=1, energy=0.5)
    for chord in _groups(inst):
        assert len(chord) == 4
        assert chord[-1] - chord[0] == 12
        assert chord[0] >= 55


@pytest.mark.parametrize("style", ["Hip-Hop", "Lo-Fi"])
def test_open_voicing_spreads_the_triad_over_more_than_an_octave(style):
    inst = engine.generate_chords(_ctx(), style, bars=8, seed=1, energy=0.5)
    for chord in _groups(inst):
        assert len(chord) == 3
        assert chord[-1] - chord[0] >= 14


def test_open_voicing_adds_the_seventh_on_request():
    inst = engine.generate_chords(
        _ctx(), "Lo-Fi", bars=4, seed=1, energy=0.5, seventh=True
    )
    assert {len(chord) for chord in _groups(inst)} == {4}


def test_close_voicing_stays_within_an_octave():
    inst = engine.generate_chords(_ctx(), "Pop", bars=8, seed=1, energy=0.5)
    for chord in _groups(inst):
        assert len(chord) == 3
        assert chord[-1] - chord[0] < 12


@pytest.mark.parametrize("style", SONG_STYLES)
def test_chord_length_follows_the_style_gate(style):
    inst = engine.generate_chords(_ctx(), style, bars=4, seed=1, energy=0.5)
    gate = genre_feel.feel_for(style).chord_gate
    for n in inst.notes:
        assert n.end - n.start == pytest.approx(gate * BAR)


def test_chord_hits_split_the_bar():
    inst = engine.generate_chords(
        _ctx(), "Pop", bars=4, seed=1, hits=[0.0, 0.25, 0.75]
    )
    assert sorted({_step(n) % 16 for n in inst.notes}) == [0, 4, 12]


def test_ghost_notes_are_quiet_and_off_the_backbeat():
    ctx = _ctx()
    ghosts = {genre_feel.SNARE: [3, 6]}
    inst = engine.generate_drums(ctx, "HipHop", bars=16, seed=4, ghosts=ghosts)
    quiet = [n for n in inst.notes if n.velocity < 50]
    assert 4 <= len(quiet) <= 32
    assert {n.pitch for n in quiet} == {genre_feel.SNARE}
    assert {_step(n) % 16 for n in quiet} <= {3, 6}
    plain = engine.generate_drums(ctx, "HipHop", bars=16, seed=4)
    assert not [n for n in plain.notes if n.velocity < 50]


def test_ghost_notes_wait_for_a_full_section():
    inst = engine.generate_drums(
        _ctx(),
        "HipHop",
        bars=8,
        seed=4,
        energy=0.4,
        ghosts={genre_feel.SNARE: [3, 6]},
    )
    assert not [n for n in inst.notes if _step(n) % 16 in (3, 6)]


def test_crash_marks_the_start_of_each_phrase():
    inst = engine.generate_drums(_ctx(), "Rock", bars=9, seed=1, crash=True)
    crashes = [n for n in inst.notes if n.pitch == engine.CRASH]
    assert [round(n.start / BAR) for n in crashes] == [0, 4, 8]
    assert {n.velocity for n in crashes} == {108}


def test_crash_is_off_by_default_and_in_quiet_sections():
    assert not [
        n
        for n in engine.generate_drums(_ctx(), "Rock", bars=8).notes
        if n.pitch == engine.CRASH
    ]
    quiet = engine.generate_drums(
        _ctx(), "Rock", bars=8, crash=True, energy=0.4
    )
    assert not [n for n in quiet.notes if n.pitch == engine.CRASH]


def test_tom_fill_replaces_the_second_half_of_the_fill_bar():
    inst = engine.generate_drums(_ctx(), "Rock", bars=4, seed=1, fill="toms")
    last = _bar_notes(inst, 3)
    assert TOMS <= {n.pitch for n in last}
    late = [n for n in last if round((n.start - 3 * BAR) / STEP) >= 8]
    assert {n.pitch for n in late} == TOMS
    assert not [n for n in inst.notes if n.pitch in TOMS and n.start < 3 * BAR]


def test_snare_roll_fills_the_second_half_of_the_fill_bar():
    inst = engine.generate_drums(_ctx(), "HipHop", bars=4, seed=1, fill="roll")
    late = [
        n
        for n in _bar_notes(inst, 3)
        if n.pitch == genre_feel.SNARE
        and round((n.start - 3 * BAR) / STEP) >= 8
    ]
    assert [round((n.start - 3 * BAR) / STEP) for n in late] == list(
        range(8, 16)
    )
    velocities = [n.velocity for n in late]
    assert velocities == sorted(velocities)
    assert velocities[-1] > velocities[0]


def test_hat_fill_crams_thirty_seconds_into_the_last_beat():
    inst = engine.generate_drums(_ctx(), "HipHop", bars=4, seed=1, fill="hats")
    hats = [n for n in inst.notes if n.pitch == genre_feel.HAT]
    half = sorted(
        (n for n in hats if (n.start / STEP) % 1 == pytest.approx(0.5)),
        key=lambda n: n.start,
    )
    assert len(half) == 4
    assert [(n.start - 3 * BAR) / STEP for n in half] == pytest.approx(
        [12.5, 13.5, 14.5, 15.5]
    )
    velocities = [n.velocity for n in half]
    assert velocities == sorted(velocities)


def test_snare_fill_hits_the_last_four_sixteenths():
    inst = engine.generate_drums(_ctx(), "Rock", bars=4, seed=1, fill="snare")
    steps = {
        round((n.start - 3 * BAR) / STEP)
        for n in _bar_notes(inst, 3)
        if n.pitch == genre_feel.SNARE
    }
    assert {12, 13, 14, 15} <= steps


def test_no_fill_means_the_groove_plays_on():
    plain = engine.generate_drums(_ctx(), "Rock", bars=8, seed=1, fill="none")
    bare = engine.generate_drums(_ctx(), "Rock", bars=8, seed=1, fill_bars=0)
    assert _signature(plain) == _signature(bare)


def test_fills_wait_for_a_full_section():
    inst = engine.generate_drums(
        _ctx(), "Rock", bars=4, seed=1, fill="toms", energy=0.4
    )
    assert not [n for n in inst.notes if n.pitch in TOMS]


def _chord(inst, start, pitches=(60, 64, 67), length=1.0):
    for p in pitches:
        inst.notes.append(pretty_midi.Note(80, p, start, start + length))


def test_strum_staggers_a_chord_from_the_bottom_up():
    inst = pretty_midi.Instrument(program=0)
    _chord(inst, 0.0)
    _chord(inst, 2.0)
    genre_feel.apply_strum(inst, 0.02)
    for base in (0.0, 2.0):
        chord = sorted(
            (n for n in inst.notes if base <= n.start < base + 0.5),
            key=lambda n: n.pitch,
        )
        assert [n.start - base for n in chord] == pytest.approx(
            [0.0, 0.02, 0.04]
        )
        assert {n.end for n in chord} == {base + 1.0}


def test_strum_never_eats_more_than_half_of_a_short_note():
    inst = pretty_midi.Instrument(program=0)
    _chord(inst, 0.0, length=0.03)
    genre_feel.apply_strum(inst, 0.05)
    assert max(n.start for n in inst.notes) == pytest.approx(0.015)


def test_no_strum_is_a_noop():
    inst = pretty_midi.Instrument(program=0)
    _chord(inst, 0.0)
    genre_feel.apply_strum(inst, 0.0)
    assert {n.start for n in inst.notes} == {0.0}


def _melody(**kw):
    options = {"ctx": _ctx(), "bars": 8, "seed": 6, "hook": HOOK}
    options.update(kw)
    return engine.generate_melody(**options)


def test_melody_is_seeded():
    assert _signature(_melody()) == _signature(_melody())
    other = _melody(seed=7, hook=None)
    assert _signature(other) != _signature(_melody(hook=None))


def test_melody_opens_with_the_hook_rhythm():
    inst = _melody()
    rhythm = tuple(onset for onset, _, _ in HOOK.notes)
    for bar in (0, 1):
        assert (
            tuple(_step(n) - 16 * bar for n in _bar_notes(inst, bar)) == rhythm
        )


def test_melody_restates_the_hook_over_the_same_chord():
    inst = _melody(ctx=_ctx((0, 5)), bars=8)
    shape = lambda b: [
        (_step(n) - 16 * b, n.pitch) for n in _bar_notes(inst, b)
    ]
    assert shape(0) == shape(4)
    assert shape(1) == shape(5)
    assert shape(0) != shape(1)


def test_half_cadence_rests_on_the_fifth_of_the_chord():
    ctx = _ctx()
    inst = _melody(bars=4)
    last = max(inst.notes, key=lambda n: n.start)
    assert (last.pitch - ctx.degree_pitch(ctx.chord_at(3) + 4, 4)) % 12 == 0


def test_closing_melody_rests_on_the_chord_root():
    ctx = _ctx()
    inst = _melody(final=True)
    last = max(inst.notes, key=lambda n: n.start)
    assert (last.pitch - ctx.degree_pitch(ctx.chord_at(7), 4)) % 12 == 0


def test_zero_bars_give_no_melody():
    assert _melody(bars=0).notes == []


def test_verses_sit_below_choruses_and_lift_raises_the_line():
    mean = lambda inst: sum(n.pitch for n in inst.notes) / len(inst.notes)
    chorus = mean(_melody(kind="chorus"))
    assert mean(_melody(kind="verse")) < chorus - 4
    assert mean(_melody(kind="chorus", lift=12)) > chorus + 5


@pytest.mark.parametrize("scale", sorted(SCALES))
def test_melody_stays_in_the_scale_for_every_scale(scale):
    ctx = _ctx((0, 3, 4, 2), scale=scale)
    allowed = {(ctx.root + i) % 12 for i in SCALES[scale]}
    for kind in ("chorus", "verse", "break"):
        inst = engine.generate_melody(ctx, bars=8, seed=3, kind=kind)
        assert inst.notes
        for n in inst.notes:
            assert n.pitch % 12 in allowed
            assert 36 <= n.pitch <= 96


def test_bass_with_an_unknown_rhythm_plays_eighths():
    unknown = engine.generate_bass(_ctx(), rhythm="mystery", bars=4, seed=5)
    eighths = engine.generate_bass(_ctx(), rhythm="eighths", bars=4, seed=5)
    assert _signature(unknown) == _signature(eighths)


def test_trim_line_cuts_a_note_at_the_next_onset():
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [
        pretty_midi.Note(80, 60, 0.0, 1.0),
        pretty_midi.Note(80, 64, 0.5, 1.5),
        pretty_midi.Note(80, 67, 2.0, 3.0),
    ]
    engine.trim_line(inst)
    assert [(n.start, n.end) for n in inst.notes] == [
        (0.0, 0.5),
        (0.5, 1.5),
        (2.0, 3.0),
    ]


def test_trim_line_leaves_notes_that_start_together_alone():
    inst = pretty_midi.Instrument(program=0)
    _chord(inst, 0.0)
    _chord(inst, 2.0, length=0.5)
    engine.trim_line(inst)
    assert {(n.start, n.end) for n in inst.notes} == {(0.0, 1.0), (2.0, 2.5)}
