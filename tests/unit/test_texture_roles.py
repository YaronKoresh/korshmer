import random
from collections import defaultdict

import pytest

from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.midi_generation import texture_roles as tr
from korshmer.midi_generation.composer import (
    MAX_TEXTURE_BATCHES,
    ComposeOptions,
    compose,
)
from korshmer.midi_generation.korshmer_schema import (
    ROUTINE_BARS,
    TRACKS_PER_ACTIVATION,
    build_schema_instruments,
    compose_routine_tracks,
)

BPM = 120.0
BAR = 240.0 / BPM
SLOT = ROUTINE_BARS * BAR
BASIC = ("drums", "bass", "chords", "pad", "melody")


def _song(batches=1, seed=3, root="A", scale="Minor", minutes=10, **kw):
    return compose(
        ComposeOptions(
            root=root,
            scale=scale,
            bpm=BPM,
            minutes=minutes,
            seed=seed,
            roles=BASIC,
            texture_batches=batches,
            **kw,
        )
    )


def _texture(comp):
    return [i for i in comp.instruments if i.name.startswith("Schema ")]


def _role_of(inst):
    return inst.name.rsplit("(", 1)[1].rstrip(")")


def _density_by_section(comp):
    cover = [0] * comp.total_bars
    for inst in _texture(comp):
        for bar in {int(n.start / BAR + 1e-6) for n in inst.notes}:
            cover[bar] += 1
    by_name = defaultdict(list)
    for s in comp.sections:
        for b in range(s.start_bar, min(comp.total_bars, s.start_bar + s.bars)):
            by_name[s.name].append(cover[b])
    return {k: sum(v) / len(v) for k, v in by_name.items()}


def test_a_batch_has_fifteen_different_jobs_in_order():
    assert len(tr.BATCH_ROLES) == TRACKS_PER_ACTIVATION
    assert len({r.name for r in tr.BATCH_ROLES}) == TRACKS_PER_ACTIVATION
    kinds = [r.kind for r in tr.BATCH_ROLES]
    assert kinds == ["drums"] * 3 + ["long"] * 5 + ["short"] * 7
    comp = _song()
    assert [_role_of(i) for i in _texture(comp)] == [
        r.name for r in tr.BATCH_ROLES
    ]


def test_standalone_schema_keeps_plain_names_and_no_roles():
    for inst in build_schema_instruments("C", "Pentatonic Minor", BPM, 0, 1):
        assert "(" not in inst.name


@pytest.mark.parametrize("seed", range(4))
def test_busy_sections_have_more_texture_than_quiet_ones(seed):
    density = _density_by_section(_song(batches=3, seed=seed))
    assert density["Chorus"] > density["Break"]
    assert density["Chorus"] > density["Intro"]
    assert density["Chorus"] >= 2 * density["Break"] or density["Break"] <= 4


def test_ninety_tracks_do_not_flood_the_quiet_sections():
    density = _density_by_section(_song(batches=MAX_TEXTURE_BATCHES))
    assert density["Chorus"] >= 2 * max(density["Break"], density["Outro"])


@pytest.mark.parametrize("seed", range(3))
def test_drums_are_kick_snare_and_hats_not_tuned_pitches(seed):
    allowed = {
        "Kick": {35, 36},
        "Snare": {37, 38, 40},
        "Hats": {42, 46},
    }
    for inst in _texture(_song(seed=seed)):
        if inst.is_drum:
            assert {n.pitch for n in inst.notes} <= allowed[_role_of(inst)]


def test_kick_lands_on_the_strong_beats_and_snare_on_the_backbeat():
    beat = 60.0 / BPM
    comp = _song(batches=1)
    for inst in _texture(comp):
        role = _role_of(inst)
        if role not in ("Kick", "Snare"):
            continue
        for n in inst.notes:
            pos = round((n.start % BAR) / beat)
            assert pos in ((0, 2, 3) if role == "Kick" else (1, 3))


def test_every_role_plays_in_its_own_register():
    comp = _song(batches=1)
    for inst in _texture(comp):
        if inst.is_drum:
            continue
        role = next(r for r in tr.BATCH_ROLES if r.name == _role_of(inst))
        root_idx = NOTE_NAMES.index("A")
        octaves = {(n.pitch - root_idx) // 12 - 1 for n in inst.notes}
        assert octaves == {role.octave}, role.name


def test_bass_and_drone_sit_below_the_high_voices():
    comp = _song(batches=1)
    low = {}
    for inst in _texture(comp):
        if not inst.is_drum:
            low[_role_of(inst)] = min(n.pitch for n in inst.notes)
    assert low["Drone"] < low["Foundation"] < low["Harmony"] < low["Air"]
    assert low["Bass Pulse"] < low["Hook"] < low["Sparkle"]


@pytest.mark.parametrize("root", ["C", "F#"])
@pytest.mark.parametrize("scale", ["Major", "Minor"])
def test_fixed_roles_open_on_the_key_and_colour_roles_avoid_it(root, scale):
    root_idx = NOTE_NAMES.index(root)
    comp = _song(root=root, scale=scale)
    for inst in _texture(comp):
        if inst.is_drum:
            continue
        role = next(r for r in tr.BATCH_ROLES if r.name == _role_of(inst))
        if role.follows_chords:
            continue
        classes = {n.pitch % 12 for n in inst.notes}
        if role.avoid_root:
            assert root_idx not in classes, role.name
        if role.root_first:
            first = sorted(inst.notes, key=lambda n: n.start)[0]
            assert first.pitch % 12 == root_idx, role.name


def _pentatonic(root, scale):
    return {(NOTE_NAMES.index(root) + i) % 12 for i in SCALES[scale]}


def _direct(roots, scale="Pentatonic Minor", root="A", bars=32):
    return compose_routine_tracks(
        random.Random(5),
        NOTE_NAMES.index(root),
        scale,
        BPM,
        bars,
        TRACKS_PER_ACTIVATION,
        bar_energy=[1.0] * bars,
        bar_roots=roots,
    )


def _by_role(instruments):
    return {_role_of(i): i for i in instruments}


@pytest.mark.parametrize("scale", ["Pentatonic Minor", "Pentatonic Major"])
def test_following_roles_move_with_the_chord_root_inside_the_scale(scale):
    still = _by_role(_direct([0] * 32, scale))
    moved = _by_role(_direct([7] * 32, scale))
    allowed = _pentatonic("A", scale)
    for name, role in ((r.name, r) for r in tr.BATCH_ROLES):
        if role.kind == "drums":
            continue
        a = [n.pitch for n in sorted(still[name].notes, key=lambda n: n.start)]
        b = [n.pitch for n in sorted(moved[name].notes, key=lambda n: n.start)]
        assert {p % 12 for p in b} <= allowed
        if role.follows_chords:
            assert a != b, name
        else:
            assert a == b, name


def test_a_drone_changes_note_when_the_chord_changes():
    roots = [0, 7, 9, 5] * 8
    drone = sorted(
        _by_role(_direct(roots))["Drone"].notes, key=lambda n: n.start
    )
    first_slot = [n.pitch % 12 for n in drone[:4]]
    assert len(set(first_slot)) >= 3
    assert first_slot[0] == NOTE_NAMES.index("A")


def test_chord_shift_takes_the_nearest_degree_and_the_lower_on_a_tie():
    minor_pent = SCALES["Pentatonic Minor"]
    assert tr.chord_shift(0, minor_pent) == 0
    assert tr.chord_shift(7, minor_pent) == 3
    assert tr.chord_shift(4, minor_pent) == 1
    assert tr.chord_shift(11, minor_pent) == 0


def test_composed_song_texture_follows_the_progression_per_bar():
    comp = _song(batches=1, minutes=2)
    drone = next(i for i in _texture(comp) if _role_of(i) == "Drone")
    pitches = {n.pitch % 12 for n in drone.notes}
    assert len(pitches) >= 2
    assert pitches <= _pentatonic("A", "Pentatonic Minor")


def test_answer_plays_where_the_hook_rests():
    answered = overlap = 0
    for seed in range(8):
        by_role = {
            _role_of(i): i for i in _texture(_song(batches=1, seed=seed))
        }
        hook = {int(n.start // SLOT) for n in by_role["Hook"].notes}
        answer = {int(n.start // SLOT) for n in by_role["Answer"].notes}
        answered += len(answer)
        overlap += len(answer & hook)
    assert answered > 0
    assert overlap <= answered * 0.1


def test_later_batches_join_only_when_the_music_is_dense():
    comp = _song(batches=MAX_TEXTURE_BATCHES)
    texture = _texture(comp)
    first, last = texture[:15], texture[-15:]

    def quiet_notes(batch):
        return sum(
            1
            for inst in batch
            for n in inst.notes
            if any(
                s.energy < 0.4
                and s.start_bar * BAR <= n.start < (s.start_bar + s.bars) * BAR
                for s in comp.sections
            )
        )

    assert quiet_notes(last) <= quiet_notes(first)


def test_roles_follow_the_pure_helpers():
    assert tr.role_for(0).name == "Kick"
    assert tr.role_for(15).name == "Kick"
    assert tr.batch_of(29) == 1
    assert tr.role_octave(tr.LONG_ROLES[0], 0) == 2
    assert tr.role_octave(tr.LONG_ROLES[0], 2) == tr.MIN_OCTAVE
    assert tr.role_octave(tr.SHORT_ROLES[5], 1) == tr.MAX_OCTAVE
    assert tr.entry_energy(tr.DRUM_ROLES[0], 0) == pytest.approx(0.4)
    assert tr.entry_energy(tr.DRUM_ROLES[2], 5) == tr.MAX_ENTRY_ENERGY
    assert tr.slot_energies([1, 1, 1, 1, 0, 0, 0, 0], 2, 4) == [1.0, 0.0]


def test_gate_keeps_wanted_slots_and_never_returns_nothing():
    rng = random.Random(1)
    energies = [0.3, 0.3, 1.0, 1.0, 0.3]
    kick = tr.DRUM_ROLES[0]
    assert tr.gate_slots(rng, {0, 2, 3}, kick, 0, energies) == {2, 3}
    assert tr.gate_slots(rng, {0, 1}, kick, 0, energies) <= {2, 3}
    none = [0.1, 0.1]
    assert tr.gate_slots(rng, {0}, kick, 0, none)
    sparkle = tr.SHORT_ROLES[6]
    assert tr.gate_slots(rng, set(), sparkle, 0, energies) >= {2}


def test_velocity_grows_with_energy_and_stays_in_midi_range():
    for role in tr.BATCH_ROLES:
        assert tr.velocity_for(role, 1.0) > tr.velocity_for(role, 0.0)
    comp = _song(batches=2)
    for inst in _texture(comp):
        assert all(1 <= n.velocity <= 127 for n in inst.notes)


@pytest.mark.parametrize("seed", range(3))
def test_same_seed_gives_the_same_arranged_texture(seed):
    a = [
        (i.name, [(n.pitch, n.start) for n in i.notes])
        for i in _texture(_song(seed=seed))
    ]
    b = [
        (i.name, [(n.pitch, n.start) for n in i.notes])
        for i in _texture(_song(seed=seed))
    ]
    assert a == b


@pytest.mark.parametrize("style", ["Pop", "Rock", "Jazz", "EDM", "Lo-Fi"])
@pytest.mark.parametrize("seed", range(3))
def test_darker_instruments_take_the_lower_registers(style, seed):
    comp = _song(batches=MAX_TEXTURE_BATCHES, seed=seed, style=style)
    texture = [i for i in _texture(comp) if not i.is_drum]
    for batch in range(MAX_TEXTURE_BATCHES):
        chunk = texture[batch * 12 : (batch + 1) * 12]
        for kind in ("long", "short"):
            roles = tr.LONG_ROLES if kind == "long" else tr.SHORT_ROLES
            names = {r.name: r for r in roles}
            voices = [i for i in chunk if _role_of(i) in names]
            voices = [i for i in voices if names[_role_of(i)].kind == kind]
            voices.sort(key=lambda i: tr.role_octave(names[_role_of(i)], batch))
            levels = [tr.brightness(i.program) for i in voices]
            assert levels == sorted(levels)


def test_match_voices_pairs_dark_with_low_and_bright_with_high():
    picks = [("Glockenspiel", 9), ("Cello", 42), ("Marimba", 12)]
    roles = (tr.SHORT_ROLES[5], tr.SHORT_ROLES[0], tr.SHORT_ROLES[2])
    matched = tr.match_voices(picks, roles, 0)
    assert [p[0] for p in matched] == ["Glockenspiel", "Cello", "Marimba"]
    assert tr.brightness(9999) == tr.DEFAULT_BRIGHTNESS
