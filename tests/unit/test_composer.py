import itertools
import random

import pretty_midi
import pytest

from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.midi_generation.composer import (
    AUTO_STRUCTURE,
    DEFAULT_ROLES,
    MAX_TEXTURE_BATCHES,
    MIN_ACTIVE_ROLES,
    MODES,
    ROLE_KEYS,
    ROLE_LABELS,
    SECTION_PROFILES,
    STRUCTURES,
    STYLE_STRUCTURES,
    ComposeOptions,
    active_roles,
    compose,
    harmony_scale,
    make_progression,
    plan_sections,
    structure_for,
    validate_options,
)
from korshmer.midi_generation.engine import SONG_STYLES
from korshmer.midi_generation.korshmer_schema import (
    TRACKS_PER_ACTIVATION,
    bars_for_minutes,
)
from korshmer.utils.track_layout import DRUMS_NAME, PROGRAM_NAMES

BPM = 120.0
BAR = 240.0 / BPM


def _opts(**kw):
    base = {
        "root": "A",
        "scale": "Minor",
        "bpm": BPM,
        "minutes": 2.0,
        "seed": 1,
    }
    base.update(kw)
    return ComposeOptions(**base)


def _role_names(composition):
    return {i.name.rstrip("0123456789 ") for i in composition.instruments}


def _sig(composition, name):
    for inst in composition.instruments:
        if inst.name == name:
            return [(round(n.start, 3), n.pitch) for n in inst.notes]
    return None


def _sounding_per_bar(composition):
    bars = composition.total_bars + composition.start_bar
    cover = [0] * bars
    for inst in composition.instruments:
        playing = set()
        for n in inst.notes:
            first = int(n.start / BAR + 1e-6)
            last = int((n.end - 1e-6) / BAR)
            playing.update(range(first, min(last, bars - 1) + 1))
        for b in playing:
            cover[b] += 1
    return cover


@pytest.mark.parametrize("structure", list(STRUCTURES))
@pytest.mark.parametrize("bars", [16, 20, 32, 44, 92, 180, 452])
def test_sections_fill_the_requested_bars_exactly(structure, bars):
    for seed in range(12):
        sections = plan_sections(random.Random(seed), bars, structure)
        assert sum(s.bars for s in sections) == bars
        assert all(s.bars >= 4 and s.bars % 4 == 0 for s in sections)
        cursor = 0
        for s in sections:
            assert s.start_bar == cursor
            assert s.name in SECTION_PROFILES
            cursor += s.bars


def test_song_structure_has_verse_chorus_and_a_break_at_normal_length():
    for seed in range(6):
        names = [
            s.name
            for s in plan_sections(random.Random(seed), 92, "Verse / Chorus")
        ]
        assert names[0] == "Intro"
        assert names[-1] == "Outro"
        assert {"Verse", "Chorus", "Break"} <= set(names)


def test_short_requests_fall_back_to_a_single_loop():
    sections = plan_sections(random.Random(1), 8, "Verse / Chorus")
    assert [s.name for s in sections] == ["Loop"]


@pytest.mark.parametrize("minutes", [0.5, 1, 3, 6])
def test_notes_stay_inside_the_requested_length(minutes):
    c = compose(_opts(minutes=minutes))
    assert c.total_bars == bars_for_minutes(minutes, BPM)
    end = max(n.end for i in c.instruments for n in i.notes)
    assert end <= c.total_bars * BAR + 1e-6
    assert end >= (c.total_bars - 8) * BAR


@pytest.mark.parametrize(
    ("root", "scale"),
    [
        ("A", "Minor"),
        ("C", "Major"),
        ("D", "Dorian"),
        ("F#", "Pentatonic Minor"),
    ],
)
def test_same_key_and_scale_with_another_seed_gives_another_song(root, scale):
    comps = [
        compose(_opts(root=root, scale=scale, seed=s, texture_batches=0))
        for s in range(6)
    ]
    for name in ("Drums", "Bass", "Chords", "Pad", "Melody"):
        sigs = [_sig(c, name) for c in comps]
        pairs = list(itertools.combinations(sigs, 2))
        assert all(a != b for a, b in pairs), name


def test_progressions_differ_between_seeds_and_between_section_kinds():
    seen = {
        tuple(make_progression(random.Random(s), "verse")) for s in range(40)
    }
    assert len(seen) >= 8
    chorus = {
        tuple(make_progression(random.Random(s), "chorus")) for s in range(40)
    }
    assert chorus != seen


def test_same_seed_gives_the_same_song():
    a = compose(_opts(seed=11))
    b = compose(_opts(seed=11))
    assert [
        (i.name, [(n.pitch, n.start, n.end) for n in i.notes])
        for i in a.instruments
    ] == [
        (i.name, [(n.pitch, n.start, n.end) for n in i.notes])
        for i in b.instruments
    ]


def test_sections_get_their_own_character():
    c = compose(_opts(minutes=3, seed=2, texture_batches=0))
    by_name = {}
    for s in c.sections:
        by_name.setdefault(s.name, s)
    assert "drums" not in by_name["Break"].roles
    assert "drums" in by_name["Chorus"].roles
    assert (
        by_name["Chorus"].energy
        > by_name["Verse"].energy
        > by_name["Intro"].energy
    )


def test_roles_decide_which_tracks_exist():
    c = compose(_opts(roles=("bass", "melody"), texture_batches=0))
    assert _role_names(c) == {"Bass", "Melody"}


def test_every_role_can_be_generated():
    c = compose(_opts(roles=ROLE_KEYS, texture_batches=0, minutes=3))
    names = {i.name for i in c.instruments}
    assert {"Drums", "Bass", "Chords", "Pad", "Melody", "Arp", "Pluck"} <= names
    assert "Counter-melody" in names
    assert all(i.notes for i in c.instruments)


@pytest.mark.parametrize("batches", [0, 1, 2, MAX_TEXTURE_BATCHES])
def test_texture_adds_batches_of_schema_tracks(batches):
    c = compose(_opts(texture_batches=batches, minutes=3))
    schema = [i for i in c.instruments if i.name.startswith("Schema ")]
    assert len(schema) == batches * TRACKS_PER_ACTIVATION
    voices = [i for i in c.instruments if not i.name.startswith("Schema ")]
    assert len(voices) >= len(DEFAULT_ROLES)
    assert _role_names(c) - {f"Schema {n}" for n in range(1, 11)} >= {
        ROLE_LABELS[r] for r in DEFAULT_ROLES
    }


def test_texture_indexes_follow_the_tracks_already_in_the_editor():
    c = compose(_opts(existing_tracks=7, texture_batches=1))
    schema = [i for i in c.instruments if i.name.startswith("Schema ")]
    voices = [i for i in c.instruments if not i.name.startswith("Schema ")]
    first = (7 + len(voices)) % 10 + 1
    assert schema[0].name.startswith(f"Schema {first} ")


def test_texture_follows_the_key_for_any_scale():
    for scale in (
        "Major",
        "Minor",
        "Dorian",
        "Mixolydian",
        "Pentatonic Major",
        "Blues",
    ):
        c = compose(
            _opts(
                scale=scale,
                root="D",
                roles=("pad", "chords"),
                texture_batches=1,
            )
        )
        parent = SCALES[
            "Pentatonic Minor"
            if (3 in SCALES[scale] and 4 not in SCALES[scale])
            else "Pentatonic Major"
        ]
        allowed = {(2 + i) % 12 for i in parent}
        for inst in c.instruments:
            if inst.name.startswith("Schema ") and not inst.is_drum:
                assert {n.pitch % 12 for n in inst.notes} <= allowed


@pytest.mark.parametrize(
    "scale", ["Major", "Minor", "Dorian", "Mixolydian", "Harmonic Minor"]
)
@pytest.mark.parametrize("root", ["C", "F#"])
def test_melodic_roles_stay_in_the_chosen_scale(scale, root):
    allowed = {(NOTE_NAMES.index(root) + i) % 12 for i in SCALES[scale]}
    c = compose(
        _opts(scale=scale, root=root, roles=ROLE_KEYS, texture_batches=0)
    )
    for inst in c.instruments:
        if inst.is_drum:
            continue
        assert {n.pitch % 12 for n in inst.notes} <= allowed, inst.name


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("minutes", [0.5, 2, 5])
def test_no_bar_has_fewer_than_two_sounding_tracks(minutes, seed):
    c = compose(_opts(minutes=minutes, seed=seed))
    assert min(_sounding_per_bar(c)) >= MIN_ACTIVE_ROLES


@pytest.mark.parametrize("seed", range(5))
def test_two_roles_never_leave_a_section_with_one(seed):
    c = compose(
        _opts(
            roles=("drums", "melody"), texture_batches=0, seed=seed, minutes=3
        )
    )
    for s in c.sections:
        assert len(s.roles) >= MIN_ACTIVE_ROLES


def test_active_roles_follow_the_profile_but_never_drop_below_two():
    assert active_roles(
        {"pad", "chords"}, ("drums", "bass", "chords", "pad"), False
    ) == ("chords", "pad")
    got = active_roles({"pad"}, ("drums", "bass", "pad"), False)
    assert len(got) == 2 and "pad" in got
    assert active_roles({"pad"}, ("drums",), False) == ("drums",)
    assert len(active_roles({"pad"}, ("drums", "bass", "pad"), True)) >= 1


def _existing(end_bars=8):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    lead = pretty_midi.Instrument(program=0, name="Lead")
    pitches = [69, 72, 74, 72, 69, 67, 69, 72]
    for i in range(end_bars * 2):
        lead.notes.append(
            pretty_midi.Note(90, pitches[i % 8], i * 1.0, i * 1.0 + 0.9)
        )
    pm.instruments.append(lead)
    return pm


def test_continue_mode_starts_after_the_existing_music():
    pm = _existing(8)
    end = pm.get_end_time()
    c = compose(_opts(mode="Continue editor", texture_batches=0), existing=pm)
    assert c.start_bar == int(-(-end // BAR))
    assert (
        min(n.start for i in c.instruments for n in i.notes)
        >= c.start_bar * BAR - 1e-6
    )
    assert (
        max(n.end for i in c.instruments for n in i.notes)
        <= (c.start_bar + c.total_bars) * BAR + 1e-6
    )


def test_continue_mode_extends_the_existing_melody():
    pm = _existing(8)
    c = compose(
        _opts(mode="Continue editor", texture_batches=0, seed=3), existing=pm
    )
    lead = next(i for i in c.instruments if i.name.startswith("Melody"))
    assert lead.notes
    allowed = {(9 + i) % 12 for i in SCALES["Minor"]}
    assert {n.pitch % 12 for n in lead.notes} <= allowed
    assert all(n.start >= pm.get_end_time() - 1e-6 for n in lead.notes)


def test_continue_mode_leaves_the_existing_music_untouched():
    pm = _existing(8)
    before = [(n.pitch, n.start, n.end) for n in pm.instruments[0].notes]
    compose(_opts(mode="Continue editor"), existing=pm)
    assert before == [
        (n.pitch, n.start, n.end) for n in pm.instruments[0].notes
    ]


def test_continue_mode_with_an_empty_editor_starts_at_the_beginning():
    c = compose(
        _opts(mode="Continue editor"), existing=pretty_midi.PrettyMIDI()
    )
    assert c.start_bar == 0


def test_new_song_mode_ignores_the_existing_music():
    c = compose(_opts(mode="New song"), existing=_existing(8))
    assert c.start_bar == 0


@pytest.mark.parametrize("root", ["Auto", "", "H"])
def test_undefined_root_is_rejected(root):
    with pytest.raises(ValueError, match="root"):
        validate_options(_opts(root=root))


@pytest.mark.parametrize("scale", ["Auto (Detect)", "", "Nonsense"])
def test_undefined_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        validate_options(_opts(scale=scale))


@pytest.mark.parametrize(
    "bad",
    [
        {"minutes": 0},
        {"minutes": 500},
        {"style": "Polka"},
        {"structure": "Nope"},
        {"mood": "Angry"},
        {"mode": "Other"},
        {"roles": ("kazoo",)},
        {"texture_batches": -1},
        {"texture_batches": MAX_TEXTURE_BATCHES + 1},
        {"texture_batches": 1.5},
        {"roles": (), "texture_batches": 0},
    ],
)
def test_invalid_options_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_options(_opts(**bad))


def test_structure_and_texture_are_chosen_by_the_engine_by_default():
    options = ComposeOptions(root="A", scale="Minor")

    assert options.structure == AUTO_STRUCTURE
    assert options.texture_batches == 0
    validate_options(options)
    c = compose(_opts())
    assert not [i for i in c.instruments if i.name.startswith("Schema ")]


def test_style_structures_name_known_styles_and_structures():
    assert set(STYLE_STRUCTURES) <= set(SONG_STYLES)
    assert set(STYLE_STRUCTURES.values()) <= set(STRUCTURES)
    assert structure_for("Pop") == "Verse / Chorus"
    assert structure_for("EDM") == "Build & Drop"
    assert structure_for("Lo-Fi") == "Journey (A-B-A)"
    assert structure_for("Jazz") == "Journey (A-B-A)"
    assert structure_for("Unlisted") == "Verse / Chorus"


@pytest.mark.parametrize("style", SONG_STYLES)
def test_auto_structure_plans_the_structure_of_the_style(style):
    auto = compose(_opts(style=style, minutes=3))
    chosen = compose(
        _opts(style=style, minutes=3, structure=structure_for(style))
    )

    names = [s.name for s in auto.sections]
    assert names == [s.name for s in chosen.sections]
    assert len(names) > 1


def test_auto_structure_differs_between_edm_and_pop():
    edm = compose(_opts(style="EDM", minutes=3))
    pop = compose(_opts(style="Pop", minutes=3))

    assert "Drop" in {s.name for s in edm.sections}
    assert "Drop" not in {s.name for s in pop.sections}
    assert "Chorus" in {s.name for s in pop.sections}


def test_texture_alone_is_a_valid_song():
    c = compose(_opts(roles=(), texture_batches=1))
    assert all(i.name.startswith("Schema ") for i in c.instruments)
    assert min(_sounding_per_bar(c)) >= MIN_ACTIVE_ROLES


def test_non_seven_note_scales_use_a_matching_harmony():
    assert harmony_scale("Pentatonic Minor") == "Minor"
    assert harmony_scale("Pentatonic Major") == "Major"
    assert harmony_scale("Dorian") == "Dorian"


def test_modes_constant_lists_all_modes():
    assert MODES == ("New song", "Continue editor", "Layer over editor")


def _instrument_key(inst):
    return (True, 0) if inst.is_drum else (False, inst.program)


@pytest.mark.parametrize("batches", [0, 1, MAX_TEXTURE_BATCHES])
def test_tracks_hold_one_track_per_instrument(batches):
    c = compose(_opts(texture_batches=batches, minutes=3))
    keys = [_instrument_key(i) for i in c.tracks]
    assert len(keys) == len(set(keys))
    assert sum(i.is_drum for i in c.tracks) == 1
    assert len(c.tracks) <= len({_instrument_key(i) for i in c.instruments})


def test_a_full_texture_collapses_into_a_handful_of_tracks():
    c = compose(_opts(texture_batches=MAX_TEXTURE_BATCHES, minutes=3))
    assert len(c.instruments) >= MAX_TEXTURE_BATCHES * TRACKS_PER_ACTIVATION
    assert len(c.tracks) <= 24


def test_tracks_keep_every_generated_note_start():
    c = compose(_opts(texture_batches=2, minutes=3))
    kept = {
        _instrument_key(t): {(round(n.start, 6), n.pitch) for n in t.notes}
        for t in c.tracks
    }
    for inst in c.instruments:
        starts = {(round(n.start, 6), n.pitch) for n in inst.notes}
        assert starts <= kept[_instrument_key(inst)]


def test_track_names_are_friendly_not_general_midi():
    c = compose(_opts(roles=ROLE_KEYS, texture_batches=2, minutes=3))
    allowed = {*PROGRAM_NAMES, DRUMS_NAME}
    assert {t.name for t in c.tracks} <= allowed
    assert len({t.name for t in c.tracks}) == len(c.tracks)
    for track in c.tracks:
        if not track.is_drum:
            assert track.name == PROGRAM_NAMES[track.program]


def test_the_generated_parts_are_not_touched_by_the_layout():
    c = compose(_opts(texture_batches=1))
    assert any(i.name.startswith("Schema ") for i in c.instruments)
    assert not any(t.name.startswith("Schema ") for t in c.tracks)
    assert all(t.notes for t in c.tracks)
