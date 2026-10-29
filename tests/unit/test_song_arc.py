import random

import pytest

from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.midi_generation import engine, genres
from korshmer.midi_generation.composer import (
    ACT_BARS,
    ARC_REFLECTION_MUTED,
    ROLE_KEYS,
    ComposeOptions,
    act_kinds,
    compose,
    harmony_scale,
    mode_options,
    plan_acts,
)

BPM = 120.0
BAR = 240.0 / BPM
PITCHED = ("bass", "chords", "pad", "melody", "arp", "pluck", "counter")
LABELS = {
    "bass": "Bass",
    "chords": "Chords",
    "pad": "Pad",
    "melody": "Melody",
    "arp": "Arp",
    "pluck": "Pluck",
    "counter": "Counter-melody",
}


def _song(minutes=20, style="Pop", seed=1, scale="Major", root="C", **kw):
    return compose(
        ComposeOptions(
            root=root,
            scale=scale,
            bpm=BPM,
            minutes=minutes,
            style=style,
            seed=seed,
            roles=kw.pop("roles", ROLE_KEYS),
            texture_batches=0,
            **kw,
        )
    )


def _voices(comp, label):
    return [
        i for i in comp.instruments if i.name.rstrip("0123456789 ") == label
    ]


@pytest.mark.parametrize("count", range(1, 15))
def test_story_starts_at_home_climbs_to_one_climax_and_returns(count):
    kinds = act_kinds(count)
    assert len(kinds) == count
    assert kinds[0] == "Statement"
    if count >= 2:
        assert kinds.count("Climax") == 1
    if count >= 3:
        assert kinds[-1] == "Return"
        assert kinds.index("Climax") >= count // 2
    if count >= 5:
        assert "Reflection" in kinds
        assert kinds.index("Reflection") == kinds.index("Climax") - 1


def test_short_song_has_a_single_unchanged_act():
    comp = _song(minutes=1)
    assert comp.total_bars <= ACT_BARS
    assert {s.act for s in comp.sections} == {0}
    assert {s.act_kind for s in comp.sections} == {"Statement"}
    for label in LABELS.values():
        assert len(_voices(comp, label)) <= 1


@pytest.mark.parametrize("style", ["Pop", "Rock", "Hip-Hop", "Lo-Fi", "Jazz"])
@pytest.mark.parametrize("seed", range(3))
def test_every_pitched_role_changes_instrument_in_a_twenty_minute_song(
    style, seed
):
    comp = _song(style=style, seed=seed)
    for label in LABELS.values():
        programs = {v.program for v in _voices(comp, label)}
        assert len(programs) >= 2, (style, seed, label)


def test_instrument_changes_only_come_from_the_genre_pools():
    profile = genres.profile_for("Rock")
    comp = _song(style="Rock")
    for role, label in LABELS.items():
        for voice in _voices(comp, label):
            assert voice.program in profile.programs[role]


def test_first_voice_keeps_the_plain_role_name_and_notes_stay_in_range():
    comp = _song()
    names = [i.name for i in comp.instruments]
    assert len(names) == len(set(names))
    for label in LABELS.values():
        assert label in names
    for inst in comp.instruments:
        assert inst.notes
        assert all(0 <= n.pitch <= 127 for n in inst.notes)
        assert all(n.end <= comp.total_bars * BAR + 1e-6 for n in inst.notes)


def test_acts_get_other_modes_that_keep_the_notes_of_the_chosen_scale():
    comp = _song(root="D", scale="Major", seed=4)
    home = {(NOTE_NAMES.index("D") + i) % 12 for i in SCALES["Major"]}
    keys = {s.key for s in comp.sections}
    assert len(keys) >= 3
    assert comp.sections[0].key == "D Major"
    assert comp.sections[-1].key == "D Major"
    for inst in comp.instruments:
        if not inst.is_drum:
            assert {n.pitch % 12 for n in inst.notes} <= home


def test_key_changes_only_on_act_boundaries():
    comp = _song()
    by_act = {}
    for s in comp.sections:
        by_act.setdefault(s.act, set()).add(s.key)
    assert all(len(keys) == 1 for keys in by_act.values())
    acts = sorted(by_act)
    for a, b in zip(acts, acts[1:], strict=False):
        assert by_act[a] != by_act[b]


def test_return_act_brings_back_the_opening_instruments_and_chords():
    comp = _song()
    first = [s for s in comp.sections if s.act == 0]
    last = [s for s in comp.sections if s.act_kind == "Return"]
    assert last
    assert comp.sections[-1].act_kind == "Return"
    assert {s.progression for s in last} & {s.progression for s in first}
    rng = random.Random(1)
    base = {r: genres.profile_for("Pop").programs[r][0] for r in PITCHED}
    acts = plan_acts(rng, 8, "Major", base, genres.profile_for("Pop"))
    assert acts[-1].programs == acts[0].programs == base
    assert acts[-1].shift == 0
    assert acts[-1].scale == "Major"


def test_planned_instruments_differ_from_the_act_before():
    base = {r: genres.profile_for("Pop").programs[r][0] for r in PITCHED}
    for seed in range(10):
        acts = plan_acts(
            random.Random(seed), 10, "Minor", base, genres.profile_for("Pop")
        )
        for before, act in zip(acts, acts[1:], strict=False):
            if act.kind in ("Statement", "Return"):
                continue
            changed = [
                r for r in PITCHED if act.programs[r] != before.programs[r]
            ]
            assert len(changed) >= 3


def test_energy_has_a_valley_and_a_peak_across_the_acts():
    comp = _song()
    by_act = {}
    for s in comp.sections:
        by_act.setdefault(s.act_kind, []).append(s.energy)
    mean = {k: sum(v) / len(v) for k, v in by_act.items()}
    assert mean["Reflection"] < mean["Development"] < mean["Climax"]
    assert mean["Return"] <= mean["Climax"]


def test_reflection_act_drops_drums_bass_and_pluck():
    comp = _song()
    reflection = [s for s in comp.sections if s.act_kind == "Reflection"]
    assert reflection
    for s in reflection:
        assert not set(s.roles) & set(ARC_REFLECTION_MUTED)


def test_climax_act_lifts_the_melody_an_octave():
    comp = _song()
    climax = [s for s in comp.sections if s.act_kind == "Climax"]
    start = min(s.start_bar for s in climax) * BAR
    end = max(s.start_bar + s.bars for s in climax) * BAR
    others = [s for s in comp.sections if s.act_kind == "Development"]
    lo, hi = (
        min(s.start_bar for s in others) * BAR,
        (min(s.start_bar for s in others) + ACT_BARS) * BAR,
    )
    melody = [n for v in _voices(comp, "Melody") for n in v.notes]
    peak = [n.pitch for n in melody if start <= n.start < end]
    plain = [n.pitch for n in melody if lo <= n.start < hi]
    assert peak and plain
    assert sum(peak) / len(peak) > sum(plain) / len(plain)


@pytest.mark.parametrize(
    "scale", ["Major", "Minor", "Dorian", "Mixolydian", "Lydian"]
)
def test_mode_options_use_only_the_notes_of_the_scale(scale):
    options = mode_options(scale)
    assert options
    home = set(SCALES[scale])
    for shift, name in options:
        assert 0 < shift < 12
        assert {(shift + i) % 12 for i in SCALES[name]} == home


def test_scales_without_named_modes_stay_on_the_home_mode():
    assert mode_options("Harmonic Minor") == []
    comp = _song(scale="Harmonic Minor", root="A")
    assert {s.key for s in comp.sections} == {"A Harmonic Minor"}
    for label in LABELS.values():
        assert len(_voices(comp, label)) >= 1


@pytest.mark.parametrize("scale", ["Pentatonic Major", "Blues", "Chromatic"])
def test_pentatonic_blues_and_chromatic_use_their_harmony_scale(scale):
    comp = _song(scale=scale, minutes=10)
    assert comp.sections[0].key.endswith(harmony_scale(scale))


def test_same_seed_gives_the_same_story():
    a = _song(seed=9)
    b = _song(seed=9)
    assert [(s.act_kind, s.key) for s in a.sections] == [
        (s.act_kind, s.key) for s in b.sections
    ]
    assert [(i.name, i.program) for i in a.instruments] == [
        (i.name, i.program) for i in b.instruments
    ]
    other = _song(seed=10)
    assert [(i.name, i.program) for i in other.instruments] != [
        (i.name, i.program) for i in a.instruments
    ]


@pytest.mark.parametrize("style", ["Pop", "Rock", "Hip-Hop", "Lo-Fi", "Jazz"])
@pytest.mark.parametrize("seed", range(3))
def test_drum_kit_changes_with_the_story_and_returns_home(style, seed):
    profile = genres.profile_for(style)
    comp = _song(style=style, seed=seed)
    kits = [i.program for i in comp.instruments if i.is_drum]
    assert len(set(kits)) >= 2
    assert set(kits) <= {profile.kit, *profile.kits}
    first = _voices(comp, "Drums")[0]
    assert first.program == profile.kit
    home = min(s.start_bar for s in comp.sections if s.act_kind == "Return")
    holders = {
        v.program
        for v in _voices(comp, "Drums")
        if any(n.start >= home * BAR for n in v.notes)
    }
    assert holders == {profile.kit}


def test_planned_kit_differs_from_the_act_before_and_return_restores_it():
    profile = genres.profile_for("Pop")
    base = {r: profile.programs[r][0] for r in PITCHED}
    for seed in range(10):
        acts = plan_acts(random.Random(seed), 9, "Major", base, profile)
        assert acts[0].kit == acts[-1].kit == profile.kit
        for before, act in zip(acts, acts[1:], strict=False):
            if act.kind not in ("Statement", "Return"):
                assert act.kit != before.kit


def _act_starts(comp):
    starts = {}
    for s in comp.sections:
        starts.setdefault(s.act, s)
    return starts


def _notes_of(comp, label):
    return [n for v in _voices(comp, label) for n in v.notes]


@pytest.mark.parametrize("style", ["Pop", "Rock", "EDM", "Jazz"])
def test_every_new_act_with_drums_begins_with_a_crash(style):
    comp = _song(style=style)
    drums = _notes_of(comp, "Drums")
    for act, section in _act_starts(comp).items():
        if act == 0 or "drums" not in section.roles:
            continue
        crash = [
            n
            for n in drums
            if n.pitch == engine.CRASH
            and abs(n.start - section.start_bar * BAR) < 1e-6
        ]
        assert crash, (style, act)


def test_the_bar_before_the_climax_is_a_breath_without_the_rhythm_section():
    comp = _song()
    starts = _act_starts(comp)
    climax = next(s for s in starts.values() if s.act_kind == "Climax")
    lo, hi = (climax.start_bar - 1) * BAR, climax.start_bar * BAR
    for label in ("Drums", "Bass", "Arp", "Pluck"):
        assert not [n for n in _notes_of(comp, label) if lo <= n.start < hi]
        assert not [n for n in _notes_of(comp, label) if n.start < lo < n.end]
    others = [
        n
        for label in ("Chords", "Pad", "Melody", "Counter-melody")
        for n in _notes_of(comp, label)
        if n.start < hi and n.end > lo
    ]
    assert others


def test_breath_is_skipped_when_nothing_else_would_sound():
    comp = _song(minutes=20, roles=("drums", "bass"))
    starts = _act_starts(comp)
    climax = next(s for s in starts.values() if s.act_kind == "Climax")
    lo, hi = (climax.start_bar - 1) * BAR, climax.start_bar * BAR
    assert [
        n
        for label in ("Drums", "Bass")
        for n in _notes_of(comp, label)
        if n.start < hi and n.end > lo
    ]


@pytest.mark.parametrize("seed", range(4))
def test_no_bar_of_a_long_song_is_silent_with_the_marked_transitions(seed):
    comp = _song(seed=seed)
    sounding = [False] * comp.total_bars
    for inst in comp.instruments:
        for n in inst.notes:
            first = int(n.start / BAR + 1e-6)
            last = int((n.end - 1e-6) / BAR)
            for b in range(first, min(last, comp.total_bars - 1) + 1):
                sounding[b] = True
    assert all(sounding)


def _existing_lead():
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    lead = pretty_midi.Instrument(0)
    for i in range(40):
        lead.notes.append(
            pretty_midi.Note(90, 60 + (i * 3 % 9), i * 0.5, i * 0.5 + 0.4)
        )
    pm.instruments.append(lead)
    return pm


@pytest.mark.parametrize("style", ["Pop", "Rock", "Jazz"])
def test_continued_melody_changes_instrument_with_the_acts(style):
    comp = compose(
        ComposeOptions(
            root="C",
            scale="Major",
            bpm=BPM,
            minutes=20,
            style=style,
            seed=2,
            roles=("melody", "bass", "chords"),
            texture_batches=0,
            mode="Continue editor",
        ),
        existing=_existing_lead(),
    )
    leads = [i for i in comp.instruments if i.name.startswith("Melody")]
    assert len(leads) >= 2
    assert leads[0].name == "Melody (continued)"
    assert len({i.name for i in leads}) == len(leads)
    pool = genres.profile_for(style).programs["melody"]
    assert {i.program for i in leads} <= set(pool)
    assert len({i.program for i in leads}) == len(leads)
    start = comp.start_bar * BAR
    assert all(n.start >= start for i in leads for n in i.notes)
