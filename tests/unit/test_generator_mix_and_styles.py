import random

import pretty_midi
import pytest

from korshmer.midi_generation import composer, genre_feel, genres
from korshmer.midi_generation.composer import (
    AUTO_STYLE,
    RHYTHM_LEAD,
    STYLE_CHOICES,
    ComposeOptions,
    compose,
    lift_rhythm_section,
    match_song_level,
    resolve_style,
    song_bar_gains,
)
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 120.0
BAR = 240.0 / BPM


def _mean(notes):
    return sum(n.velocity for n in notes) / len(notes)


def _role_notes(composition):
    found = {"drums": [], "bass": [], "melody": []}
    for inst in composition.instruments:
        name = (inst.name or "").lower()
        if inst.is_drum:
            found["drums"] += inst.notes
        elif "bass" in name:
            found["bass"] += inst.notes
        elif "melody" in name:
            found["melody"] += inst.notes
    return found


def _song(bars=32, quiet=40, loud=110):
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    lead = pretty_midi.Instrument(0, name="Vocals")
    keys = pretty_midi.Instrument(4, name="Keys")
    for index in range(bars):
        velocity = quiet if index < bars // 2 else loud
        for step in range(4):
            start = index * BAR + step * BAR / 4
            lead.notes.append(
                pretty_midi.Note(
                    velocity, 69 + (step * 2) % 7, start, start + 0.3
                )
            )
        for pitch in (57, 60, 64):
            keys.notes.append(
                pretty_midi.Note(
                    velocity, pitch, index * BAR, (index + 1) * BAR
                )
            )
    song.instruments += [lead, keys]
    return song


def test_phonk_is_a_complete_style():
    assert "Phonk" in SONG_STYLES
    assert "Phonk" in genres.GENRES
    assert "Phonk" in genre_feel.FEELS
    assert genres.GENRES["Phonk"].kit == genres.KIT_808
    assert genre_feel.FEELS["Phonk"].drum_vel > 1.0
    for seed in range(1, 31):
        made = compose(
            ComposeOptions(
                "A", "Minor", bpm=140.0, minutes=1.0, style="Phonk", seed=seed
            )
        )
        assert made.style == "Phonk"
        roles = _role_notes(made)
        assert roles["drums"] and roles["bass"] and roles["melody"]
        drummed_bars = sum(
            section.bars
            for section in made.sections
            if "drums" in section.roles
        )
        assert len(roles["drums"]) / drummed_bars >= 8


@pytest.mark.parametrize(
    "bpm, scale, expected",
    [
        (70.0, "Minor", "Lo-Fi"),
        (70.0, "Major", "Jazz"),
        (95.0, "Minor", "Hip-Hop"),
        (95.0, "Major", "Pop"),
        (118.0, "Minor", "Pop"),
        (128.0, "Major", "EDM"),
        (145.0, "Minor", "Phonk"),
        (170.0, "Major", "Rock"),
    ],
)
def test_auto_style_follows_tempo_and_scale(bpm, scale, expected):
    blend = resolve_style(AUTO_STYLE, bpm, scale)
    assert genres.primary(blend) == expected
    assert set(genres.parts(blend)) <= set(SONG_STYLES)
    assert len(set(genres.parts(blend))) == len(genres.parts(blend))
    assert 1 <= len(genres.parts(blend)) <= 1 + composer.AUTO_BLEND_PARTNERS
    assert resolve_style("Jazz", bpm, scale) == "Jazz"
    assert expected in SONG_STYLES


def test_auto_is_the_default_style_and_reports_what_it_chose():
    assert STYLE_CHOICES[0] == AUTO_STYLE
    assert set(STYLE_CHOICES[1:]) == set(SONG_STYLES)
    options = ComposeOptions("A", "Minor", bpm=145.0, minutes=0.5, seed=2)
    assert options.style == AUTO_STYLE
    assert genres.primary(compose(options).style) == "Phonk"
    with pytest.raises(ValueError, match="Unknown style"):
        compose(ComposeOptions("A", "Minor", style="Polka"))


@pytest.mark.parametrize("style", SONG_STYLES)
def test_drums_and_bass_sit_above_the_melody_in_a_new_song(style):
    made = compose(
        ComposeOptions("C", "Major", bpm=BPM, minutes=1.0, style=style, seed=5)
    )
    roles = _role_notes(made)
    lead = _mean(roles["melody"])
    for role in ("drums", "bass"):
        assert _mean(roles[role]) >= min(127, lead + RHYTHM_LEAD) - 2, role


def test_a_layer_follows_the_loudness_of_the_song_below():
    song = _song()
    made = compose(
        ComposeOptions(
            "A",
            "Minor",
            bpm=BPM,
            style="Pop",
            seed=3,
            mode="Layer over editor",
            roles=("drums", "bass", "chords", "arp"),
        ),
        existing=song,
    )
    half = made.total_bars // 2 * BAR + made.grid_offset
    notes = [n for inst in made.instruments for n in inst.notes]
    quiet = _mean([n for n in notes if n.start < half - BAR])
    loud = _mean([n for n in notes if n.start >= half + BAR])
    assert loud - quiet >= 15
    roles = _role_notes(made)
    lead = _mean(song.instruments[0].notes)
    for role in ("drums", "bass"):
        assert lead <= _mean(roles[role]) <= lead + RHYTHM_LEAD + 6, role
    others = [
        n
        for inst in made.instruments
        if not inst.is_drum and "bass" not in (inst.name or "").lower()
        for n in inst.notes
    ]
    assert abs(_mean(others) - lead) <= 10


def test_song_gains_are_relative_and_bounded():
    gains = song_bar_gains(_song(bars=8, quiet=20, loud=120), BAR, 10)
    low, high = composer.SONG_GAIN_RANGE
    assert gains[:4] == [low] * 4
    assert all(1.0 <= g <= high for g in gains[4:8])
    assert gains[8:] == [composer.SONG_QUIET_GAIN] * 2
    empty = pretty_midi.PrettyMIDI()
    assert song_bar_gains(empty, BAR, 3) == [1.0, 1.0, 1.0]


def _voice(velocity, count=4, drum=False):
    inst = pretty_midi.Instrument(0, is_drum=drum)
    inst.notes = [
        pretty_midi.Note(velocity, 60, i * 0.5, i * 0.5 + 0.2)
        for i in range(count)
    ]
    return inst


def test_rhythm_section_is_lifted_but_never_lowered():
    voices = [
        ("drums", _voice(70, drum=True)),
        ("bass", _voice(120)),
        ("melody", _voice(90)),
        ("pad", _voice(50)),
    ]
    lift_rhythm_section(voices)
    levels = {role: _mean(inst.notes) for role, inst in voices}
    assert levels == {"drums": 96, "bass": 120, "melody": 90, "pad": 50}
    quiet = [("bass", _voice(40))]
    lift_rhythm_section(quiet, _voice(60).notes)
    assert _mean(quiet[0][1].notes) == 66
    untouched = [("bass", _voice(40))]
    lift_rhythm_section(untouched)
    assert _mean(untouched[0][1].notes) == 40


def test_layer_level_is_set_from_the_song():
    voices = [("drums", _voice(110, drum=True)), ("chords", _voice(100))]
    match_song_level(voices, _song(bars=2, quiet=50, loud=50))
    assert _mean(voices[0][1].notes) == 50 + RHYTHM_LEAD
    assert _mean(voices[1][1].notes) == 50


def test_fresh_choices_do_not_repeat_until_all_were_used():
    rng = random.Random(1)
    used = []
    picks = [composer._fresh(rng, ("a", "b", "c"), used) for _ in range(3)]
    assert sorted(picks) == ["a", "b", "c"]
    assert composer._fresh(rng, ("a", "b", "c"), used) in ("a", "b", "c")


def test_layer_sections_take_their_energy_from_the_song():
    made = compose(
        ComposeOptions(
            "A", "Minor", bpm=BPM, style="Pop", seed=3, mode="Layer over editor"
        ),
        existing=_song(),
    )
    energies = [section.energy for section in made.sections]
    assert max(energies) - min(energies) >= 0.3
    assert energies[0] < energies[-1]


@pytest.mark.parametrize("bpm", [70.0, 95.0, 118.0, 128.0, 145.0])
def test_auto_mixes_the_styles_that_fit_the_tempo(bpm):
    blend = resolve_style(AUTO_STYLE, bpm, "Minor")
    assert len(genres.parts(blend)) >= 2
    lead, *partners = genres.parts(blend)
    best = max(composer.style_fit(s, bpm, "Minor") for s in SONG_STYLES)
    for partner in partners:
        fit = composer.style_fit(partner, bpm, "Minor")
        assert fit >= composer.AUTO_BLEND_KEEP * best


def test_auto_partners_change_with_the_tempo():
    slow = set(genres.parts(resolve_style(AUTO_STYLE, 75.0, "Minor")))
    fast = set(genres.parts(resolve_style(AUTO_STYLE, 145.0, "Minor")))
    assert slow != fast
    assert "Phonk" not in slow
    assert "Lo-Fi" not in fast


def test_a_blend_takes_the_lead_feel_and_widens_its_pools():
    blend = genres.blend_name(("Hip-Hop", "Lo-Fi", "Jazz"))
    lead = genre_feel.feel_for("Hip-Hop")
    mixed = genre_feel.feel_for(blend)
    assert mixed.swing == lead.swing
    assert mixed.phrases == lead.phrases
    for name in ("Hip-Hop", "Lo-Fi", "Jazz"):
        feel = genre_feel.feel_for(name)
        assert all(g in mixed.grooves for g in feel.grooves)
        assert all(p in mixed.progressions for p in feel.progressions)
    share = sum(g in lead.grooves for g in mixed.grooves) / len(mixed.grooves)
    assert share > 1 / 3
    profile = genres.profile_for(blend)
    assert profile.kit == genres.profile_for("Hip-Hop").kit
    assert profile.kit not in profile.kits
    for role in genres.PITCHED_ROLES:
        for name in ("Hip-Hop", "Lo-Fi", "Jazz"):
            assert set(genres.profile_for(name).programs[role]) <= set(
                profile.programs[role]
            )


def test_a_blend_of_unknown_names_falls_back_like_a_single_style():
    assert genres.profile_for("Polka + Waltz") is genres.GENRES["Pop"]
    assert genres.profile_for("Jazz + Polka") is genres.GENRES["Jazz"]
    assert genre_feel.feel_for("Jazz + Polka") is genre_feel.FEELS["Jazz"]


def test_an_auto_song_uses_instruments_from_every_style_it_mixes():
    seen = {}
    for seed in range(1, 9):
        made = compose(
            ComposeOptions(
                "A",
                "Minor",
                bpm=95.0,
                minutes=1.0,
                seed=seed,
                texture_batches=0,
            )
        )
        lead, *partners = genres.parts(made.style)
        for inst in made.instruments:
            if inst.is_drum:
                continue
            for name in (lead, *partners):
                pools = genres.profile_for(name).programs.values()
                if any(inst.program in pool for pool in pools):
                    seen.setdefault(name, set()).add(inst.program)
    assert len(seen) >= 2


@pytest.mark.parametrize(
    "bpm, minor, expected",
    [
        (70.0, True, {"Melancholic", "Dreamy"}),
        (140.0, True, {"Melancholic", "Dreamy", "Energetic"}),
        (70.0, False, {"Dreamy", "Uplifting"}),
        (140.0, False, {"Uplifting", "Energetic"}),
    ],
)
def test_auto_mood_follows_tempo_and_scale(bpm, minor, expected):
    assert set(composer.auto_moods(bpm, minor)) == expected


@pytest.mark.parametrize("style", SONG_STYLES)
def test_every_style_offers_several_sounds_for_its_band(style):
    programs = genres.profile_for(style).programs
    for role in ("chords", "pad", "arp", "pluck", "counter"):
        assert len(set(programs[role])) >= 4, (style, role)
