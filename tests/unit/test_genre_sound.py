import pathlib
import struct

import pytest

from korshmer.midi_generation import ComposeOptions, compose
from korshmer.midi_generation.composer import ROLE_KEYS
from korshmer.midi_generation.engine import SONG_STYLES
from korshmer.midi_generation.genres import (
    CHIPTUNE_PROGRAMS,
    GENRES,
    KNOWN_KITS,
    PITCHED_ROLES,
    profile_for,
)
from korshmer.midi_generation.korshmer_schema import LONG_TRACKS, SHORT_TRACKS

SF3 = (
    pathlib.Path(__file__).resolve().parents[2] / "fluidsynth" / "soundfont.sf3"
)


def _song(style, seed=3, **kw):
    options = {
        "root": "A",
        "scale": "Minor",
        "bpm": 120.0,
        "minutes": 1.0,
        "style": style,
        "seed": seed,
        "roles": ROLE_KEYS,
        "texture_batches": 1,
    }
    options.update(kw)
    return compose(ComposeOptions(**options))


def _soundfont_presets():
    presets = set()
    with SF3.open("rb") as fh:
        fh.seek(12)
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            cid, size = struct.unpack("<4sI", head)
            if cid == b"LIST":
                kind = fh.read(4)
                end = fh.tell() + size - 4
                if kind == b"pdta":
                    while fh.tell() < end:
                        sub, sub_size = struct.unpack("<4sI", fh.read(8))
                        data = fh.read(sub_size + (sub_size & 1))
                        if sub == b"phdr":
                            for i in range(sub_size // 38 - 1):
                                rec = data[i * 38 : (i + 1) * 38]
                                prog, bank = struct.unpack("<HH", rec[20:24])
                                presets.add((bank, prog))
                    return presets
                fh.seek(end + (size & 1))
            else:
                fh.seek(size + (size & 1), 1)
    return presets


def test_every_style_has_a_profile():
    assert set(SONG_STYLES) <= set(GENRES)


def test_unknown_style_falls_back_to_the_default_profile():
    assert profile_for("Polka") is GENRES["Pop"]


@pytest.mark.parametrize("style", sorted(GENRES))
def test_each_profile_is_complete_and_valid(style):
    profile = GENRES[style]
    for role in PITCHED_ROLES:
        options = profile.programs[role]
        assert options
        assert all(0 <= p <= 127 for p in options)
    assert profile.kit in KNOWN_KITS
    assert len(profile.texture_long) >= LONG_TRACKS
    assert len(profile.texture_short) >= SHORT_TRACKS
    assert all(0 <= p <= 127 for p in profile.texture_long.values())
    assert all(0 <= p <= 127 for p in profile.texture_short.values())


@pytest.mark.parametrize("style", SONG_STYLES)
def test_role_instruments_come_from_the_genre_pool(style):
    profile = profile_for(style)
    for seed in range(4):
        c = _song(style, seed, texture_batches=0)
        for inst in c.instruments:
            if inst.is_drum:
                assert inst.program in (profile.kit, *profile.kits)
                continue
            role = next(
                r for r in PITCHED_ROLES if inst.name.lower().startswith(r)
            )
            assert inst.program in profile.programs[role], (style, inst.name)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_texture_instruments_and_drums_follow_the_genre(style):
    profile = profile_for(style)
    pool = set(profile.texture_long.values()) | set(
        profile.texture_short.values()
    )
    c = _song(style, texture_batches=2)
    schema = [i for i in c.instruments if i.name.startswith("Schema ")]
    assert schema
    for inst in schema:
        if inst.is_drum:
            assert inst.program == profile.kit
        else:
            assert inst.program in pool


@pytest.mark.parametrize("style", [s for s in SONG_STYLES if s != "EDM"])
def test_only_edm_may_use_the_chiptune_leads(style):
    for seed in range(8):
        c = _song(style, seed, texture_batches=2)
        used = {i.program for i in c.instruments if not i.is_drum}
        assert not used & set(CHIPTUNE_PROGRAMS), (style, seed)


def test_genres_use_different_drum_kits():
    kits = {s: profile_for(s).kit for s in SONG_STYLES}
    assert kits["Hip-Hop"] != kits["Pop"]
    assert kits["Jazz"] != kits["Pop"]
    assert kits["Rock"] != kits["Pop"]
    assert kits["EDM"] != kits["Pop"]
    assert len(set(kits.values())) >= 5


@pytest.mark.parametrize("style", SONG_STYLES)
def test_alternative_kits_are_known_and_differ_from_the_main_kit(style):
    profile = profile_for(style)
    assert profile.kits
    assert profile.kit not in profile.kits
    assert len(set(profile.kits)) == len(profile.kits)
    assert set(profile.kits) <= set(KNOWN_KITS)


def test_the_song_in_a_genre_is_not_the_same_sound_as_another_genre():
    sounds = {}
    for style in SONG_STYLES:
        c = _song(style, texture_batches=0)
        sounds[style] = {i.name: i.program for i in c.instruments}
    names = list(sounds)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert sounds[a] != sounds[b], (a, b)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_the_seed_changes_the_instruments_within_a_genre(style):
    seen = set()
    for seed in range(12):
        c = _song(style, seed, texture_batches=0)
        seen.add(tuple(sorted((i.name, i.program) for i in c.instruments)))
    assert len(seen) >= 3


def test_the_same_seed_gives_the_same_instruments():
    a = _song("Jazz", 5)
    b = _song("Jazz", 5)
    assert [(i.name, i.program) for i in a.instruments] == [
        (i.name, i.program) for i in b.instruments
    ]


def test_the_bundled_soundfont_has_every_program_and_kit_used():
    from korshmer.audio_processors import resolve_drum_kit

    presets = _soundfont_presets()
    available_kits = {prog for bank, prog in presets if bank == 128}
    for profile in GENRES.values():
        used = set(profile.texture_long.values()) | set(
            profile.texture_short.values()
        )
        for options in profile.programs.values():
            used |= set(options)
        for program in used:
            assert (0, program) in presets, (profile.name, program)
        # Every kit the genre uses must play: either the bundled SoundFont
        # carries it (true for the shipped GeneralUser GS, where this is an
        # identity) or it resolves to a related kit the bank does carry.
        for kit in (profile.kit, *profile.kits):
            resolved = resolve_drum_kit(kit, available_kits)
            assert (128, resolved) in presets, (profile.name, kit, resolved)
