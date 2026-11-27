import dataclasses

import pytest

from korshmer.midi_generation import composer, genres
from korshmer.midi_generation.composer import (
    COLOUR_ROLES,
    SONG_PALETTE,
    ComposeOptions,
    compose,
)
from korshmer.midi_generation.engine import SONG_STYLES

SIGNATURE_SOUNDS = {("Phonk", "melody"), ("Phonk", "bass")}


def _song(style, seed, minutes=1.5, **kw):
    return compose(
        ComposeOptions(
            "A",
            "Minor",
            bpm=120.0,
            minutes=minutes,
            style=style,
            seed=seed,
            texture_batches=0,
            **kw,
        )
    )


def test_every_general_midi_sound_belongs_to_some_style():
    found = set()
    for profile in genres.GENRES.values():
        for pool in profile.programs.values():
            found |= set(pool)
        found |= set(profile.texture_long.values())
        found |= set(profile.texture_short.values())
    assert found == set(range(128))


@pytest.mark.parametrize("style", SONG_STYLES)
def test_every_part_of_a_style_has_many_sounds(style):
    for role, pool in genres.profile_for(style).programs.items():
        if (style, role) in SIGNATURE_SOUNDS:
            continue
        assert len(set(pool)) >= 5, (style, role)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_a_song_changes_colours_between_its_sections(style):
    changed = 0
    for seed in range(1, 9):
        song = _song(style, seed)
        names = {
            inst.name.rstrip("0123456789 ")
            for inst in song.instruments
            if not inst.is_drum
        }
        voices = [inst for inst in song.instruments if not inst.is_drum]
        changed += len(voices) > len(names)
    assert changed >= 5


def test_repeated_sections_keep_their_colours():
    profile = genres.profile_for("Pop")
    programs = {role: pool[0] for role, pool in profile.programs.items()}
    first = composer.section_colours(7, 0, "Chorus", programs, profile)
    again = composer.section_colours(7, 0, "Chorus", programs, profile)
    assert first == again
    assert set(first) <= set(COLOUR_ROLES)
    other = [
        composer.section_colours(7, 0, name, programs, profile)
        for name in ("Verse", "Bridge", "Intro", "Outro", "Break")
    ]
    assert any(colours != first for colours in other)


@pytest.mark.parametrize("style", ["Pop", "Rock", "Jazz"])
def test_a_long_song_keeps_a_small_palette_but_songs_differ(style):
    union = {}
    for seed in range(1, 7):
        song = _song(style, seed, minutes=6)
        own = {}
        for inst in song.instruments:
            if inst.is_drum:
                continue
            role = inst.name.rstrip("0123456789 ")
            own.setdefault(role, set()).add(inst.program)
            union.setdefault(role, set()).add(inst.program)
        assert all(len(found) <= SONG_PALETTE for found in own.values())
    assert max(len(found) for found in union.values()) > SONG_PALETTE


def test_the_sounds_chosen_do_not_change_the_notes(monkeypatch):
    plain = _song("Pop", 5)
    profile = genres.GENRES["Pop"]
    wider = dataclasses.replace(
        profile,
        programs={
            role: (*pool, *range(96, 104))
            for role, pool in profile.programs.items()
        },
    )
    monkeypatch.setitem(genres.GENRES, "Pop", wider)
    other = _song("Pop", 5)

    def notes(song, word):
        return sorted(
            (round(n.start, 4), n.pitch)
            for inst in song.instruments
            if inst.name.startswith(word)
            for n in inst.notes
        )

    for word in ("Melody", "Bass", "Drums"):
        assert notes(plain, word) == notes(other, word)
