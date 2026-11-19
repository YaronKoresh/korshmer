import functools
import itertools
import statistics

import pretty_midi
import pytest

from korshmer.midi_generation.composer import (
    ANSWER_NAME,
    ROLE_KEYS,
    ComposeOptions,
    compose,
    lead_spans,
)
from korshmer.midi_generation.engine import MELODY_MOODS, SONG_STYLES

BPM = 120.0
BAR = 240.0 / BPM
BEAT = BAR / 4
SONG_BARS = 64
LAYER_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
MOOD_SEEDS = (21, 22, 23, 24, 25, 26)
CHORD_SEEDS = (3, 7, 11, 19, 23, 31, 42, 57, 64, 77, 85, 96)
HOME, FIFTH = 0, 4
BAND = ("Chords", "Pad", "Arp", "Pluck")
SAME_CHORD_LIMITS = {
    "Pop": 0.26,
    "Hip-Hop": 0.30,
    "Rock": 0.35,
    "Jazz": 0.30,
    "Phonk": 0.425,
}


@functools.cache
def _generated_song():
    made = compose(
        ComposeOptions(
            "A", "Minor", bpm=BPM, bars=SONG_BARS, style="Pop", seed=500
        )
    )
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    song.instruments.extend(made.instruments)
    return song


def _chords_only_song(bars=32, quiet=80, loud=80):
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    keys = pretty_midi.Instrument(4, name="Keys")
    triads = ((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62))
    for index in range(bars):
        level = quiet if index < bars // 2 else loud
        for pitch in triads[index % 4]:
            keys.notes.append(
                pretty_midi.Note(level, pitch, index * BAR, (index + 1) * BAR)
            )
    song.instruments.append(keys)
    return song


def _sung_song(bars=32):
    song = _chords_only_song(bars)
    voice = pretty_midi.Instrument(53, name="Vocals")
    for index in range(0, bars, 4):
        start = index * BAR
        for step in range(12):
            at = start + step * BEAT / 2
            voice.notes.append(
                pretty_midi.Note(84, 69 + step % 5, at, at + BEAT / 2)
            )
        held = start + 6 * BEAT
        voice.notes.append(pretty_midi.Note(84, 72, held, held + BEAT * 2))
    song.instruments.append(voice)
    return song


def _layer(song, seed, mood="Auto", style="Pop", roles=ROLE_KEYS):
    options = ComposeOptions(
        "A",
        "Minor",
        bpm=BPM,
        style=style,
        seed=seed,
        mood=mood,
        mode="Layer over editor",
        roles=roles,
    )
    return compose(options, existing=song)


def _notes(made, *words):
    return [
        note
        for inst in made.instruments
        if any((inst.name or "").startswith(word) for word in words)
        for note in inst.notes
    ]


def _drums(made):
    return [n for inst in made.instruments if inst.is_drum for n in inst.notes]


def _bar_of(note, made):
    return int((note.start - made.grid_offset) / BAR + 1e-6)


def _bars(made, word):
    bars = [set() for _ in range(made.total_bars)]
    for note in _notes(made, word):
        index = _bar_of(note, made)
        if 0 <= index < made.total_bars:
            step = round(
                (note.start - made.grid_offset) / BAR * 16 - index * 16
            )
            bars[index].add((step, note.pitch))
    return [frozenset(found) for found in bars]


def _same_bars(first, second, word):
    one, other = _bars(first, word), _bars(second, word)
    heard = [k for k in range(len(one)) if one[k] or other[k]]
    return sum(one[k] == other[k] for k in heard) / len(heard)


def _loudness(notes):
    return statistics.mean(note.velocity for note in notes)


def _per_bar(notes, made):
    return len(notes) / len({_bar_of(note, made) for note in notes})


def test_layers_over_one_song_choose_other_instruments_per_section_for_each_seed():
    song = _generated_song()
    layers = [_layer(song, seed) for seed in LAYER_SEEDS]
    plans = {
        tuple(section.roles for section in made.sections) for made in layers
    }
    parts = {
        tuple((s.name, s.start_bar, s.bars) for s in made.sections)
        for made in layers
    }

    assert len(plans) >= 6
    assert len(parts) == 1
    assert all(made.total_bars == SONG_BARS for made in layers)
    assert all(made.bar_chords == layers[0].bar_chords for made in layers)


def test_layered_pad_and_arp_do_not_play_the_same_bars_for_every_seed():
    song = _generated_song()
    layers = [_layer(song, seed) for seed in LAYER_SEEDS[:6]]
    pairs = list(itertools.combinations(layers, 2))

    for word in ("Pad", "Arp"):
        same = statistics.mean(_same_bars(a, b, word) for a, b in pairs)
        assert same <= 0.3, word


def test_a_layer_with_the_same_seed_is_the_same_and_another_seed_changes_most_bars():
    song = _generated_song()
    first, again, other = (_layer(song, seed) for seed in (4, 4, 5))

    for word in ("Bass", "Chords", "Pad", "Arp"):
        assert _same_bars(first, again, word) == 1.0, word
    for word in ("Chords", "Pad"):
        assert _same_bars(first, other, word) <= 0.3, word


def _hits_per_bar(notes, made):
    starts = {round(note.start, 2) for note in notes}
    return len(starts) / len({_bar_of(note, made) for note in notes})


def _mood_features(song, mood):
    layers = [_layer(song, seed, mood) for seed in MOOD_SEEDS]
    return {
        "chord hits": statistics.mean(
            _hits_per_bar(_notes(made, "Chords"), made) for made in layers
        ),
        "band loudness": statistics.mean(
            _loudness(_notes(made, *BAND)) for made in layers
        ),
        "pad pitch": statistics.mean(
            statistics.mean(n.pitch for n in _notes(made, "Pad"))
            for made in layers
        ),
        "bass notes": statistics.mean(
            _per_bar(_notes(made, "Bass"), made) for made in layers
        ),
    }


def test_each_mood_gives_the_layers_their_own_density_loudness_and_register():
    song = _generated_song()
    feel = {mood: _mood_features(song, mood) for mood in MELODY_MOODS}

    busy, calm = feel["Energetic"], feel["Melancholic"]
    assert busy["chord hits"] >= calm["chord hits"] + 0.4
    assert busy["bass notes"] >= feel["Dreamy"]["bass notes"] + 0.8
    assert busy["band loudness"] >= feel["Dreamy"]["band loudness"] + 5
    assert feel["Uplifting"]["band loudness"] >= calm["band loudness"] + 2
    assert feel["Dreamy"]["pad pitch"] >= calm["pad pitch"] + 3


def test_no_mood_moves_the_layers_far_from_the_loudness_of_the_song():
    song = _generated_song()
    level = _loudness(
        [n for inst in song.instruments if not inst.is_drum for n in inst.notes]
    )
    spread = []
    for mood in MELODY_MOODS:
        made = _layer(song, MOOD_SEEDS[0], mood)
        band = _loudness(_notes(made, *BAND))
        spread.append(band)
        assert abs(band - level) <= 10, mood
        assert _loudness(_notes(made, "Bass")) >= level, mood

    assert max(spread) - min(spread) >= 5


def test_layers_over_a_sung_song_differ_per_seed_but_never_play_over_the_singer():
    song = _sung_song(SONG_BARS)
    vocal = song.instruments[-1].notes
    spans = lead_spans(vocal, 0.0, SONG_BARS * BAR, BEAT)

    def while_sung(notes):
        inside = [n for n in notes if any(a <= n.start < b for a, b in spans)]
        return len(inside) / len(notes)

    plans = set()
    for seed, mood in zip(LAYER_SEEDS, itertools.cycle(MELODY_MOODS)):
        made = _layer(song, seed, mood)
        plans.add(tuple(section.roles for section in made.sections))
        answers = _notes(made, ANSWER_NAME)
        assert len(answers) >= 16, seed
        assert while_sung(answers) <= 0.1, seed
        pluck = _notes(made, "Pluck")
        assert not pluck or while_sung(pluck) <= 0.1, seed

    assert len(plans) >= 5


def test_a_layer_swells_in_the_last_bars_before_a_louder_part_of_the_song():
    song = _chords_only_song(quiet=45, loud=105)
    for seed in LAYER_SEEDS[:5]:
        made = _layer(song, seed)
        band = _notes(made, *BAND)
        early = _loudness([n for n in band if _bar_of(n, made) == 12])
        late = _loudness([n for n in band if _bar_of(n, made) == 15])
        assert late - early >= 3, seed


def test_the_first_verse_of_a_layer_has_lighter_drums_than_the_later_verses():
    song = _chords_only_song(bars=64)
    shares = []
    for seed in LAYER_SEEDS:
        made = _layer(song, seed)
        assert {section.name for section in made.sections} == {"Verse"}
        hits = _drums(made)
        first = sum(1 for n in hits if _bar_of(n, made) < 16) / 16
        later = sum(1 for n in hits if _bar_of(n, made) >= 16) / 48
        shares.append(first / later)

    assert statistics.mean(shares) <= 0.9


def test_a_layered_melody_plays_through_every_section_of_a_song_without_a_tune():
    song = _chords_only_song()
    for seed in LAYER_SEEDS[:4]:
        made = _layer(song, seed, roles=("melody", "chords", "pad"))
        bars = {_bar_of(note, made) for note in _notes(made, "Melody")}
        assert len(bars) >= 24, seed


@functools.cache
def _songs(style):
    return [
        compose(
            ComposeOptions(
                "A", "Minor", bpm=BPM, bars=SONG_BARS, style=style, seed=seed
            )
        )
        for seed in CHORD_SEEDS
    ]


def _degrees(song):
    return [
        section.progression[bar % len(section.progression)]
        for section in song.sections
        for bar in range(section.bars)
    ]


def _first_chords(style, names=None):
    return [
        section.progression[0]
        for song in _songs(style)
        for section in song.sections
        if names is None or section.name in names
    ]


@pytest.mark.parametrize("style", sorted(SAME_CHORD_LIMITS))
def test_two_songs_seldom_have_the_same_chord_in_the_same_bar(style):
    shares = [
        sum(
            a == b
            for a, b in zip(one.bar_chords, other.bar_chords, strict=True)
        )
        / SONG_BARS
        for one, other in itertools.combinations(_songs(style), 2)
    ]

    assert statistics.mean(shares) <= SAME_CHORD_LIMITS[style]


@pytest.mark.parametrize("style", ["Rock", "Hip-Hop", "Phonk"])
def test_sections_often_start_away_from_the_home_chord(style):
    first = _first_chords(style)

    assert sum(chord == HOME for chord in first) / len(first) <= 0.66


@pytest.mark.parametrize("style", ["EDM", "Hip-Hop", "Lo-Fi", "Jazz", "Phonk"])
def test_build_and_break_sections_never_start_on_the_home_chord(style):
    first = _first_chords(style, ("Build", "Break"))

    assert first
    assert HOME not in first


@pytest.mark.parametrize("style", SONG_STYLES)
def test_some_sections_hold_each_chord_for_two_bars_but_the_chorus_never_does(
    style,
):
    sections = [s for song in _songs(style) for s in song.sections[:-1]]
    held = [s for s in sections if len(s.progression) == 8]

    assert len(held) >= 4
    for section in held:
        assert section.progression[0::2] == section.progression[1::2]
        assert section.bars >= 8
        assert section.name not in ("Chorus", "Drop", "B")


@pytest.mark.parametrize("style", ["Pop", "Rock", "Hip-Hop", "Phonk"])
def test_a_dozen_songs_use_many_different_chord_orders(style):
    used = {s.progression for song in _songs(style) for s in song.sections}

    assert len(used) >= 50


@pytest.mark.parametrize("style", SONG_STYLES)
def test_songs_with_moved_chords_still_close_on_five_then_home(style):
    for song in _songs(style):
        chords = _degrees(song)
        assert len(chords) == SONG_BARS
        assert chords[-2:] == [FIFTH, HOME]
    assert any(
        len(s.progression) == 8 for song in _songs(style) for s in song.sections
    )


def test_phonk_moves_its_chords_but_the_home_chord_stays_the_most_used():
    opened_away = 0
    for song in _songs("Phonk"):
        chords = _degrees(song)
        counts = {chord: chords.count(chord) for chord in set(chords)}
        assert max(counts, key=counts.get) == HOME
        assert counts[HOME] / SONG_BARS >= 0.33
        assert all(
            s.key.split(" ", 1)[1] in ("Minor", "Phrygian")
            for s in song.sections
        )
        drops = [s for s in song.sections if s.name == "Drop"]
        assert all(len(set(s.progression[:3])) < 3 for s in drops)
        opened_away += sum(s.progression[0] != HOME for s in song.sections)

    assert opened_away >= 30
