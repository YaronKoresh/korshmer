import functools
import hashlib
import itertools
import os
import statistics
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import korshmer
from korshmer.midi_generation import seeds
from korshmer.midi_generation.composer import ComposeOptions, compose
from korshmer.midi_generation.engine import MELODY_MOODS, SONG_STYLES
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.ui.widgets.material.length_box import BarsPicker
from korshmer.ui.widgets.material.seed_box import SeedBox

BPM = 120.0
BAR = 240.0 / BPM
BEAT = BAR / 4
STEP = BAR / 16
VARIETY_SEEDS = tuple(range(501, 517))
MOOD_SEEDS = tuple(range(601, 607))
NIGHT_DRIVE = 15594235522922440856
LONG_SEED = 987654321098765432109876543210
MOOD_MARGINS = {
    "lead pitch": 3.0,
    "lead notes per bar": 0.4,
    "lead note length": 0.1,
    "lead loudness": 5.0,
    "lead leaps": 0.08,
    "minor chords": 0.06,
    "chord loudness": 5.0,
    "drum loudness": 4.0,
    "bass notes per bar": 0.4,
}
MOODS_APART_IN = 3


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@functools.cache
def _song(style, seed, mood="Auto", scale="Minor", bars=64, structure="Auto"):
    return compose(
        ComposeOptions(
            root="A" if scale == "Minor" else "C",
            scale=scale,
            bpm=BPM,
            bars=bars,
            style=style,
            mood=mood,
            seed=seed,
            structure=structure,
        )
    )


def _notes(song, role):
    return sorted(
        (
            note
            for part in song.instruments
            if (part.name or "").startswith(role)
            for note in part.notes
        ),
        key=lambda note: note.start,
    )


def _bar_of(note):
    return int(note.start / BAR + 1e-6)


def _bars(song, role, with_pitch=True):
    shapes = [set() for _ in range(song.total_bars)]
    for note in _notes(song, role):
        bar = _bar_of(note)
        step = round((note.start - bar * BAR) / STEP * 2) / 2
        shapes[bar].add((step, note.pitch) if with_pitch else step)
    return [frozenset(shape) for shape in shapes]


def _fingerprint(song):
    lines = [
        (
            part.name,
            part.program,
            part.is_drum,
            [
                (n.pitch, n.velocity, round(n.start, 6), round(n.end, 6))
                for n in part.notes
            ],
        )
        for part in song.instruments
    ]
    return hashlib.sha256(repr(lines).encode()).hexdigest()


def _hook_rhythm(song):
    return tuple((onset, length) for onset, length, _ in song.hook.notes)


def _pairs(style):
    songs = [_song(style, seed) for seed in VARIETY_SEEDS]
    return list(itertools.combinations(songs, 2))


def _share(pairs, same):
    return sum(1 for a, b in pairs if same(a, b)) / len(pairs)


def test_a_number_of_thirty_digits_is_used_whole_as_the_seed():
    assert seeds.seed_number(str(LONG_SEED)) == LONG_SEED
    assert seeds.seed_number(LONG_SEED) == LONG_SEED
    assert seeds.seed_number("987 654 321,098") == 987654321098
    song = compose(ComposeOptions("A", "Minor", bars=16, seed=str(LONG_SEED)))

    assert song.seed == LONG_SEED


def test_long_seeds_that_share_their_first_or_last_digits_give_other_songs():
    first = _fingerprint(_song("Pop", LONG_SEED, bars=16))
    other_end = _fingerprint(_song("Pop", LONG_SEED + 1, bars=16))
    other_start = _fingerprint(_song("Pop", LONG_SEED + 10**29, bars=16))
    other_middle = _fingerprint(_song("Pop", LONG_SEED + 10**15, bars=16))

    assert len({first, other_end, other_start, other_middle}) == 4


def test_a_word_or_phrase_is_turned_into_the_same_number_on_every_machine():
    assert seeds.seed_number("night drive") == NIGHT_DRIVE
    assert seeds.seed_number("  Night   DRIVE ") == NIGHT_DRIVE
    assert seeds.seed_number("night drives") != NIGHT_DRIVE
    assert seeds.seed_number("שיר של לילה") == seeds.seed_number("שיר של לילה")
    assert seeds.seed_number("שיר של לילה") != seeds.seed_number("שיר של יום")
    assert seeds.seed_label("Night Drive") == f'"Night Drive" = {NIGHT_DRIVE}'
    assert seeds.seed_label("0042") == "42"


def test_a_phrase_seed_gives_the_same_song_as_its_number():
    by_text = compose(ComposeOptions("A", "Minor", bars=16, seed="night drive"))
    by_number = compose(ComposeOptions("A", "Minor", bars=16, seed=NIGHT_DRIVE))

    assert by_text.seed == NIGHT_DRIVE
    assert _fingerprint(by_text) == _fingerprint(by_number)


def test_the_same_phrase_gives_the_same_song_in_separate_python_runs():
    script = (
        "import hashlib\n"
        "from korshmer.midi_generation.composer import ComposeOptions, compose\n"
        "song = compose(ComposeOptions('A', 'Minor', bpm=120.0, bars=16, "
        "style='Pop', seed='night drive'))\n"
        "lines = [(p.name, p.program, p.is_drum, [(n.pitch, n.velocity, "
        "round(n.start, 6), round(n.end, 6)) for n in p.notes]) "
        "for p in song.instruments]\n"
        "print(hashlib.sha256(repr(lines).encode()).hexdigest())\n"
    )
    source = os.path.dirname(os.path.dirname(korshmer.__file__))
    found = set()
    for salt in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=salt, PYTHONPATH=source)
        run = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        found.add(run.stdout.strip())
    here = _fingerprint(
        compose(
            ComposeOptions(
                "A", "Minor", bpm=BPM, bars=16, style="Pop", seed="night drive"
            )
        )
    )

    assert found == {here}


@pytest.mark.parametrize("empty", [None, "", "   ", "0", "000", "random"])
def test_an_empty_seed_picks_a_fresh_long_seed_and_reports_it(empty):
    assert seeds.seed_number(empty) is None
    first = compose(ComposeOptions("A", "Minor", bars=8, seed=empty))
    second = compose(ComposeOptions("A", "Minor", bars=8, seed=empty))
    again = compose(ComposeOptions("A", "Minor", bars=8, seed=first.seed))

    assert len(str(first.seed)) == 18
    assert first.seed != second.seed
    assert _fingerprint(first) != _fingerprint(second)
    assert _fingerprint(again) == _fingerprint(first)


@pytest.mark.parametrize("seed", [1.5, True, ("a",), "x" * 500])
def test_a_seed_that_is_no_number_or_short_text_is_refused(seed):
    with pytest.raises(ValueError, match="seed"):
        compose(ComposeOptions("A", "Minor", bars=8, seed=seed))


def test_the_seed_field_takes_long_numbers_and_phrases_and_empty_means_random(
    qapp,
):
    field = SeedBox()

    assert field.seed() is None
    assert field.value() == 0
    assert field.placeholderText() == "Random"
    field.setText("123456789012345678901234")
    assert field.seed() == 123456789012345678901234
    assert field.value() == 123456789012345678901234
    field.setText("night drive")
    assert field.seed() == NIGHT_DRIVE
    assert field.label() == f'"night drive" = {NIGHT_DRIVE}'
    field.setValue(77)
    assert field.text() == "77"
    assert field.seed() == 77
    field.setValue(0)
    assert field.text() == ""
    assert field.seed() is None


def test_the_generator_names_the_seed_it_used_and_keeps_an_empty_field_empty(
    qapp,
):
    class Window(EditorActionsMixin):
        pass

    window = Window()
    window.spin_gen_seed = SeedBox()

    assert window._generator_seed() is None
    assert window._generator_seed_label(424242424242424242) == (
        "424242424242424242"
    )
    assert window.spin_gen_seed.text() == ""
    window.spin_gen_seed.setText("Night Drive")
    assert window._generator_seed() == NIGHT_DRIVE
    assert window._generator_seed_label(NIGHT_DRIVE) == (
        f'"Night Drive" = {NIGHT_DRIVE}'
    )


@pytest.mark.parametrize("style", SONG_STYLES)
def test_two_seeds_rarely_share_the_rhythm_of_their_melody_hook(style):
    pairs = _pairs(style)
    same = _share(pairs, lambda a, b: _hook_rhythm(a) == _hook_rhythm(b))
    rhythms = {_hook_rhythm(a) for a, _ in pairs}

    assert len(pairs) >= 30
    assert same <= 0.15
    assert len(rhythms) >= 8


@pytest.mark.parametrize("style", SONG_STYLES)
def test_two_seeds_rarely_give_the_same_instruments_to_the_same_sections(
    style,
):
    def plan(song):
        return [(s.name, s.bars, s.roles) for s in song.sections]

    assert _share(_pairs(style), lambda a, b: plan(a) == plan(b)) <= 0.1
    role_plans = {
        tuple(
            dict.fromkeys(
                (s.name, s.roles) for s in _song(style, seed).sections
            )
        )
        for seed in VARIETY_SEEDS
    }
    assert len(role_plans) >= 6


@pytest.mark.parametrize("style", SONG_STYLES)
def test_two_seeds_share_few_identical_pad_and_bass_bars(style):
    for role, limit in (("Pad", 0.2), ("Bass", 0.15)):
        shares = []
        for a, b in _pairs(style):
            mine, theirs = _bars(a, role), _bars(b, role)
            heard = [i for i in range(64) if mine[i] or theirs[i]]
            shares.append(sum(mine[i] == theirs[i] for i in heard) / len(heard))
        assert statistics.mean(shares) <= limit, role


@pytest.mark.parametrize("style", SONG_STYLES)
def test_every_style_draws_on_many_chord_progressions_and_grooves(style):
    songs = [_song(style, seed) for seed in VARIETY_SEEDS]
    progressions = {s.progression for song in songs for s in song.sections[1:2]}
    grooves = {
        Counter(bar for bar in _bars(song, "Drums") if bar).most_common(1)[0][0]
        for song in songs
    }

    assert len(progressions) >= 6
    assert len(grooves) >= 5


@pytest.mark.parametrize("style", ["Pop", "Rock", "Hip-Hop", "Lo-Fi", "Jazz"])
def test_a_song_of_any_seed_is_still_built_from_a_repeated_hook(style):
    for seed in VARIETY_SEEDS[:6]:
        song = _song(style, seed)
        rhythms = Counter(
            bar for bar in _bars(song, "Melody", with_pitch=False) if bar
        )
        assert rhythms.most_common(1)[0][1] / sum(rhythms.values()) >= 0.25
        assert len(rhythms) >= 3


@pytest.mark.parametrize("style", ["Pop", "Rock", "Hip-Hop"])
def test_the_first_verse_has_a_lighter_drum_groove_than_the_verse_after_it(
    style,
):
    lighter = 0
    for seed in VARIETY_SEEDS[:8]:
        song = _song(style, seed, bars=96, structure="Verse / Chorus")
        verses = [s for s in song.sections if s.name == "Verse"]
        hits = Counter(_bar_of(note) for note in _notes(song, "Drums"))
        first, later = (
            statistics.mean(
                hits[bar] for bar in range(s.start_bar, s.start_bar + 3)
            )
            for s in verses[:2]
        )
        lighter += first < later
    assert lighter >= 5


def test_a_build_section_grows_louder_towards_the_drop():
    rises = []
    for seed in VARIETY_SEEDS[:8]:
        song = _song("EDM", seed, bars=96, structure="Build & Drop")
        build = next(s for s in song.sections if s.name == "Build")
        middle = (build.start_bar + build.bars / 2) * BAR
        end = (build.start_bar + build.bars) * BAR
        band = [
            note
            for role in ("Bass", "Chords", "Pad")
            for note in _notes(song, role)
            if build.start_bar * BAR <= note.start < end
        ]
        early = [n.velocity for n in band if n.start < middle]
        late = [n.velocity for n in band if n.start >= middle]
        rises.append(statistics.mean(late) - statistics.mean(early))
    assert min(rises) > 0
    assert statistics.mean(rises) >= 5


def _mood_features(song):
    lead = _notes(song, "Melody")
    chords = _notes(song, "Chords")
    drums = _notes(song, "Drums")
    bass = _notes(song, "Bass")
    moves = [
        abs(b.pitch - a.pitch) for a, b in zip(lead, lead[1:], strict=False)
    ]
    minor = [
        any(
            (pc + 3) % 12 in chord and (pc + 4) % 12 not in chord
            for pc in chord
            if (pc + 7) % 12 in chord or (pc + 6) % 12 in chord
        )
        for chord in song.bar_chords
        if chord
    ]
    return {
        "lead pitch": statistics.mean(n.pitch for n in lead),
        "lead notes per bar": len(lead) / len({_bar_of(n) for n in lead}),
        "lead note length": statistics.mean(
            (n.end - n.start) / BEAT for n in lead
        ),
        "lead loudness": statistics.mean(n.velocity for n in lead),
        "lead leaps": sum(1 for m in moves if m >= 5) / len(moves),
        "minor chords": sum(minor) / len(minor),
        "chord loudness": statistics.mean(n.velocity for n in chords),
        "drum loudness": statistics.mean(n.velocity for n in drums),
        "bass notes per bar": len(bass) / len({_bar_of(n) for n in bass}),
    }


@functools.cache
def _mood_rows(style):
    return [
        (
            mood,
            _mood_features(
                _song(style, seed, mood, ("Minor", "Major")[seed % 2], 32)
            ),
        )
        for mood in MELODY_MOODS
        for seed in MOOD_SEEDS
    ]


@functools.cache
def _mood_means():
    means = {}
    for mood in MELODY_MOODS:
        rows = [
            features
            for style in SONG_STYLES
            for name, features in _mood_rows(style)
            if name == mood
        ]
        means[mood] = {
            key: statistics.mean(row[key] for row in rows) for key in rows[0]
        }
    return means


@pytest.mark.parametrize("pair", list(itertools.combinations(MELODY_MOODS, 2)))
def test_every_two_moods_differ_clearly_in_at_least_three_musical_ways(pair):
    means = _mood_means()
    apart = [
        key
        for key, margin in MOOD_MARGINS.items()
        if abs(means[pair[0]][key] - means[pair[1]][key]) >= margin
    ]

    assert len(apart) >= MOODS_APART_IN, apart


def test_each_mood_has_the_character_its_name_promises():
    means = _mood_means()

    def others(mood, key):
        return [means[name][key] for name in MELODY_MOODS if name != mood]

    energetic, dreamy = means["Energetic"], means["Dreamy"]
    melancholic, uplifting = means["Melancholic"], means["Uplifting"]
    assert energetic["lead note length"] <= 0.8 * min(
        others("Energetic", "lead note length")
    )
    assert energetic["lead notes per bar"] >= 0.4 + max(
        others("Energetic", "lead notes per bar")
    )
    assert energetic["chord loudness"] > max(
        others("Energetic", "chord loudness")
    )
    assert dreamy["lead leaps"] >= 0.08 + max(others("Dreamy", "lead leaps"))
    assert dreamy["lead loudness"] < min(others("Dreamy", "lead loudness"))
    assert dreamy["lead pitch"] >= 5 + melancholic["lead pitch"]
    assert melancholic["lead pitch"] <= min(others("Melancholic", "lead pitch"))
    assert melancholic["minor chords"] >= 0.08 + uplifting["minor chords"]
    assert uplifting["minor chords"] < min(others("Uplifting", "minor chords"))
    assert uplifting["lead pitch"] >= 3 + melancholic["lead pitch"]


@pytest.mark.parametrize("style", SONG_STYLES)
def test_the_mood_of_a_song_can_be_told_from_its_notes(style):
    rows = _mood_rows(style)
    names = [mood for mood, _ in rows]
    matrix = np.array([[row[key] for key in MOOD_MARGINS] for _, row in rows])
    scaled = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-9)
    right = 0
    for index, mood in enumerate(names):
        centres = {
            name: scaled[
                [i for i, n in enumerate(names) if n == name and i != index]
            ].mean(0)
            for name in MELODY_MOODS
        }
        guess = min(
            centres,
            key=lambda name: np.linalg.norm(scaled[index] - centres[name]),
        )
        right += guess == mood

    assert right / len(rows) >= 0.75


def test_auto_mood_plays_a_major_key_brighter_and_louder_than_a_minor_key():
    def loudness(scale):
        return statistics.mean(
            _mood_features(_song(style, seed, "Auto", scale, 32))[
                "lead loudness"
            ]
            for style in ("Pop", "Jazz", "Lo-Fi")
            for seed in MOOD_SEEDS
        )

    assert loudness("Major") >= loudness("Minor") + 6


@pytest.mark.parametrize("style", SONG_STYLES)
@pytest.mark.parametrize("bars", [5, 13, 30, 47])
def test_a_song_has_exactly_the_bars_asked_for_even_between_groups_of_four(
    style, bars
):
    song = _song(style, 7, bars=bars)
    last_end = max(n.end for part in song.instruments for n in part.notes)
    sounding = {_bar_of(n) for part in song.instruments for n in part.notes}

    assert song.total_bars == bars
    assert sum(section.bars for section in song.sections) == bars
    assert last_end <= bars * BAR + 1e-6
    assert sounding == set(range(bars))
    assert all(section.bars % 4 == 0 for section in song.sections[:-1])


def test_a_song_of_thirteen_bars_still_closes_on_its_home_chord(qapp):
    song = _song("Pop", 7, bars=13)
    picker = BarsPicker(4, 2048, 64, 4, tempo=BPM)
    picker.box.setValue(13)

    assert song.total_bars == 13
    assert len(song.bar_chords) == 13
    assert song.bar_chords[-1] == {9, 0, 4}
    assert song.bar_chords[-2] == {4, 7, 11}
    assert picker.box.value() == 13
    assert picker.time.text() == "About 0:26 at 120 BPM"
