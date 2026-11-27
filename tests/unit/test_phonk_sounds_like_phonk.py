import random
import statistics
from collections import Counter
from statistics import mean, median

import numpy as np
import pretty_midi
import pytest

from korshmer import audio_processors
from korshmer.audio_processing import guitar_amp
from korshmer.midi_generation import genre_feel, genres
from korshmer.midi_generation.composer import (
    CHOP_SECTIONS,
    ROLE_KEYS,
    ComposeOptions,
    compose,
)
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 140.0
BAR = 240.0 / BPM
SIXTEENTH = BAR / 16
BEAT = BAR / 4
SEEDS = (1, 2, 3, 4, 5, 6)
KICK, SNARE, CLAP, CLOSED_HAT = 36, 38, 39, 42
AGOGO_BELL = 113
BRIGHT_TOY_PROGRAMS = {4, 5, 9, 80, 81, 84, 112, 114}
SAMPLE_RATE = 44100


def _flavor_of(seed):
    return genres.flavored("Phonk", random.Random(f"flavor:{seed}"))


def _seed_of(seed, flavor):
    while _flavor_of(seed) != flavor:
        seed += 1000
    return seed


def _phonk(seed, root="A", scale="Minor", flavor="Phonk", **kw):
    return compose(
        ComposeOptions(
            root,
            scale,
            bpm=BPM,
            minutes=1.5,
            style="Phonk",
            seed=_seed_of(seed, flavor),
            **kw,
        )
    )


def _voices(song, word):
    return [
        inst
        for inst in song.instruments
        if not inst.is_drum and word in (inst.name or "")
    ]


def _notes(song, word):
    return [note for inst in _voices(song, word) for note in inst.notes]


def _drum_hits(song, *pitches):
    return [
        note
        for inst in song.instruments
        if inst.is_drum
        for note in inst.notes
        if note.pitch in pitches
    ]


def _bar_of(note):
    return int(note.start / BAR + 1e-6)


def _step_of(note):
    return round((note.start - _bar_of(note) * BAR) / SIXTEENTH, 2)


def _drop_bars(song):
    return {
        bar
        for section in song.sections
        if section.name == "Drop"
        for bar in range(section.start_bar, section.start_bar + section.bars)
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_lead_is_played_only_by_the_bell_that_stands_in_for_a_cowbell(
    seed,
):
    song = _phonk(seed)
    lead = _voices(song, "Melody")
    assert lead
    assert {inst.program for inst in lead} == {AGOGO_BELL}


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_lead_is_a_dense_riff_of_short_sixteenth_notes_in_the_drop(seed):
    song = _phonk(seed)
    drop = _drop_bars(song)
    riff = [note for note in _notes(song, "Melody") if _bar_of(note) in drop]
    assert riff
    bars = {_bar_of(note) for note in riff}
    assert len(riff) / len(bars) >= 6
    assert median(note.end - note.start for note in riff) <= SIXTEENTH * 2
    off_the_eighths = [note for note in riff if round(_step_of(note)) % 2]
    assert len(off_the_eighths) / len(riff) >= 0.2


def test_phonk_lead_sits_in_the_middle_register_not_high_like_a_music_box():
    medians = []
    for seed in range(1, 31):
        song = _phonk(seed)
        drop = _drop_bars(song)
        pitches = [
            note.pitch
            for note in _notes(song, "Melody")
            if _bar_of(note) in drop
        ]
        medians.append(median(pitches))
        assert 50 <= medians[-1] <= 76, seed
    central = sum(57 <= found <= 74 for found in medians)
    assert central >= 0.8 * len(medians)
    assert 59 <= mean(medians) <= 69


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_lead_loops_one_unchanged_bar_through_a_third_of_the_drop(seed):
    song = _phonk(seed)
    drop = _drop_bars(song)
    loops = Counter()
    for bar in drop:
        hits = tuple(
            sorted(
                (round(_step_of(note)), note.pitch)
                for note in _notes(song, "Melody")
                if _bar_of(note) == bar
            )
        )
        if hits:
            loops[hits] += 1
    assert loops.most_common(1)[0][1] / sum(loops.values()) >= 0.35


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_clap_and_snare_fall_only_on_the_third_beat(seed):
    song = _phonk(seed)
    backbeat = _drum_hits(song, CLAP, SNARE)
    assert backbeat
    assert {_step_of(note) for note in backbeat} == {8.0}


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_kick_is_sparse_and_louder_than_the_hats(seed):
    song = _phonk(seed)
    kicks = _drum_hits(song, KICK)
    hats = _drum_hits(song, CLOSED_HAT)
    bars = {_bar_of(note) for note in kicks}
    assert 2 <= len(kicks) / len(bars) <= 4.5
    assert (
        mean(n.velocity for n in kicks) >= mean(n.velocity for n in hats) + 12
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_hats_roll_in_thirty_seconds_and_in_triplets(seed):
    song = _phonk(seed)
    starts = sorted({round(note.start, 5) for note in _drum_hits(song, 42)})
    gaps = [
        round((later - earlier) / SIXTEENTH, 2)
        for earlier, later in zip(starts, starts[1:], strict=False)
    ]
    assert 0.5 in gaps
    assert 0.67 in gaps
    steady = sum(1 for gap in gaps if gap >= 1.0)
    assert steady / len(gaps) >= 0.5


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_bass_is_a_low_sustained_sub_bass(seed):
    song = _phonk(seed)
    bass = _notes(song, "Bass")
    assert bass
    assert {inst.program for inst in _voices(song, "Bass")} == set(
        guitar_amp.DRIVE_FOR_PROGRAM
    ) - {AGOGO_BELL}
    assert min(note.pitch for note in bass) >= 28
    assert median(note.pitch for note in bass) <= 40
    assert max(note.pitch for note in bass) <= 52
    assert median(note.end - note.start for note in bass) >= 0.85 * BEAT


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_bass_notes_start_together_with_kick_hits(seed):
    song = _phonk(seed)
    kicks = {round(note.start, 4) for note in _drum_hits(song, KICK)}
    bass = _notes(song, "Bass")
    together = [note for note in bass if round(note.start, 4) in kicks]
    assert len(together) / len(bass) >= 0.95


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_chords_are_bare_fifths_without_bright_keyboards(seed):
    song = _phonk(seed)
    programs = {i.program for i in song.instruments if not i.is_drum}
    assert not programs & BRIGHT_TOY_PROGRAMS
    stacks = {}
    for note in _notes(song, "Chords"):
        stacks.setdefault(round(note.start, 4), []).append(note.pitch)
    assert stacks
    for pitches in stacks.values():
        low = min(pitches)
        assert {(pitch - low) % 12 for pitch in pitches} <= {0, 7}


def _with_counter(seed, style="Phonk"):
    if style == "Phonk":
        return _phonk(seed, roles=ROLE_KEYS)
    return compose(
        ComposeOptions(
            "A",
            "Minor",
            bpm=BPM,
            minutes=1.5,
            style=style,
            seed=seed,
            roles=ROLE_KEYS,
        )
    )


def _loud_bars(song):
    return {
        bar
        for section in song.sections
        if section.name in CHOP_SECTIONS
        for bar in range(section.start_bar, section.start_bar + section.bars)
    }


def _chops(song):
    loud = _loud_bars(song)
    return [
        note for note in _notes(song, "Counter-melody") if _bar_of(note) in loud
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_phonk_counter_part_is_short_repeated_voice_chops_in_drops_and_breaks(
    seed,
):
    song = _with_counter(seed)
    chops = _chops(song)
    assert chops
    loud = _loud_bars(song)
    programs = {
        inst.program
        for inst in _voices(song, "Counter-melody")
        if any(_bar_of(note) in loud for note in inst.notes)
    }
    assert programs <= set(genre_feel.feel_for("Phonk").chop_programs)
    assert programs <= set(genres.profile_for("Phonk").programs["counter"])
    assert min(note.end - note.start for note in chops) >= 0.85 * SIXTEENTH
    assert min(note.velocity for note in chops) >= 95
    for note in chops:
        section = next(
            s
            for s in song.sections
            if s.start_bar <= _bar_of(note) < s.start_bar + s.bars
        )
        assert (_bar_of(note) - section.start_bar) % 2 == 1
    assert max(note.end - note.start for note in chops) <= SIXTEENTH * 2
    assert all(55 <= note.pitch <= 76 for note in chops)
    lead = {round(note.start, 4) for note in _notes(song, "Melody")}
    assert not {round(note.start, 4) for note in chops} & lead
    figures = Counter()
    for bar in {_bar_of(note) for note in chops}:
        figures[
            tuple(sorted(_step_of(n) for n in chops if _bar_of(n) == bar))
        ] += 1
    assert figures.most_common(1)[0][1] / sum(figures.values()) >= 0.4


def test_phonk_songs_play_riffs_of_their_own():
    riffs = []
    for seed in range(1, 9):
        song = _phonk(seed)
        drop = _drop_bars(song)
        bars = Counter(
            tuple(
                sorted(
                    round(_step_of(n))
                    for n in _notes(song, "Melody")
                    if _bar_of(n) == bar
                )
            )
            for bar in drop
        )
        riffs.append(set(bars.most_common(1)[0][0]))
    assert len({tuple(sorted(riff)) for riff in riffs}) >= 7
    pairs = [
        len(a & b) / len(a | b)
        for i, a in enumerate(riffs)
        for b in riffs[i + 1 :]
    ]
    assert statistics.mean(pairs) <= 0.45


def test_choosing_phonk_also_chooses_its_counter_part():
    from korshmer.midi_generation.composer import STYLE_ROLES
    from korshmer.mixins.pages.generator import GeneratorPageMixin

    class _Box:
        def __init__(self):
            self.checked = False

        def setChecked(self, value):
            self.checked = value

    page = GeneratorPageMixin()
    page.gen_role_boxes = {"counter": _Box(), "melody": _Box()}
    page._add_style_roles("Pop")
    assert not page.gen_role_boxes["counter"].checked
    page._add_style_roles("Phonk")
    assert page.gen_role_boxes["counter"].checked
    assert not page.gen_role_boxes["melody"].checked
    assert STYLE_ROLES["Phonk"] == ("counter",)


def test_voice_chops_need_the_counter_part_and_belong_to_phonk():
    assert not _voices(_phonk(1), "Counter-melody")
    assert genre_feel.feel_for("Phonk").chops
    for style in SONG_STYLES:
        if style != "Phonk":
            assert not genre_feel.feel_for(style).chops


def test_phonk_band_sounds_change_between_songs_but_stay_dark():
    sounds = {"Chords": set(), "Pad": set()}
    for seed in range(1, 13):
        song = _phonk(seed)
        for word, found in sounds.items():
            found |= {inst.program for inst in _voices(song, word)}
        programs = {i.program for i in song.instruments if not i.is_drum}
        assert not programs & BRIGHT_TOY_PROGRAMS, seed
    assert len(sounds["Chords"]) >= 3
    assert len(sounds["Pad"]) >= 3


def test_phonk_in_a_major_key_is_centred_on_the_relative_minor():
    shares = []
    centred = 0
    for seed in range(1, 21):
        song = _phonk(seed, root="C", scale="Major")
        for section in song.sections:
            assert section.key.split(" ", 1)[1] in ("Minor", "Phrygian")
        assert song.sections[0].key == "A Minor"
        downbeats = Counter(
            note.pitch % 12
            for note in _notes(song, "Bass")
            if abs(note.start - _bar_of(note) * BAR) < 1e-6
        )
        centred += downbeats.most_common(1)[0][0] == 9
        shares.append(downbeats[9] / sum(downbeats.values()))
    assert centred >= 0.9 * len(shares)
    assert mean(shares) >= 0.4
    c_major = {0, 2, 4, 5, 7, 9, 11}
    assert {n.pitch % 12 for n in _notes(song, "Melody")} <= c_major
    assert _phonk(seed, root="D", scale="Minor").sections[0].key == "D Minor"


def _harmonics_over_fundamental_db(signal, fundamental):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
    near = (freqs > fundamental * 0.8) & (freqs < fundamental * 1.2)
    above = freqs > fundamental * 1.5
    return 10 * np.log10(spectrum[above].sum() / spectrum[near].sum())


def test_sub_bass_drive_adds_grit_to_a_low_note_and_keeps_its_fundamental():
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    clean = 0.3 * np.sin(2 * np.pi * 55.0 * t) * np.exp(-t)
    driven = guitar_amp.drive(clean, SAMPLE_RATE, guitar_amp.SUB_DRIVE)
    assert driven.shape == (len(clean), 2)
    assert driven.dtype == np.float32
    assert _harmonics_over_fundamental_db(clean, 55.0) < -40
    grit = _harmonics_over_fundamental_db(driven[:, 0].astype(float), 55.0)
    assert -12 < grit < 0
    spectrum = np.abs(np.fft.rfft(driven[:, 0]))
    freqs = np.fft.rfftfreq(len(clean), 1 / SAMPLE_RATE)
    assert abs(freqs[spectrum.argmax()] - 55.0) < 2


def test_sub_bass_drive_leaves_notes_above_the_bass_register_untouched():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    lead = 0.3 * np.sin(2 * np.pi * 440.0 * t) * np.hanning(SAMPLE_RATE)
    driven = guitar_amp.drive(lead, SAMPLE_RATE, guitar_amp.SUB_DRIVE)
    change = driven[:, 0] - lead
    assert np.sqrt(np.mean(change**2)) < 0.01 * np.sqrt(np.mean(lead**2))


def test_bell_drive_makes_a_decaying_hit_denser_without_changing_its_level():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    hit = 0.4 * np.sin(2 * np.pi * 800.0 * t) * np.exp(-t * 25)
    driven = guitar_amp.drive(hit, SAMPLE_RATE, guitar_amp.BELL_DRIVE)[:, 0]

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

    def crest(x):
        return float(np.abs(x).max()) / rms(x)

    assert rms(driven) == pytest.approx(rms(hit), rel=0.05)
    assert crest(driven) < 0.8 * crest(hit)


def test_drive_handles_silence_and_empty_input():
    silent = guitar_amp.drive(np.zeros(1000), SAMPLE_RATE, guitar_amp.SUB_DRIVE)
    assert silent.shape == (1000, 2)
    assert not silent.any()
    empty = guitar_amp.drive(np.zeros(0), SAMPLE_RATE, guitar_amp.BELL_DRIVE)
    assert empty.shape == (0, 2)


def _mix_through_fake_synth(monkeypatch, tracks):
    rendered = []

    def fake_batch(instruments, level=True):
        rendered.append(sorted({i.program for i in instruments}))
        return np.full((SAMPLE_RATE, 2), 0.1, np.float32)

    driven = []
    monkeypatch.setattr(
        guitar_amp,
        "drive",
        lambda audio, sr, settings: driven.append(settings) or audio,
    )
    mix = audio_processors._render_with_guitar_amps(
        tracks, fake_batch, SAMPLE_RATE
    )
    assert mix.shape == (SAMPLE_RATE, 2)
    return rendered, driven


def test_render_drives_the_phonk_bass_and_bell_and_nothing_else(monkeypatch):
    song = _phonk(1)
    piano = pretty_midi.Instrument(0)
    piano.notes.append(pretty_midi.Note(90, 60, 0, 1))
    tracks = [piano, *_voices(song, "Bass"), *_voices(song, "Melody")]
    rendered, driven = _mix_through_fake_synth(monkeypatch, tracks)
    assert rendered[0] == [0]
    assert len(rendered) == 3
    assert set(driven) == {guitar_amp.SUB_DRIVE, guitar_amp.BELL_DRIVE}


def test_render_does_not_drive_a_high_lead_that_uses_the_bass_sound(
    monkeypatch,
):
    bass_program = _voices(_phonk(1), "Bass")[0].program
    lead = pretty_midi.Instrument(bass_program)
    lead.notes.append(pretty_midi.Note(90, 76, 0, 1))
    rendered, driven = _mix_through_fake_synth(monkeypatch, [lead])
    assert rendered == [[bass_program]]
    assert driven == []


class _CountingSynth:
    def __init__(self, *args, **kwargs):
        self.position = 0
        self.note_positions = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sf_load(self, path):
        return 1

    def set_reverb(self, *args):
        pass

    def set_chorus(self, *args):
        pass

    def cc(self, *args):
        pass

    def program_select(self, *args):
        pass

    def note_on(self, channel, pitch, velocity):
        self.note_positions.append((pitch, self.position))

    def note_off(self, channel, pitch):
        pass

    def get_samples(self, count):
        self.position += count
        return np.zeros(count), np.zeros(count)


def test_a_late_note_is_rendered_on_time_after_thousands_of_earlier_notes(
    tmp_path, monkeypatch
):
    synths = []

    def make(*args, **kwargs):
        synths.append(_CountingSynth())
        return synths[-1]

    monkeypatch.setattr(audio_processors, "FluidSynth", make)
    monkeypatch.setattr(
        audio_processors, "save_normalized_audio", lambda *a, **k: None
    )
    monkeypatch.setattr(
        audio_processors, "read_soundfont_kits", lambda p: set()
    )
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    hats = pretty_midi.Instrument(0, is_drum=True)
    for index in range(4000):
        start = index * SIXTEENTH / 3
        hats.notes.append(
            pretty_midi.Note(90, CLOSED_HAT, start, start + SIXTEENTH / 7)
        )
    late = 4000 * SIXTEENTH / 3
    hats.notes.append(pretty_midi.Note(110, KICK, late, late + 0.1))
    midi.instruments.append(hats)
    path = tmp_path / "long.mid"
    midi.write(str(path))
    audio_processors.render_midi_to_wav(
        str(path), str(tmp_path / "long.wav"), "x", "y", bpm=BPM
    )
    kick_at = next(
        at for pitch, at in synths[0].note_positions if pitch == KICK
    )
    assert abs(kick_at - late * SAMPLE_RATE) <= 0.002 * SAMPLE_RATE


def test_phonk_songs_come_in_several_flavours_of_one_style():
    found = Counter(_flavor_of(seed) for seed in range(1, 41))
    assert set(found) == set(genres.FLAVORS["Phonk"])
    assert found["Phonk"] >= 12
    for flavor in found:
        assert genres.primary(flavor) == "Phonk"
        assert flavor in genre_feel.FEELS and flavor in genres.GENRES
    assert _phonk(1, flavor="Phonk/House").style == "Phonk/House"


def test_memphis_phonk_has_a_dark_held_lead_and_a_cowbell_in_the_drums():
    lengths = []
    for seed in SEEDS:
        song = _phonk(seed, flavor="Phonk/Memphis")
        drop = _drop_bars(song)
        lead = [n for n in _notes(song, "Melody") if _bar_of(n) in drop]
        assert {i.program for i in _voices(song, "Melody")} <= {0, 19, 14}
        assert _drum_hits(song, genre_feel.COWBELL)
        lengths.append(median(n.end - n.start for n in lead))
        programs = {i.program for i in song.instruments if not i.is_drum}
        assert not programs & BRIGHT_TOY_PROGRAMS
    drift = [
        median(
            n.end - n.start
            for n in _notes(_phonk(seed), "Melody")
            if _bar_of(n) in _drop_bars(_phonk(seed))
        )
        for seed in SEEDS[:3]
    ]
    assert mean(lengths) >= 1.5 * mean(drift)


def test_house_phonk_kicks_every_beat_and_claps_on_two_and_four():
    for seed in SEEDS:
        song = _phonk(seed, flavor="Phonk/House")
        drop = _drop_bars(song)
        kicks = Counter(
            round(_step_of(n))
            for n in _drum_hits(song, KICK)
            if _bar_of(n) in drop
        )
        bars = len(
            {_bar_of(n) for n in _drum_hits(song, KICK) if _bar_of(n) in drop}
        )
        assert all(kicks[step] >= 0.8 * bars for step in (0, 4, 8, 12))
        backbeat = {
            round(_step_of(n))
            for n in _drum_hits(song, CLAP, SNARE)
            if _bar_of(n) in drop
        }
        assert {4, 12} <= backbeat
        assert {i.program for i in _voices(song, "Melody")} == {AGOGO_BELL}
        assert {i.program for i in _voices(song, "Bass")} == {87}


def test_drift_phonk_songs_do_not_share_one_beat():
    beats = set()
    for seed in range(1, 13):
        song = _phonk(seed)
        drop = _drop_bars(song)
        bar = min(drop)
        beats.add(
            tuple(
                sorted(
                    (n.pitch, round(_step_of(n)))
                    for n in _drum_hits(song, KICK, 42, 44, 46)
                    if _bar_of(n) == bar
                )
            )
        )
    assert len(beats) >= 9


def test_an_auto_song_led_by_phonk_brings_its_voice_chops():
    song = compose(ComposeOptions("A", "Minor", bpm=148.0, minutes=1.0, seed=3))
    assert genres.primary(song.style) == "Phonk"
    assert _voices(song, "Counter-melody")
    chosen = compose(
        ComposeOptions(
            "A",
            "Minor",
            bpm=148.0,
            minutes=1.0,
            seed=3,
            roles=("drums", "bass", "melody"),
        )
    )
    assert not _voices(chosen, "Counter-melody")
    pop = compose(ComposeOptions("A", "Major", bpm=110.0, minutes=1.0, seed=3))
    assert not _voices(pop, "Counter-melody")
