import numpy as np
import pretty_midi
import pytest

from korshmer.audio_processing.tempo_estimation import (
    TempoProfile,
    steady_sections,
    tempo_profile,
)
from korshmer.midi.tempo_follow import (
    follow_song_tempo,
    follow_tempo,
    real_offset,
    write_midi,
)
from korshmer.midi.tempo_map import TempoMap
from tests.fixtures.drum_loops import SAMPLE_RATE, make_drum_loop

SLOW, FAST = 100.0, 120.0
TRUE_MAP = TempoMap.from_anchors(
    [(0.0, 0), (24.0, 40), *[(24.0 + 0.55 * k, 40 + k) for k in range(1, 9)]]
    + [(28.4 + 0.5 * k, 48 + k) for k in (1, 200)]
)


def _profile():
    times = np.arange(4.0, 120.0, 2.0)
    bpms = np.interp(times, [23.0, 30.0], [SLOW, FAST])
    return TempoProfile(FAST, times, bpms, steady_sections(times, bpms))


def _sixteenths(end_beat=220):
    return TRUE_MAP.seconds_at(np.arange(0, end_beat, 0.25))


def _midi(onsets):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(90, 60, t, t + 0.1) for t in onsets]
    inst.pitch_bends.append(pretty_midi.PitchBend(10, float(onsets[4])))
    midi.instruments.append(inst)
    return midi


def test_steady_sections_skip_the_stretch_where_the_tempo_moves():
    profile = _profile()
    assert [round(bpm) for _, _, bpm in profile.sections] == [100, 120]
    assert profile.sections[0][1] <= 24.0 <= profile.sections[1][0] + 8.0
    assert profile.changes
    flat = TempoProfile(
        FAST, profile.times, np.full(len(profile.times), FAST), ()
    )
    assert not flat.changes


def test_the_followed_map_puts_every_note_back_on_an_even_grid():
    onsets = _sixteenths()
    found = follow_tempo(_profile(), onsets)
    assert not found.is_uniform
    assert found.bpms[0] == pytest.approx(SLOW, abs=0.2)
    assert found.main_bpm(end=float(onsets[-1])) == pytest.approx(FAST, abs=0.2)
    even = found.warp(onsets, FAST)
    step = 60.0 / FAST / 4
    steady = (onsets < 22.0) | (onsets > 32.0)
    for part in (even[onsets < 22.0], even[onsets > 32.0]):
        off = (part - part[0]) / step
        assert np.abs(off - np.round(off)).max() < 0.08
    assert steady.sum() > 300


def test_too_few_notes_fall_back_to_one_tempo():
    found = follow_tempo(_profile(), [1.0, 2.0, 3.0])
    assert found.is_uniform
    assert found.bpms == (FAST,)


def test_a_steady_song_is_not_followed():
    times = np.arange(4.0, 60.0, 2.0)
    bpms = np.full(len(times), FAST)
    steady = TempoProfile(FAST, times, bpms, steady_sections(times, bpms))
    midi = _midi(np.arange(0, 60, 0.125))
    before = [n.start for n in midi.instruments[0].notes]
    assert follow_song_tempo(steady, [midi], FAST) is None
    assert follow_song_tempo(None, [midi], FAST) is None
    assert [n.start for n in midi.instruments[0].notes] == before


def test_following_warps_the_notes_and_the_written_file_restores_them(tmp_path):
    onsets = _sixteenths()
    midi = _midi(onsets)
    follower = follow_song_tempo(_profile(), [midi], FAST)
    assert follower is not None
    assert "BPM" in follower.summary()
    warped = np.array([n.start for n in midi.instruments[0].notes])
    assert np.diff(warped[onsets > 32.0]) == pytest.approx(0.125, abs=0.01)

    path = tmp_path / "followed.mid"
    write_midi(follower, midi.instruments, str(path), FAST)
    loaded = pretty_midi.PrettyMIDI(str(path))
    starts = np.array(sorted(n.start for n in loaded.instruments[0].notes))
    assert starts == pytest.approx(onsets, abs=0.003)
    written = TempoMap.from_pretty_midi(loaded)
    assert written.main_bpm(end=float(onsets[-1])) == pytest.approx(
        FAST, abs=0.2
    )
    beats = written.beats_at(starts[onsets > 32.0]) * 4
    assert np.abs(beats - beats[0] - np.round(beats - beats[0])).max() < 0.08
    assert loaded.instruments[0].pitch_bends[0].time == pytest.approx(
        onsets[4], abs=0.003
    )
    assert real_offset(follower, 0.0) == pytest.approx(0.0)
    assert real_offset(None, 1.5) == 1.5
    assert real_offset(follower, None) is None


def test_without_a_follower_the_file_has_one_tempo(tmp_path):
    midi = _midi(np.arange(0, 8, 0.5))
    path = tmp_path / "plain.mid"
    write_midi(None, midi.instruments, str(path), 97.6)
    tempos = pretty_midi.PrettyMIDI(str(path)).get_tempo_changes()[1]
    assert list(tempos) == pytest.approx([97.6])


def test_the_profile_of_real_audio_finds_both_tempos():
    intro, _ = make_drum_loop(109.1, bars=16)
    body, _ = make_drum_loop(118.0, bars=40)
    profile = tempo_profile(np.concatenate([intro, body]), SAMPLE_RATE)
    assert profile.main_bpm == pytest.approx(118.0, abs=0.3)
    assert profile.changes
    assert profile.sections[0][2] == pytest.approx(109.1, abs=0.6)
    assert profile.sections[-1][2] == pytest.approx(118.0, abs=0.6)
    steady, _ = make_drum_loop(123.4, bars=24)
    assert not tempo_profile(steady, SAMPLE_RATE).changes


def _grid_error(even, step):
    off = (even - even[0]) / step
    return np.abs(off - np.round(off))


def test_a_jump_in_the_beat_at_a_cut_is_absorbed_by_the_map():
    times = np.arange(4.0, 150.0, 2.0)
    bpms = np.interp(times, [23.0, 30.0], [SLOW, FAST])
    profile = TempoProfile(FAST, times, bpms, steady_sections(times, bpms))
    onsets = _sixteenths(end_beat=280)
    cut = 80.0
    jumped = np.where(onsets >= cut, onsets + 0.07, onsets)
    step = 60.0 / FAST / 4

    blind = follow_tempo(profile, jumped)
    aware = follow_tempo(profile, jumped, breaks=[cut])
    after = jumped > cut + 3.0
    before = (jumped > 34.0) & (jumped < cut - 3.0)
    for found in (blind, aware):
        assert not found.is_uniform
    even = aware.warp(jumped, FAST)
    both = np.concatenate([even[before], even[after]])
    assert _grid_error(both, step).max() < 0.1
    assert aware.main_bpm(end=float(jumped[-1])) == pytest.approx(FAST, abs=0.2)
    assert aware.largest_change(0.0, 140.0, FAST) == pytest.approx(
        1 - SLOW / FAST, abs=0.01
    )


def test_a_section_whose_notes_fit_the_grid_poorly_is_still_followed():
    rng = np.random.default_rng(5)
    onsets = _sixteenths()
    loose = np.where(
        onsets > 32.0, onsets + rng.uniform(-0.05, 0.05, len(onsets)), onsets
    )
    found = follow_tempo(_profile(), np.sort(loose))
    assert not found.is_uniform
    assert found.bpms[0] == pytest.approx(SLOW, abs=0.2)
    assert found.main_bpm(end=float(onsets[-1])) == pytest.approx(FAST, abs=0.5)
