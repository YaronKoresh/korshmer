import os
import shutil

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer.midi.adaptive_snap import (
    DIVISIONS,
    allowed_divisions,
    choose_division,
    finest_for,
    snap_adaptive,
    supports,
)
from tests.fixtures.drum_loops import SAMPLE_RATE, make_drum_loop

TEMPO = 120.0
BEAT = 60.0 / TEMPO


def _midi(*notes, drum=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=0, is_drum=drum)
    for pitch, beat_start, beat_length in notes:
        inst.notes.append(
            pretty_midi.Note(
                80, pitch, beat_start * BEAT, (beat_start + beat_length) * BEAT
            )
        )
    pm.instruments.append(inst)
    return pm


def _starts(pm):
    return [round(n.start / BEAT, 4) for n in pm.instruments[0].notes]


def test_long_note_close_to_the_beat_takes_the_quarter_step():
    pm = _midi((60, 1.03, 1.5))
    counts = snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [1.0]
    assert counts["1/4"] == 1


def test_long_note_between_beats_falls_to_the_eighth_or_sixteenth():
    pm = _midi((60, 1.52, 1.5), (62, 4.26, 1.5))
    counts = snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [1.5, 4.25]
    assert counts["1/8"] == 1 and counts["1/16"] == 1


def test_medium_note_never_takes_the_quarter_step():
    pm = _midi((60, 2.02, 0.5))
    counts = snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [2.0]
    assert counts == {"1/4": 0, "1/8": 1, "1/16": 0}


def test_short_notes_always_use_the_finest_step():
    pm = _midi((60, 3.03, 0.25), (62, 3.3, 0.25))
    counts = snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [3.0, 3.25]
    assert counts["1/16"] == 2 and counts["1/4"] == counts["1/8"] == 0


def test_short_note_keeps_its_length_not_stretched_to_a_full_step():
    pm = _midi((60, 2.0, 0.2))
    snap_adaptive(pm, TEMPO, 0.0, finest=2)
    note = pm.instruments[0].notes[0]
    step = BEAT / 2
    assert note.end - note.start < step
    assert note.end - note.start == pytest.approx(0.2 * BEAT, abs=1e-6)


def test_one_pass_mixes_all_three_steps():
    pm = _midi(
        (60, 0.02, 2.0),
        (62, 2.51, 0.6),
        (64, 3.27, 0.2),
        (65, 4.97, 1.2),
    )
    counts = snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [0.0, 2.5, 3.25, 5.0]
    assert counts["1/4"] == 2 and counts["1/8"] == 1 and counts["1/16"] == 1


def test_a_far_syncopation_is_not_pulled_onto_the_beat():
    pm = _midi((60, 1.2, 1.5))
    snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [1.25]


def test_tolerance_is_capped_in_seconds_at_slow_tempos():
    def division_for(tempo, beats_off):
        beat = 60.0 / tempo
        note = pretty_midi.Note(
            80, 60, (1 + beats_off) * beat, (1 + beats_off) * beat + 2 * beat
        )
        return choose_division(note, False, tempo, 0.0, (1, 2, 4))[0]

    assert division_for(120.0, 0.07) == 1
    assert division_for(60.0, 0.07) == 4


def test_drums_follow_their_own_rules():
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    kit = pretty_midi.Instrument(program=0, is_drum=True)
    for pitch, beat in ((36, 0.03), (42, 1.03), (38, 2.27), (42, 3.27)):
        kit.notes.append(
            pretty_midi.Note(90, pitch, beat * BEAT, beat * BEAT + 0.08)
        )
    pm.instruments.append(kit)
    counts = snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == [0.0, 1.0, 2.25, 3.25]
    assert counts["1/4"] == 1
    assert counts["1/8"] == 1
    assert counts["1/16"] == 2


def test_finest_step_limits_the_choices():
    notes = [(60, 3.03, 0.25), (62, 1.03, 1.5)]
    for finest, expected in ((1, {"1/4": 2}), (2, {"1/4": 1, "1/8": 1})):
        pm = _midi(*notes)
        counts = snap_adaptive(pm, TEMPO, 0.0, finest=finest)
        assert {k: v for k, v in counts.items() if v} == expected
        assert set(counts) <= {"1/4", "1/8"}


def test_thirty_second_step_is_used_only_when_asked_for():
    notes = [(60, 3.13, 0.12)]
    pm = _midi(*notes)
    snap_adaptive(pm, TEMPO, 0.0, finest=4)
    assert _starts(pm) == [3.25]
    pm = _midi(*notes)
    counts = snap_adaptive(pm, TEMPO, 0.0, finest=8)
    assert _starts(pm) == [3.125]
    assert counts["1/32"] == 1


def test_strength_blends_between_original_and_grid():
    pm = _midi((60, 1.04, 1.5))
    snap_adaptive(pm, TEMPO, 0.0, strength=0.0)
    assert _starts(pm) == [1.04]
    snap_adaptive(pm, TEMPO, 0.0, strength=0.5)
    assert _starts(pm) == [1.02]


def test_note_length_is_kept_and_nothing_starts_before_zero():
    pm = _midi((60, 0.01, 1.5), (62, 4.0, 1.0))
    before = [n.end - n.start for n in pm.instruments[0].notes]
    snap_adaptive(pm, TEMPO, 0.0)
    after = [n.end - n.start for n in pm.instruments[0].notes]
    assert after == pytest.approx(before)
    assert all(n.start >= 0 for n in pm.instruments[0].notes)


def test_swing_moves_notes_on_odd_steps_only():
    pm = _midi((60, 0.5, 0.5), (62, 1.0, 0.5))
    snap_adaptive(pm, TEMPO, 0.0, swing=1.0)
    first, second = _starts(pm)
    assert first > 0.5
    assert second == 1.0


def test_decimal_tempo_and_offset_are_supported():
    tempo, offset = 123.4, 0.137
    beat = 60.0 / tempo
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(80, 60, offset + 2.02 * beat, offset + 4 * beat)
    )
    pm.instruments.append(inst)
    snap_adaptive(pm, tempo, offset)
    assert inst.notes[0].start == pytest.approx(offset + 2 * beat)


def test_same_pitch_overlaps_are_trimmed_after_snapping():
    pm = _midi((60, 1.02, 2.0), (60, 1.98, 1.0))
    snap_adaptive(pm, TEMPO, 0.0)
    first, second = pm.instruments[0].notes
    assert first.end <= second.start


def test_snapping_twice_changes_nothing_more():
    pm = _midi((60, 0.02, 2.0), (62, 2.51, 0.6), (64, 3.27, 0.2))
    snap_adaptive(pm, TEMPO, 0.0)
    once = _starts(pm)
    snap_adaptive(pm, TEMPO, 0.0)
    assert _starts(pm) == once


@pytest.mark.parametrize("tempo", (0, -5, None))
def test_bad_tempo_leaves_the_notes_alone(tempo):
    pm = _midi((60, 1.03, 1.5))
    assert snap_adaptive(pm, tempo, 0.0)["1/4"] == 0
    assert _starts(pm) == [1.03]


def test_helpers():
    assert [supports(d) for d in (1, 2, 4, 8, 16, 3, 6)] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert finest_for(16) == DIVISIONS[-1]
    assert finest_for(2) == 2
    assert allowed_divisions(4) == (1, 2, 4)
    assert allowed_divisions(0) is None


def test_process_track_snaps_jittered_drums_to_the_grid(tmp_path, monkeypatch):
    import korshmer.track_processor as track_processor

    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (0, "Major")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )

    def fake_trim(path, job_dir):
        target = os.path.join(job_dir, "temp_trimmed.wav")
        shutil.copy2(path, target)
        return target

    audio, _ = make_drum_loop(120.0, bars=8)
    shift = int(0.012 * SAMPLE_RATE)
    audio = np.concatenate([np.zeros(shift, dtype=np.float32), audio])

    def fake_run_demucs(_audio, output_dir):
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        sf.write(os.path.join(target, "drums.wav"), audio, SAMPLE_RATE)

    monkeypatch.setattr(track_processor, "trim_audio_silence", fake_trim)
    monkeypatch.setattr(track_processor, "run_demucs", fake_run_demucs)
    song = tmp_path / "Tune.wav"
    sf.write(song, audio, SAMPLE_RATE)
    progress = []
    generated = track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        selected_stems=["drums"],
        progress_callback=lambda v, desc="": progress.append(desc),
    )
    midis = [p for p in generated if p.endswith(".mid")]
    assert midis
    assert any("Snapped notes to the grid" in text for text in progress)
    for path in midis:
        pm = pretty_midi.PrettyMIDI(path)
        tempo = pm.get_tempo_changes()[1][0]
        step = 60.0 / tempo / 4
        starts = [n.start for i in pm.instruments for n in i.notes]
        assert starts
        phases = np.array(starts) / step
        on_grid = np.abs(phases - np.round(phases)) < 0.02
        offsets = np.array(starts) - starts[0]
        assert (
            on_grid.mean() > 0.9
            or (np.abs(offsets / step - np.round(offsets / step)) < 0.02).mean()
            > 0.9
        )


def _backbeat_song(first_beat):
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    bass = pretty_midi.Instrument(program=33)
    for i in range(16):
        start = first_beat + i * BEAT
        drums.notes.append(pretty_midi.Note(100, 36, start, start + 0.1))
        drums.notes.append(
            pretty_midi.Note(80, 42, start + BEAT / 2, start + BEAT / 2 + 0.05)
        )
        bass.notes.append(pretty_midi.Note(90, 40, start, start + BEAT))
    pm.instruments.extend([drums, bass])
    return pm


def test_fitted_offset_on_a_sixteenth_does_not_swing_the_beats():
    from korshmer.midi_algorithms import fit_beat_grid

    first_beat = 0.3
    pm = _backbeat_song(first_beat)
    onsets = [n.start for i in pm.instruments for n in i.notes]
    tempo, offset, _ = fit_beat_grid(onsets, TEMPO, grid_divisions=4)
    counts = snap_adaptive(pm, tempo, offset, finest=4, swing=1.0, phase_div=4)
    assert counts["1/4"] > 0
    for inst in pm.instruments:
        for note in inst.notes:
            if note.pitch in (36, 40):
                position = (note.start - first_beat) / BEAT
                assert position == pytest.approx(round(position), abs=1e-6)


def test_beat_offset_search_keeps_a_correct_offset():
    from korshmer.midi.adaptive_snap import find_beat_offset

    pm = _backbeat_song(0.0)
    assert find_beat_offset(pm, TEMPO, 0.0, 4) == 0.0
    assert find_beat_offset(pm, TEMPO, 0.0, 1) == 0.0
    assert find_beat_offset(_midi(), TEMPO, 0.05, 4) == 0.05


NON_FINITE = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.parametrize("tempo", NON_FINITE[:2])
def test_beat_grid_fit_rejects_a_non_finite_tempo_hint(tempo):
    from korshmer.midi_algorithms import fit_beat_grid

    onsets = np.arange(0, 20, 0.25)
    assert fit_beat_grid(onsets, tempo) is None


@pytest.mark.parametrize("divisions", (0, -4, *NON_FINITE))
def test_beat_grid_fit_rejects_a_bad_grid(divisions):
    from korshmer.midi_algorithms import fit_beat_grid

    onsets = np.arange(0, 20, 0.25)
    assert fit_beat_grid(onsets, 120, grid_divisions=divisions) is None


@pytest.mark.parametrize(
    "kwargs",
    (
        {"tempo": float("nan")},
        {"tempo": float("inf")},
        {"tempo": 0},
        {"offset": float("nan")},
        {"grid_divisions": 0},
        {"grid_divisions": float("nan")},
        {"strength": float("nan")},
        {"strength": 2},
        {"swing": float("inf")},
    ),
)
def test_apply_beat_grid_rejects_bad_numbers(kwargs):
    from korshmer.midi_algorithms import apply_beat_grid

    arguments = {
        "tempo": TEMPO,
        "offset": 0.0,
        "grid_divisions": 4,
        "strength": 1.0,
        "swing": 0.0,
    }
    arguments.update(kwargs)
    pm = _midi((60, 1.03, 1.5))
    with pytest.raises(ValueError):
        apply_beat_grid(pm, **arguments)
    assert _starts(pm) == [1.03]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"tempo": float("nan")},
        {"tempo": float("inf")},
        {"offset": float("nan")},
        {"strength": float("nan")},
        {"strength": -1},
        {"swing": float("nan")},
    ),
)
def test_snap_adaptive_rejects_bad_numbers_and_leaves_the_notes(kwargs):
    arguments = {"tempo": TEMPO, "offset": 0.0, "strength": 1.0, "swing": 0.0}
    arguments.update(kwargs)
    pm = _midi((60, 1.03, 1.5))
    with pytest.raises(ValueError):
        snap_adaptive(pm, **arguments)
    assert _starts(pm) == [1.03]


@pytest.mark.parametrize("drum,pitch", ((True, 42), (False, 60)))
def test_two_notes_snapped_to_one_start_become_one_note(drum, pitch, tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=0, is_drum=drum)
    inst.notes.append(pretty_midi.Note(60, pitch, 0.49, 0.55))
    inst.notes.append(pretty_midi.Note(110, pitch, 0.52, 0.58))
    inst.notes.append(pretty_midi.Note(80, pitch, 1.0, 1.2))
    pm.instruments.append(inst)
    snap_adaptive(pm, TEMPO, 0.0, finest=4)
    path = tmp_path / "snapped.mid"
    pm.write(str(path))
    notes = sorted(
        pretty_midi.PrettyMIDI(str(path)).instruments[0].notes,
        key=lambda n: n.start,
    )
    assert len(notes) == 2
    assert notes[0].start == pytest.approx(0.5, abs=1e-3)
    assert notes[0].end == pytest.approx(0.56, abs=3e-3)
    assert notes[0].velocity == 110
