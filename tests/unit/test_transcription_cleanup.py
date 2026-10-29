import inspect

import numpy as np
import pretty_midi
import pytest

from korshmer import track_processor
from korshmer.midi.transcription_cleanup import (
    CleanupOptions,
    cap_polyphony,
    clamp_pitch_range,
    clean_transcription,
    close_note_gaps,
    correct_octave_jumps,
    drop_quiet_notes,
    estimate_key,
    estimate_key_confidence,
    grid_confidence,
    merge_fragments,
    merge_note_variants,
    options_for_cleanup,
    quantize_light,
    remove_ghost_notes,
    remove_neighbour_notes,
    remove_sub_octave_notes,
    smooth_note_velocity,
)
from tests.fixtures.transcription_fixtures import (
    SONGS,
    load_truth,
    note_scores,
    to_pretty_midi,
)

MOCK_OPTIONS = CleanupOptions(
    remove_ghosts=True,
    merge_reattacks=True,
    reattack_ratio=None,
    remove_sub_octaves=False,
    quantize_division=0,
)


def mock_model_output(truth, tonic, mode, seed):
    rng = np.random.default_rng(seed)
    scale = {
        (tonic + s) % 12
        for s in (
            (0, 2, 4, 5, 7, 9, 11)
            if mode == "major"
            else (0, 2, 3, 5, 7, 8, 10)
        )
    }
    notes = []
    for pitch, start, end in truth:
        if rng.random() < 0.08:
            continue
        s = start + rng.uniform(-0.03, 0.03)
        e = max(s + 0.1, end + rng.uniform(-0.05, 0.05))
        velocity = int(rng.integers(75, 110))
        if rng.random() < 0.12:
            wrong = pitch + int(rng.choice([-1, 1]))
            if wrong % 12 not in scale:
                notes.append((wrong, s, e, int(velocity * 0.55)))
                continue
        if rng.random() < 0.12 and e - s > 0.3:
            mid = s + (e - s) * rng.uniform(0.3, 0.7)
            notes.append((pitch, s, mid, velocity))
            notes.append((pitch, mid + 0.01, e, velocity))
        else:
            notes.append((pitch, s, e, velocity))
        if rng.random() < 0.35 and pitch + 12 < 108:
            notes.append((pitch + 12, s, e, int(velocity * 0.45)))
        if rng.random() < 0.12 and pitch + 19 < 108:
            notes.append((pitch + 19, s, e, int(velocity * 0.4)))
    length = max(e for _, _, e in truth)
    for _ in range(max(3, len(truth) // 8)):
        s = rng.uniform(0, length - 0.2)
        pitch = int(rng.integers(40, 84))
        notes.append((pitch, s, s + rng.uniform(0.02, 0.05), 40))
    return to_pretty_midi(notes)


def _flat(midi):
    return [
        (n.pitch, n.start, n.end) for i in midi.instruments for n in i.notes
    ]


def test_estimate_key_finds_scale_of_major_and_minor_fixtures():
    steps = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
    checked = 0
    for name, tonic, scale, *_ in SONGS:
        if scale not in steps:
            continue
        notes = to_pretty_midi(load_truth(name)).instruments[0].notes
        root, mode = estimate_key(notes)
        found = {(root + s) % 12 for s in steps[mode.lower()]}
        expected = {(tonic + s) % 12 for s in steps[scale]}
        assert found == expected, name
        checked += 1
    assert checked == 6


def test_ghost_octave_removed_but_loud_octave_kept():
    midi = to_pretty_midi(
        [
            (40, 0.0, 1.0, 100),
            (52, 0.0, 1.0, 40),
            (40, 2.0, 3.0, 100),
            (52, 2.0, 3.0, 95),
        ]
    )
    assert remove_ghost_notes(midi) == 1
    pitches = sorted(n.pitch for n in midi.instruments[0].notes)
    assert pitches == [40, 40, 52]


def test_weak_out_of_key_note_snaps_and_strong_long_note_stays():
    notes = [(60, i * 0.5, i * 0.5 + 0.4, 90) for i in range(8)]
    notes += [(64, 4.0 + i * 0.5, 4.4 + i * 0.5, 90) for i in range(8)]
    notes += [(67, 8.0 + i * 0.5, 8.4 + i * 0.5, 90) for i in range(8)]
    notes.append((61, 12.0, 12.2, 40))
    notes.append((66, 13.0, 14.0, 100))
    cleaned, report = clean_transcription(
        to_pretty_midi(notes),
        key=(0, "Major"),
        options=CleanupOptions(snap_to_key=True),
    )
    pitches = [n.pitch for n in cleaned.instruments[0].notes]
    assert 61 not in pitches
    assert 66 in pitches
    assert report["notes_snapped"] == 1


def test_merge_fragments_and_blips():
    midi = to_pretty_midi(
        [(60, 0.0, 0.5, 80), (60, 0.51, 1.0, 70), (62, 2.0, 2.02, 80)]
    )
    removed = merge_fragments(midi.instruments[0], 0.04, 0.06)
    notes = midi.instruments[0].notes
    assert removed == 2
    assert len(notes) == 1 and notes[0].end == 1.0


def test_disabled_returns_unchanged_copy():
    midi = to_pretty_midi([(60, 0.0, 1.0, 80), (72, 0.0, 1.0, 20)])
    out, report = clean_transcription(
        midi, options=CleanupOptions(enabled=False)
    )
    assert _flat(out) == _flat(midi) and out is not midi
    assert report == {}


def test_drums_are_untouched():
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes = [
        pretty_midi.Note(80, 36, 0.0, 0.02),
        pretty_midi.Note(80, 38, 0.5, 0.52),
    ]
    midi.instruments.append(drums)
    out, _ = clean_transcription(midi)
    assert len(out.instruments[0].notes) == 2


def test_track_processor_accepts_cleanup_options():
    params = inspect.signature(track_processor.process_track).parameters
    assert params["cleanup_options"].default is None


def test_steps_raise_mean_f1_on_mocked_songs():
    before, after = [], []
    for index, (name, tonic, scale, *_) in enumerate(SONGS):
        truth = load_truth(name)
        noisy = mock_model_output(truth, tonic, scale, seed=1000 + index)
        cleaned, report = clean_transcription(noisy, options=MOCK_OPTIONS)
        f_before = note_scores(truth, _flat(noisy))[2]
        f_after = note_scores(truth, _flat(cleaned))[2]
        print(name, round(f_before, 3), "->", round(f_after, 3), report)
        before.append(f_before)
        after.append(f_after)
    print(
        "mean F1",
        round(float(np.mean(before)), 3),
        "->",
        round(float(np.mean(after)), 3),
    )
    assert np.mean(after) - np.mean(before) > 0.05


def test_merge_note_variants_keeps_agreed_notes_with_first_timing():
    first = to_pretty_midi(
        [(60, 0.0, 0.5, 90), (64, 0.5, 1.0, 80), (67, 1.0, 1.5, 70)]
    )
    second = to_pretty_midi([(60, 0.02, 0.4, 50), (67, 1.3, 1.6, 50)])
    merged = merge_note_variants(
        [first, second], tolerance=0.06, keep_seconds=None
    )
    notes = [
        (n.pitch, n.start, n.end, n.velocity)
        for n in merged.instruments[0].notes
    ]
    assert notes == [(60, 0.0, 0.5, 90)]
    union = merge_note_variants([first, second], need=1)
    assert len(union.instruments[0].notes) == 3
    assert len(first.instruments[0].notes) == 3


def test_merge_note_variants_single_input_is_a_copy():
    first = to_pretty_midi([(60, 0.0, 0.5, 90)])
    merged = merge_note_variants([first], need=1)
    assert merged is not first
    assert len(merged.instruments[0].notes) == 1


def test_reattack_with_lower_velocity_joins_but_louder_one_stays():
    midi = to_pretty_midi(
        [
            (60, 0.0, 0.5, 90),
            (60, 0.5, 1.0, 60),
            (62, 2.0, 2.5, 80),
            (62, 2.5, 3.0, 95),
        ]
    )
    merge_fragments(midi.instruments[0], 0.06, 0.06, 0.9)
    notes = [(n.pitch, n.start, n.end) for n in midi.instruments[0].notes]
    assert notes == [(60, 0.0, 1.0), (62, 2.0, 2.5), (62, 2.5, 3.0)]


def test_overlapping_same_pitch_notes_always_join():
    midi = to_pretty_midi([(60, 0.0, 0.6, 90), (60, 0.4, 1.0, 95)])
    merge_fragments(midi.instruments[0], 0.0, 0.0, 0.5)
    assert len(midi.instruments[0].notes) == 1


def test_sub_octave_note_dropped_only_when_weaker():
    midi = to_pretty_midi(
        [
            (72, 0.0, 1.0, 100),
            (60, 0.0, 1.0, 50),
            (72, 2.0, 3.0, 100),
            (60, 2.0, 3.0, 99),
        ]
    )
    assert remove_sub_octave_notes(midi, 0.9) == 1
    assert sorted(n.pitch for n in midi.instruments[0].notes) == [60, 72, 72]


def test_neighbour_semitone_note_dropped_when_much_weaker():
    midi = to_pretty_midi([(60, 0, 1, 100), (61, 0, 1, 40), (62, 2, 3, 40)])
    assert remove_neighbour_notes(midi, 0.6) == 1
    assert sorted(n.pitch for n in midi.instruments[0].notes) == [60, 62]


def test_drop_quiet_notes_uses_stem_median():
    midi = to_pretty_midi([(60, i, i + 0.5, 100) for i in range(5)])
    midi.instruments[0].notes.append(pretty_midi.Note(20, 70, 6, 6.5))
    assert drop_quiet_notes(midi, 0.5) == 1
    assert drop_quiet_notes(midi, 0.0) == 0


def test_weak_snap_is_skipped_when_key_is_unclear():
    chromatic = [(60 + (i % 12), i * 0.3, i * 0.3 + 0.2, 40) for i in range(48)]
    midi = to_pretty_midi(chromatic)
    _, _, confidence = estimate_key_confidence(midi.instruments[0].notes)
    assert confidence < 0.8
    _, report = clean_transcription(
        midi, options=CleanupOptions(snap_to_key=True)
    )
    assert "notes_snapped" not in report
    assert report["key_confidence"] == confidence


def test_octave_jump_in_a_single_line_is_corrected():
    notes = [(60 + (i % 3), i * 0.5, i * 0.5 + 0.4, 80) for i in range(8)]
    notes[4] = (74, 2.0, 2.4, 80)
    midi = to_pretty_midi(notes)
    assert correct_octave_jumps(midi.instruments[0], 5) == 1
    pitches = [n.pitch for n in midi.instruments[0].notes]
    assert pitches[4] == 62 and max(pitches) <= 62
    steady = to_pretty_midi([(60, i, i + 0.5, 80) for i in range(6)])
    assert correct_octave_jumps(steady.instruments[0]) == 0


def test_pitch_range_drops_notes_outside():
    midi = to_pretty_midi([(20, 0, 1, 80), (40, 0, 1, 80), (90, 0, 1, 80)])
    assert clamp_pitch_range(midi.instruments[0], 24, 72) == 2
    low = options_for_cleanup("bass.wav")
    assert low.pitch_range == (24, 72) and not low.remove_sub_octaves
    assert options_for_cleanup("other.wav").pitch_range == ()
    assert options_for_cleanup("vocals.wav").pitch_range == (36, 96)


def test_polyphony_cap_keeps_the_loudest():
    midi = to_pretty_midi(
        [(60, 0, 2, 50), (64, 0, 2, 90), (67, 0, 2, 80), (72, 0, 2, 70)]
    )
    assert cap_polyphony(midi.instruments[0], 2) == 2
    assert sorted(n.pitch for n in midi.instruments[0].notes) == [64, 67]
    assert cap_polyphony(midi.instruments[0], 0) == 0


def test_close_note_gaps_only_for_tiny_gaps():
    midi = to_pretty_midi(
        [(60, 0, 0.45, 80), (62, 0.5, 1.0, 80), (64, 2, 3, 80)]
    )
    assert close_note_gaps(midi.instruments[0], 0.1) == 1
    notes = midi.instruments[0].notes
    assert notes[0].end == 0.5 and notes[1].end == 1.0


def test_grid_confidence_high_for_grid_low_for_random():
    step = 0.125
    on_grid = [0.01 + step * k for k in (0, 3, 4, 9, 12, 20, 25)]
    phase, strength = grid_confidence(on_grid, step)
    assert strength > 0.99 and abs(phase - 0.01) < 1e-6
    rng = np.random.default_rng(3)
    _, weak = grid_confidence(list(rng.uniform(0, 10, 40)), step)
    assert weak < 0.5


def test_quantize_snaps_clear_grid_and_ignores_free_timing():
    step = 60.0 / 120.0 * 4 / 16
    notes = [
        (60, step * k + 0.012, step * (k + 2) + 0.03, 80)
        for k in (0, 3, 4, 9, 12, 20, 25)
    ]
    midi = to_pretty_midi(notes)
    raw_ends = [n.end for n in midi.instruments[0].notes]
    assert quantize_light(midi, 120.0, 16, 0.06, 0.75) == len(notes)
    starts = [n.start for n in midi.instruments[0].notes]
    offset = starts[0]
    assert all(
        abs((s - offset) / step - round((s - offset) / step)) < 1e-6
        for s in starts
    )
    phase = offset
    ends = [n.end for n in midi.instruments[0].notes]
    assert all(
        e <= raw_end + 1e-9 and raw_end - e <= 0.06
        for e, raw_end in zip(ends, raw_ends, strict=True)
    )
    assert all(
        abs((e - phase) / step - round((e - phase) / step)) < 1e-6 for e in ends
    )
    rng = np.random.default_rng(5)
    free = to_pretty_midi(
        [(60, t, t + 0.2, 80) for t in sorted(rng.uniform(0, 10, 30))]
    )
    assert quantize_light(free, 120.0, 16, 0.06, 0.75) == 0
    assert quantize_light(free, 120.0, 0, 0.06, 0.75) == 0


def test_smooth_velocity_pulls_outlier_toward_neighbours():
    midi = to_pretty_midi(
        [(60, i, i + 0.5, v) for i, v in enumerate([80, 82, 20, 81, 79])]
    )
    smooth_note_velocity(midi.instruments[0], 1.0)
    assert min(n.velocity for n in midi.instruments[0].notes) >= 79
    assert smooth_note_velocity(midi.instruments[0], 0.0) == 0


def test_report_lists_every_step_that_ran():
    midi = to_pretty_midi(
        [(72, 0, 1, 100), (60, 0, 1, 40), (64, 2, 3, 90), (64, 3.0, 3.4, 50)]
    )
    options = CleanupOptions(
        merge_reattacks=True,
        remove_ghosts=True,
        remove_neighbours=True,
        velocity_floor=0.2,
        pitch_range=(30, 100),
        fix_octave_jumps=True,
        snap_to_key=True,
        polyphony_cap=3,
        close_gaps=0.1,
        smooth_velocity=0.5,
    )
    _, report = clean_transcription(midi, options=options)
    for key in (
        "fragments_merged",
        "ghosts_removed",
        "sub_octaves_removed",
        "neighbours_removed",
        "quiet_removed",
        "out_of_range",
        "octave_jumps",
        "key",
        "polyphony_trimmed",
        "gaps_closed",
        "quantized",
    ):
        assert key in report, key


@pytest.mark.parametrize(
    ("root", "mode", "expected"),
    [
        (2, "Lydian", {2, 4, 6, 8, 9, 11, 1}),
        (9, "Dorian", {9, 11, 0, 2, 4, 6, 7}),
        (0, "Major", {0, 2, 4, 5, 7, 9, 11}),
        (9, "Minor", {9, 11, 0, 2, 4, 5, 7}),
        (0, "Pentatonic Minor", {0, 3, 5, 7, 10}),
        (0, "Unknown mode", {0, 2, 3, 5, 7, 8, 10}),
    ],
)
def test_scale_classes_follow_every_scale_name(root, mode, expected):
    from korshmer.midi.transcription_cleanup import _scale_classes

    assert _scale_classes(root, mode) == expected
