import pretty_midi

from korshmer.midi.transcription_cleanup import (
    CleanupOptions,
    clean_transcription,
    merge_note_runs,
)


def _midi(spec, pitch=60):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    for start, end, velocity in spec:
        inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
    midi.instruments.append(inst)
    return midi


def test_fragments_of_one_held_note_become_one_note():
    spec = [
        (0.0, 0.3, 80),
        (0.38, 0.68, 78),
        (0.74, 1.09, 77),
        (1.18, 1.48, 75),
    ]
    cleaned, report = clean_transcription(_midi(spec))
    notes = cleaned.instruments[0].notes
    assert len(notes) == 1
    assert (notes[0].start, notes[0].end) == (0.0, 1.48)
    assert notes[0].velocity == 80
    assert report["runs_merged"] == 3


def test_two_notes_are_not_a_run():
    spec = [(0.0, 0.3, 80), (0.38, 0.68, 78)]
    cleaned, _ = clean_transcription(
        _midi(spec), options=CleanupOptions(merge_reattacks=False)
    )
    assert len(cleaned.instruments[0].notes) == 2


def test_louder_repeated_presses_stay_separate():
    spec = [(0.0, 0.3, 50), (0.38, 0.68, 90), (0.76, 1.06, 120)]
    inst = _midi(spec).instruments[0]
    assert merge_note_runs(inst, 0.12, 3, 1.2) == 0
    assert len(inst.notes) == 3


def test_notes_far_apart_stay_separate():
    spec = [(0.0, 0.3, 80), (0.6, 0.9, 80), (1.2, 1.5, 80)]
    inst = _midi(spec).instruments[0]
    assert merge_note_runs(inst, 0.12, 3, 1.2) == 0


def test_drums_are_not_merged():
    midi = _midi([(0.0, 0.05, 80), (0.1, 0.15, 80), (0.2, 0.25, 80)], pitch=42)
    midi.instruments[0].is_drum = True
    cleaned, _ = clean_transcription(midi)
    assert len(cleaned.instruments[0].notes) == 3


def test_touching_repeated_notes_stay_separate():
    spec = [(k * 0.5, k * 0.5 + 0.5, 90) for k in range(8)]
    cleaned, report = clean_transcription(_midi(spec, pitch=40))
    assert len(cleaned.instruments[0].notes) == 8
    assert report["runs_merged"] == 0


def _repeated_notes(count=16, step=0.25, length=0.2, pitch=40):
    spec = [(k * step, k * step + length, 100) for k in range(count)]
    return _midi(spec, pitch=pitch)


def test_repeated_bass_notes_keep_their_rhythm():
    from korshmer.midi.transcription_cleanup import options_for_cleanup

    cleaned, report = clean_transcription(
        _repeated_notes(), options=options_for_cleanup("bass.wav")
    )

    assert len(cleaned.instruments[0].notes) == 16
    assert "runs_merged" not in report


def test_other_stems_still_merge_retriggered_runs():
    from korshmer.midi.transcription_cleanup import options_for_cleanup

    cleaned, _ = clean_transcription(
        _repeated_notes(), options=options_for_cleanup("guitar.wav")
    )

    assert len(cleaned.instruments[0].notes) < 16
