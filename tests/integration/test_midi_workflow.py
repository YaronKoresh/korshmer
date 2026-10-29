import numpy as np
import pretty_midi
import pytest


@pytest.mark.integration
class TestMIDIImportExportWorkflow:
    def test_midi_import_and_export(self, tmp_path):

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(10):
            note = pretty_midi.Note(
                velocity=100,
                pitch=60 + i,
                start=i * 0.5,
                end=(i + 1) * 0.5,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        input_file = tmp_path / "input.mid"
        midi.write(str(input_file))

        loaded_midi = pretty_midi.PrettyMIDI(str(input_file))
        output_file = tmp_path / "output.mid"
        loaded_midi.write(str(output_file))

        assert input_file.exists()
        assert output_file.exists()
        assert input_file.stat().st_size > 0
        assert output_file.stat().st_size > 0

        final_midi = pretty_midi.PrettyMIDI(str(output_file))
        assert len(final_midi.instruments) == 1
        assert len(final_midi.instruments[0].notes) == 10

    def test_midi_transposition_workflow(self, tmp_path):
        from korshmer.midi import transpose_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        original_pitches = [60, 62, 64, 65, 67]
        for i, pitch in enumerate(original_pitches):
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        transposed = transpose_midi(midi, semitones=2)

        assert len(transposed.instruments[0].notes) == len(original_pitches)
        for orig_pitch, note in zip(
            original_pitches, transposed.instruments[0].notes, strict=False
        ):
            assert note.pitch == orig_pitch + 2

    def test_midi_quantization_workflow(self, tmp_path):
        from korshmer.midi import export_to_json

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            jitter = 0.01 * (np.random.rand() - 0.5)
            note = pretty_midi.Note(
                velocity=100,
                pitch=60,
                start=i * 0.5 + jitter,
                end=(i + 1) * 0.5 + jitter,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        json_data = export_to_json(midi)
        assert isinstance(json_data, str)
        assert "notes" in json_data or "instruments" in json_data

    def test_midi_velocity_editing_workflow(self, tmp_path):
        from korshmer.midi import (
            compress_velocity,
            lofi_velocity,
            randomize_velocity,
        )

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=64 + i * 10, pitch=60, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        compressed = compress_velocity(midi, ratio=0.5)
        assert len(compressed.instruments[0].notes) == 5

        lofi = lofi_velocity(midi)
        assert len(lofi.instruments[0].notes) == 5

        randomized = randomize_velocity(midi, amount=20)
        assert len(randomized.instruments[0].notes) == 5

    def test_midi_duration_editing_workflow(self, tmp_path):
        from korshmer.midi import generate_legato, remove_short_notes

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        durations = [0.1, 0.05, 0.3, 0.02, 0.5]
        for i, duration in enumerate(durations):
            note = pretty_midi.Note(
                velocity=100, pitch=60, start=i * 0.6, end=i * 0.6 + duration
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        filtered = remove_short_notes(midi, min_duration=0.08)

        assert len(filtered.instruments[0].notes) < len(durations)

        legato = generate_legato(midi)
        assert len(legato.instruments[0].notes) == len(durations)

    def test_midi_pitch_range_workflow(self, tmp_path):
        from korshmer.midi import limit_pitch_range

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        pitches = [40, 50, 60, 70, 80, 90]
        for i, pitch in enumerate(pitches):
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        limited = limit_pitch_range(midi, min_pitch=55, max_pitch=75)
        for note in limited.instruments[0].notes:
            assert 55 <= note.pitch <= 75

    def test_midi_rhythm_extraction_workflow(self, tmp_path):
        from korshmer.midi import extract_rhythm

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        rhythm_times = [0.0, 0.5, 0.75, 1.0, 1.5]
        for time in rhythm_times:
            note = pretty_midi.Note(
                velocity=100, pitch=60, start=time, end=time + 0.1
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        rhythm = extract_rhythm(midi)
        assert rhythm is not None

    def test_midi_octave_manipulation_workflow(self, tmp_path):
        from korshmer.midi import double_octave

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=100, pitch=60 + i, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        doubled = double_octave(midi)

        assert len(doubled.instruments[0].notes) == 10

    def test_midi_time_stretch_workflow(self, tmp_path):
        from korshmer.midi import time_stretch

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=100, pitch=60, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        stretched = time_stretch(midi, factor=2.0)

        original_end = midi.instruments[0].notes[-1].end
        stretched_end = stretched.instruments[0].notes[-1].end
        assert stretched_end > original_end * 1.8

    def test_midi_musical_transformations_workflow(self, tmp_path):
        from korshmer.midi import apply_mirror, apply_retrograde

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=100, pitch=60 + i * 2, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        retrograde = apply_retrograde(midi)
        assert len(retrograde.instruments[0].notes) == 5

        mirrored = apply_mirror(midi)
        assert len(mirrored.instruments[0].notes) == 5

    def test_midi_articulation_workflow(self, tmp_path):
        from korshmer.midi import apply_crescendo, apply_strumming

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=80, pitch=60 + i, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        crescendo = apply_crescendo(midi)

        velocities = [note.velocity for note in crescendo.instruments[0].notes]
        assert velocities[-1] > velocities[0]

        strummed = apply_strumming(midi)
        assert len(strummed.instruments[0].notes) == 5

    def test_midi_voice_leading_workflow(self, tmp_path):
        from korshmer.midi import apply_drop2, force_monophonic

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        chord_pitches = [60, 64, 67, 72]
        for pitch in chord_pitches:
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=0.0, end=1.0
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        drop2 = apply_drop2(midi)
        assert len(drop2.instruments[0].notes) == 4

        mono = force_monophonic(midi)

        assert len(mono.instruments[0].notes) <= 4

    def test_midi_cleanup_workflow(self, tmp_path):
        from korshmer.midi import remove_empty_bars

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        note1 = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5)
        note2 = pretty_midi.Note(velocity=100, pitch=62, start=4.0, end=4.5)
        instrument.notes.append(note1)
        instrument.notes.append(note2)

        midi.instruments.append(instrument)

        cleaned = remove_empty_bars(midi)

        assert cleaned.instruments[0].notes[1].start < note2.start


@pytest.mark.integration
class TestMIDIPianoRollIntegration:
    def test_piano_roll_image_generation(self, tmp_path):
        from korshmer.midi import generate_piano_roll_image

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(10):
            note = pretty_midi.Note(
                velocity=100,
                pitch=60 + (i % 12),
                start=i * 0.5,
                end=(i + 1) * 0.5,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        output_path = tmp_path / "piano_roll.png"
        generate_piano_roll_image(midi, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0
