import pretty_midi

from korshmer.audio_processing.music_generation import MusicGenerator


class TestHarmonyVelocityClamp:
    def test_low_velocity_note_no_negative(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=5, pitch=60, start=0.0, end=1.0)
        )
        midi.instruments.append(inst)
        gen = MusicGenerator()
        result = gen.generate_harmony(midi, interval=3, key=60, scale="major")
        for inst in result.instruments:
            for note in inst.notes:
                assert note.velocity >= 1, (
                    f"Harmony velocity {note.velocity} is below 1"
                )

    def test_zero_velocity_note_clamped(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=0, pitch=60, start=0.0, end=1.0)
        )
        midi.instruments.append(inst)
        gen = MusicGenerator()
        result = gen.generate_harmony(midi, interval=3, key=60, scale="major")
        for inst in result.instruments:
            for note in inst.notes:
                assert note.velocity >= 1, (
                    f"Harmony velocity {note.velocity} is below 1"
                )

    def test_normal_velocity_reduced_by_10(self):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        )
        midi.instruments.append(inst)
        gen = MusicGenerator()
        result = gen.generate_harmony(midi, interval=3, key=60, scale="major")
        for inst in result.instruments:
            for note in inst.notes:
                assert note.velocity == 70
