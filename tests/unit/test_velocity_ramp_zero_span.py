import pretty_midi

from korshmer.midi_algorithms import apply_velocity_ramp


class TestVelocityRampZeroSpan:
    def test_single_zero_duration_note(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.0)
        )
        pm.instruments.append(inst)
        result = apply_velocity_ramp(pm, start_v=40, end_v=127)
        assert result.instruments[0].notes[0].velocity == 40

    def test_multiple_notes_same_time(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        for pitch in [60, 64, 67]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=1.0, end=1.0)
            )
        pm.instruments.append(inst)
        result = apply_velocity_ramp(pm, start_v=50, end_v=100)
        for note in result.instruments[0].notes:
            assert note.velocity == 50

    def test_normal_notes_still_ramp(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        )
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=64, start=1.0, end=2.0)
        )
        pm.instruments.append(inst)
        result = apply_velocity_ramp(pm, start_v=40, end_v=120)
        assert result.instruments[0].notes[0].velocity == 40
        assert result.instruments[0].notes[1].velocity == 80
