import pretty_midi

from korshmer.midi_algorithms import generate_smart_harmony


class TestSmartHarmonyOctavesIndex:
    def test_low_note_octaves_no_wraparound(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=36, start=0.0, end=1.0)
        )
        pm.instruments.append(inst)
        result = generate_smart_harmony(
            pm, root_note_idx=0, scale_name="Major", harmony_type="Octaves"
        )
        harmony_inst = result.instruments[-1]
        for note in harmony_inst.notes:
            assert note.pitch <= 36, (
                f"Octave harmony note {note.pitch} should not be above "
                f"original pitch 36"
            )

    def test_mid_note_octaves_produces_lower(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        )
        pm.instruments.append(inst)
        result = generate_smart_harmony(
            pm, root_note_idx=0, scale_name="Major", harmony_type="Octaves"
        )
        harmony_inst = result.instruments[-1]
        for note in harmony_inst.notes:
            assert note.pitch <= 60, (
                f"Octave harmony note {note.pitch} should not be above "
                f"original pitch 60"
            )
