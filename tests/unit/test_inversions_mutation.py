import pretty_midi

from korshmer.midi_algorithms import apply_smart_inversions


def _make_midi_with_chord(preceding_pitch, chord_pitches, chord_start=1.0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=80, pitch=preceding_pitch, start=0.0, end=0.5)
    )
    for p in chord_pitches:
        inst.notes.append(
            pretty_midi.Note(
                velocity=80, pitch=p, start=chord_start, end=chord_start + 0.5
            )
        )
    pm.instruments.append(inst)
    return pm


def _get_chord_pitches(pm, chord_start=1.0):
    notes = sorted(
        [
            n
            for n in pm.instruments[0].notes
            if abs(n.start - chord_start) < 0.01
        ],
        key=lambda n: n.pitch,
    )
    return [n.pitch for n in notes]


class TestSmartInversionsNoMutation:
    def test_selects_closest_inversion_to_target(self):
        pm = _make_midi_with_chord(70, [60, 64, 67])
        apply_smart_inversions(pm)
        pitches = _get_chord_pitches(pm)
        assert pitches == [67, 72, 76]

    def test_root_position_kept_when_closest(self):
        pm = _make_midi_with_chord(63, [60, 64, 67])
        apply_smart_inversions(pm)
        pitches = _get_chord_pitches(pm)
        assert pitches == [60, 64, 67]

    def test_first_inversion_selected(self):
        pm = _make_midi_with_chord(67, [60, 64, 67])
        apply_smart_inversions(pm)
        pitches = _get_chord_pitches(pm)
        assert pitches == [64, 67, 72]
