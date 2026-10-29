import pretty_midi
import pytest

from korshmer.midi_algorithms import apply_groove_template


def _make_midi(bpm=120.0, note_count=16):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)
    sixteenth = 60 / bpm / 4
    for i in range(note_count):
        start = i * sixteenth
        dur = sixteenth * 0.8
        inst.notes.append(
            pretty_midi.Note(
                velocity=80, pitch=60, start=start, end=start + dur
            )
        )
    pm.instruments.append(inst)
    return pm


class TestGrooveTemplateDuration:
    @pytest.mark.parametrize(
        "template", ["MPC Swing", "Laid Back", "Rushing", "Drunken"]
    )
    def test_note_duration_preserved(self, template):
        pm = _make_midi()
        original_durations = [
            n.end - n.start for inst in pm.instruments for n in inst.notes
        ]
        result = apply_groove_template(pm, template=template, bpm=120.0)
        for inst in result.instruments:
            for i, note in enumerate(inst.notes):
                new_dur = note.end - note.start
                assert abs(new_dur - original_durations[i]) < 1e-9, (
                    f"Note {i} duration changed from {original_durations[i]:.6f} "
                    f"to {new_dur:.6f} with template '{template}'"
                )

    @pytest.mark.parametrize("template", ["MPC Swing", "Drunken"])
    def test_end_never_before_start(self, template):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        sixteenth = 60 / 120.0 / 4
        for i in range(32):
            start = i * sixteenth
            inst.notes.append(
                pretty_midi.Note(
                    velocity=80, pitch=60, start=start, end=start + 0.005
                )
            )
        pm.instruments.append(inst)
        result = apply_groove_template(pm, template=template, bpm=120.0)
        for inst in result.instruments:
            for i, note in enumerate(inst.notes):
                assert note.end >= note.start, (
                    f"Note {i}: end ({note.end}) < start ({note.start}) "
                    f"with template '{template}'"
                )
