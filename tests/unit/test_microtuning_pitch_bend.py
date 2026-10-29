import io

import pretty_midi

from korshmer.midi_algorithms import apply_microtuning


def _make_midi_with_notes(count=5):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i in range(count):
        inst.notes.append(
            pretty_midi.Note(
                velocity=80, pitch=60 + i, start=i * 0.5, end=i * 0.5 + 0.4
            )
        )
    pm.instruments.append(inst)
    return pm


class TestMicrotuningPitchBend:
    def test_pitch_bends_in_signed_range(self):
        pm = _make_midi_with_notes(20)
        result = apply_microtuning(pm, amount=3000)
        for inst in result.instruments:
            for pb in inst.pitch_bends:
                assert -8192 <= pb.pitch <= 8191, (
                    f"Pitch bend {pb.pitch} outside signed range [-8192, 8191]"
                )

    def test_write_does_not_crash(self):
        pm = _make_midi_with_notes(10)
        result = apply_microtuning(pm, amount=3000)
        buf = io.BytesIO()
        result.write(buf)
        buf.seek(0)
        reloaded = pretty_midi.PrettyMIDI(buf)
        assert len(reloaded.instruments[0].pitch_bends) == 10

    def test_bends_centered_near_zero(self):
        pm = _make_midi_with_notes(200)
        result = apply_microtuning(pm, amount=2000)
        bends = [
            pb.pitch for inst in result.instruments for pb in inst.pitch_bends
        ]
        mean_bend = sum(bends) / len(bends)
        assert abs(mean_bend) < 500, (
            f"Mean bend {mean_bend} too far from 0; values may be offset"
        )
